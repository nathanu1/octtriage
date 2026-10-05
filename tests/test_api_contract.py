from __future__ import annotations

import asyncio
import io
import threading
from types import SimpleNamespace

import httpx
from fastapi.testclient import TestClient
from PIL import Image

from api.main import LazyPredictor, create_app
from octtriage.serving.audit import AuditStore
from octtriage.serving.errors import ModelUnavailableError
from octtriage.settings import Settings


class FakePredictor:
    ready = True

    def predict(self, payload: bytes, filename: str):
        image = Image.new("RGB", (8, 8), color=(40, 40, 40))
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        return SimpleNamespace(
            triage_tier="URGENT",
            confidence=0.91,
            findings=["CNV"],
            class_probabilities={"CNV": 0.91, "DME": 0.04, "DRUSEN": 0.03, "NORMAL": 0.02},
            escalation_reasons=[],
            requires_repeat_acquisition=False,
            quality={"passed": True, "reasons": [], "metrics": {"dynamic_range": 120.0}},
            model_version="test-model-v1",
            request_sha256="a" * 64,
            saliency_png=buffer.getvalue(),
            saliency_class="CNV",
            thumbnail_png=buffer.getvalue(),
            disclaimer="Clinical decision-support prototype. This output is not a diagnosis.",
        )


def settings_for(tmp_path, **overrides) -> Settings:
    return Settings(
        project_root=tmp_path,
        runtime_dir=tmp_path / "runtime",
        database_path=tmp_path / "runtime" / "audit.db",
        checkpoint_path=tmp_path / "best.pt",
        **overrides,
    )


def client_for(tmp_path, predictor=None, **overrides) -> TestClient:
    app_settings = settings_for(tmp_path, **overrides)
    store = AuditStore(app_settings.database_path)
    return TestClient(create_app(predictor or FakePredictor(), store, app_settings))


def submit(client: TestClient, payload: bytes = b"valid-test-payload", **data):
    return client.post(
        "/v1/triage", data=data, files={"file": ("scan.png", payload, "image/png")}
    )


def test_api_contract_and_override_are_audited(tmp_path) -> None:
    client = client_for(tmp_path)
    health = client.get("/health").json()
    assert health["model_ready"] is True
    assert "not a diagnosis" in health["disclaimer"]
    response = submit(client, study_id="DEID-001")
    assert response.status_code == 200
    body = response.json()
    assert body["triage_tier"] == "URGENT"
    assert body["requires_clinician_review"] is True
    assert "not a diagnosis" in body["disclaimer"]
    assert body["model_version"] == "test-model-v1"
    assert body["saliency_map_url"]
    assert body["saliency_class"] == "CNV"

    worklist = client.get("/v1/worklist").json()["items"]
    assert len(worklist) == 1
    assert worklist[0]["audit_id"] == body["audit_id"]

    override = client.post(
        "/v1/overrides",
        json={
            "audit_id": body["audit_id"],
            "new_tier": "SEMI_URGENT",
            "clinician_id": "clinician-test",
            "reason": "Reviewed the source scan and prior imaging.",
        },
    )
    assert override.status_code == 200
    assert override.json()["original_tier"] == "URGENT"
    assert client.get("/v1/worklist").json()["items"][0]["effective_tier"] == "SEMI_URGENT"


def test_worklist_exposes_media_urls_not_server_paths(tmp_path) -> None:
    client = client_for(tmp_path)
    body = submit(client).json()
    item = client.get("/v1/worklist").json()["items"][0]
    assert "thumbnail_path" not in item
    assert "saliency_path" not in item
    assert item["thumbnail_url"] == body["thumbnail_url"]
    assert item["saliency_map_url"] == body["saliency_map_url"]
    assert client.get(item["thumbnail_url"]).headers["content-type"] == "image/png"


def test_prediction_detail_keeps_original_decision_and_override_history(tmp_path) -> None:
    client = client_for(tmp_path)
    audit_id = submit(client).json()["audit_id"]
    for tier in ["SEMI_URGENT", "ROUTINE"]:
        client.post(
            "/v1/overrides",
            json={
                "audit_id": audit_id,
                "new_tier": tier,
                "clinician_id": "clinician-test",
                "reason": f"Reviewed; reprioritized to {tier}.",
            },
        )
    detail = client.get(f"/v1/predictions/{audit_id}").json()
    assert detail["prediction"]["triage_tier"] == "URGENT"
    assert detail["effective_tier"] == "ROUTINE"
    assert [item["original_tier"] for item in detail["overrides"]] == ["URGENT", "SEMI_URGENT"]
    assert [item["new_tier"] for item in detail["overrides"]] == ["SEMI_URGENT", "ROUTINE"]
    assert client.get("/v1/predictions/unknown").status_code == 404


def test_empty_upload_is_rejected(tmp_path) -> None:
    response = submit(client_for(tmp_path), payload=b"")
    assert response.status_code == 422
    assert response.json()["decision_support_only"] is True
    assert "not a diagnosis" in response.json()["disclaimer"]


def test_oversized_upload_is_rejected(tmp_path) -> None:
    response = submit(client_for(tmp_path, max_upload_bytes=8), payload=b"x" * 9)
    assert response.status_code == 413


def test_missing_model_returns_503_and_records_nothing(tmp_path) -> None:
    app_settings = settings_for(tmp_path)
    store = AuditStore(app_settings.database_path)
    client = TestClient(create_app(LazyPredictor(app_settings), store, app_settings))
    assert client.get("/health").json()["model_ready"] is False
    response = submit(client)
    assert response.status_code == 503
    assert response.json()["decision_support_only"] is True
    assert store.list_worklist() == []


def test_model_load_failure_returns_503(tmp_path) -> None:
    class BrokenPredictor:
        ready = True

        def predict(self, payload: bytes, filename: str):
            raise ModelUnavailableError("Model checkpoint could not be loaded: corrupt")

    response = submit(client_for(tmp_path, predictor=BrokenPredictor()))
    assert response.status_code == 503
    assert "could not be loaded" in response.json()["detail"]


def test_health_stays_responsive_while_inference_runs(tmp_path) -> None:
    started = threading.Event()
    release = threading.Event()
    finished = threading.Event()

    class BlockingPredictor(FakePredictor):
        def predict(self, payload: bytes, filename: str):
            started.set()
            release.wait(timeout=5)
            finished.set()
            return super().predict(payload, filename)

    app_settings = settings_for(tmp_path)
    app = create_app(BlockingPredictor(), AuditStore(app_settings.database_path), app_settings)

    async def exercise() -> tuple[int, bool, int]:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            triage = asyncio.create_task(
                client.post("/v1/triage", files={"file": ("scan.png", b"x", "image/png")})
            )
            while not started.is_set():
                await asyncio.sleep(0.01)
            health = await client.get("/health")
            answered_during_inference = not finished.is_set()
            release.set()
            return health.status_code, answered_during_inference, (await triage).status_code

    assert asyncio.run(exercise()) == (200, True, 200)
