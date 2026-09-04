from __future__ import annotations

import io
from types import SimpleNamespace

from fastapi.testclient import TestClient
from PIL import Image

from api.main import create_app
from octtriage.serving.audit import AuditStore
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
            thumbnail_png=buffer.getvalue(),
            disclaimer="Clinical decision-support prototype. This output is not a diagnosis.",
        )


def test_api_contract_and_override_are_audited(tmp_path) -> None:
    app_settings = Settings(
        project_root=tmp_path,
        runtime_dir=tmp_path / "runtime",
        database_path=tmp_path / "runtime" / "audit.db",
        checkpoint_path=tmp_path / "best.pt",
    )
    store = AuditStore(app_settings.database_path)
    client = TestClient(create_app(FakePredictor(), store, app_settings))
    health = client.get("/health").json()
    assert health["model_ready"] is True
    assert "not a diagnosis" in health["disclaimer"]
    response = client.post(
        "/v1/triage",
        data={"study_id": "DEID-001"},
        files={"file": ("scan.png", b"valid-test-payload", "image/png")},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["triage_tier"] == "URGENT"
    assert body["requires_clinician_review"] is True
    assert "not a diagnosis" in body["disclaimer"]
    assert body["model_version"] == "test-model-v1"
    assert body["saliency_map_url"]

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


def test_empty_upload_is_rejected(tmp_path) -> None:
    app_settings = Settings(
        project_root=tmp_path,
        runtime_dir=tmp_path / "runtime",
        database_path=tmp_path / "runtime" / "audit.db",
        checkpoint_path=tmp_path / "best.pt",
    )
    client = TestClient(
        create_app(FakePredictor(), AuditStore(app_settings.database_path), app_settings)
    )
    response = client.post("/v1/triage", files={"file": ("scan.png", b"", "image/png")})
    assert response.status_code == 422
    assert response.json()["decision_support_only"] is True
    assert "not a diagnosis" in response.json()["disclaimer"]
