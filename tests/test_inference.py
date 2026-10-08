from __future__ import annotations

import io

import numpy as np
import pytest
import torch
from PIL import Image

from octtriage.models.classifier import build_classifier
from octtriage.models.registry import register_model
from octtriage.serving.errors import ModelUnavailableError
from octtriage.serving.inference import ModelInferenceService
from octtriage.settings import Settings

CLASSES = ["CNV", "DME", "DRUSEN", "NORMAL"]


def png_payload(image: Image.Image) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


@pytest.fixture(scope="module")
def checkpoint(tmp_path_factory):
    torch.manual_seed(0)
    model = build_classifier(len(CLASSES), backbone="efficientnet_b0", pretrained=False)
    path = tmp_path_factory.mktemp("model") / "best.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "classes": CLASSES,
            "image_size": 224,
            "backbone": "efficientnet_b0",
            "model_version": "unit-test-model",
            "epoch": 1,
            "validation_metrics": {"urgent_recall": 1.0},
        },
        path,
    )
    return path


def service_for(checkpoint_path, **overrides) -> ModelInferenceService:
    return ModelInferenceService(
        Settings(checkpoint_path=checkpoint_path, **overrides), device="cpu"
    )


def test_quality_passing_scan_returns_decision_and_saliency(checkpoint, oct_image) -> None:
    result = service_for(checkpoint).predict(png_payload(oct_image()), "scan.png")
    assert result.quality["passed"] is True
    assert result.triage_tier in {"URGENT", "SEMI_URGENT", "ROUTINE"}
    assert result.model_version == "unit-test-model"
    assert set(result.class_probabilities) == set(CLASSES)
    assert result.saliency_class == result.findings[0]
    saliency = Image.open(io.BytesIO(result.saliency_png))
    assert saliency.size == (320, 240)


def test_saliency_explains_the_finding_that_drives_the_tier(checkpoint, oct_image) -> None:
    service = service_for(checkpoint)
    # NORMAL is the argmax, but the CNV candidate probability promotes the case to URGENT.
    service._view_probabilities = lambda *_args: np.asarray(
        [[[0.30, 0.02, 0.03, 0.65], [0.28, 0.02, 0.03, 0.67]]]
    )
    explained: list[int] = []
    service._saliency = lambda _bundle, _transform, _frame, index: explained.append(index) or b""
    result = service.predict(png_payload(oct_image()), "scan.png")
    assert result.triage_tier == "URGENT"
    assert result.findings[0] == "CNV"
    assert result.saliency_class == "CNV"
    assert explained == [CLASSES.index("CNV")]


def test_quality_failure_is_triaged_without_a_model(tmp_path) -> None:
    service = service_for(tmp_path / "missing.pt")
    result = service.predict(png_payload(Image.new("L", (256, 256), color=0)), "blank.png")
    assert result.triage_tier == "URGENT"
    assert result.findings == ["UNINTERPRETABLE_SCAN"]
    assert result.requires_repeat_acquisition is True
    assert result.saliency_png is None
    assert result.saliency_class is None


def test_missing_or_corrupt_checkpoint_reports_model_unavailable(tmp_path, oct_image) -> None:
    payload = png_payload(oct_image())
    with pytest.raises(ModelUnavailableError, match="No trained checkpoint"):
        service_for(tmp_path / "missing.pt").predict(payload, "scan.png")
    corrupt = tmp_path / "corrupt.pt"
    corrupt.write_bytes(b"not a checkpoint")
    with pytest.raises(ModelUnavailableError, match="could not be loaded"):
        service_for(corrupt).predict(payload, "scan.png")


def test_registered_checkpoint_must_match_its_recorded_hash(
    checkpoint, tmp_path, oct_image
) -> None:
    payload = png_payload(oct_image())
    manifest = tmp_path / "manifest.csv"
    manifest.write_text("path,label,patient_id,split\n", encoding="utf-8")
    registry = tmp_path / "registry.json"
    register_model(
        registry,
        model_version="unit-test-model",
        checkpoint_path=checkpoint,
        metrics={},
        training_manifest=manifest,
    )
    result = service_for(checkpoint, model_registry_path=registry).predict(payload, "scan.png")
    assert result.model_version == "unit-test-model"

    tampered = tmp_path / "tampered.pt"
    tampered.write_bytes(checkpoint.read_bytes() + b"\0")
    service = service_for(tampered, model_registry_path=registry)
    with pytest.raises(ModelUnavailableError, match="does not match the SHA-256"):
        service.predict(payload, "scan.png")


def test_checkpoint_temperature_softens_served_confidence(checkpoint, tmp_path, oct_image) -> None:
    payload = png_payload(oct_image())
    stored = torch.load(checkpoint, weights_only=True)
    softened = tmp_path / "softened.pt"
    torch.save({**stored, "temperature": 5.0}, softened)
    raw = service_for(checkpoint).predict(payload, "scan.png").class_probabilities
    calibrated = service_for(softened).predict(payload, "scan.png").class_probabilities
    assert max(calibrated.values()) < max(raw.values())

    invalid = tmp_path / "invalid.pt"
    torch.save({**stored, "temperature": 0.0}, invalid)
    with pytest.raises(ModelUnavailableError, match="temperature must be positive"):
        service_for(invalid).predict(payload, "scan.png")
