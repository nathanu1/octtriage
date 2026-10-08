from __future__ import annotations

import dataclasses
import json

import pandas as pd
import pytest
import torch
import yaml

from octtriage.evaluation import evaluate as evaluate_module
from octtriage.settings import settings
from octtriage.training import train as train_module

CLASSES = ["CNV", "DME", "DRUSEN", "NORMAL"]


@pytest.fixture
def synthetic_study(tmp_path, oct_image):
    rows = []
    for class_index, label in enumerate(CLASSES):
        for patient, split in enumerate(["train", "train", "val", "test"]):
            patient_id = f"{class_index}{patient}"
            path = tmp_path / "images" / label / f"{label}-{patient_id}-1.png"
            path.parent.mkdir(parents=True, exist_ok=True)
            oct_image(seed=class_index * 10 + patient).save(path)
            rows.append(
                {"path": str(path), "label": label, "patient_id": patient_id, "split": split}
            )
    manifest = tmp_path / "manifest.csv"
    pd.DataFrame(rows).to_csv(manifest, index=False)

    config = yaml.safe_load(settings.model_config_path.read_text(encoding="utf-8"))
    config.update(
        image_size=64, batch_size=4, epochs=1, num_workers=0, pretrained_imagenet=False
    )
    config_path = tmp_path / "model.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return manifest, config_path


def test_training_checkpoint_evaluates_with_policy_metrics(
    tmp_path, synthetic_study, monkeypatch
) -> None:
    manifest, config_path = synthetic_study
    registry = tmp_path / "registry.json"
    monkeypatch.setattr(
        train_module, "settings", dataclasses.replace(settings, model_registry_path=registry)
    )
    checkpoint = train_module.train(manifest, tmp_path / "best.pt", config_path, "cpu")

    record = json.loads(registry.read_text(encoding="utf-8"))
    assert record["active_model_version"] == record["models"][0]["model_version"]
    assert len(record["models"][0]["training_manifest_sha256"]) == 64

    stored = torch.load(checkpoint, weights_only=True)
    assert stored["temperature"] >= 1.0
    assert record["models"][0]["metrics"]["temperature"] == stored["temperature"]

    metrics = evaluate_module.evaluate(manifest, checkpoint, tmp_path / "metrics", "cpu")
    assert metrics["provenance"]["temperature"] == stored["temperature"]
    assert 0.0 <= metrics["expected_calibration_error_uncalibrated"] <= 1.0
    policy = metrics["triage_policy"]
    assert policy["cases"] == len(CLASSES)
    assert sum(sum(counts.values()) for counts in policy["tier_counts_by_true_label"].values()) == 4
    assert 0.0 <= policy["urgent_tier_sensitivity"] <= 1.0
    provenance = metrics["provenance"]
    assert provenance["model_version"] == record["active_model_version"]
    assert provenance["patients"] == 4
    assert len(provenance["checkpoint_sha256"]) == 64
    for name in ["test_metrics.json", "confusion_matrix.png", "calibration.png"]:
        assert (tmp_path / "metrics" / name).is_file()
