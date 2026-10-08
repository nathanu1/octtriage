from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from octtriage.config import load_yaml
from octtriage.data.dataset import OCTManifestDataset
from octtriage.data.io import read_bscan
from octtriage.data.manifest import assert_no_patient_leakage
from octtriage.data.quality import QualityAssessment, assess_quality
from octtriage.data.transforms import inference_transform
from octtriage.evaluation.calibration import softmax
from octtriage.evaluation.metrics import (
    classification_metrics,
    expected_calibration_error,
    patient_bootstrap_intervals,
    triage_policy_metrics,
)
from octtriage.models.classifier import ModelBundle, load_bundle
from octtriage.models.registry import sha256_file
from octtriage.settings import settings
from octtriage.triage.aggregation import decide_from_views
from octtriage.triage.engine import TriageDecision, TriageEngine


class PolicyEvaluationDataset(Dataset):
    """Yield each test B-scan as serving sees it: original and flipped views plus quality."""

    def __init__(
        self,
        manifest: pd.DataFrame,
        split: str,
        transform,
        classes: list[str],
        quality_rules: dict[str, object],
    ) -> None:
        self.base = OCTManifestDataset(manifest, split, transform, classes)
        self.quality_rules = quality_rules

    @property
    def frame(self) -> pd.DataFrame:
        return self.base.frame

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, index: int):
        row = self.base.frame.iloc[index]
        image = read_bscan(row["path"])
        quality = assess_quality(image, self.quality_rules)
        views = torch.stack(
            [
                self.base.transform(image),
                self.base.transform(image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)),
            ]
        )
        target = self.base.class_to_idx[row["label"]]
        return views, target, quality.passed, "|".join(quality.reasons)


@torch.inference_mode()
def collect_view_logits(
    model, loader, device
) -> tuple[np.ndarray, np.ndarray, list[bool], list[list[str]]]:
    """Return targets, logits shaped (images, views, classes), and quality verdicts."""
    targets: list[int] = []
    logits: list[np.ndarray] = []
    quality_passed: list[bool] = []
    quality_reasons: list[list[str]] = []
    for views, batch_targets, batch_passed, batch_reasons in loader:
        batch, view_count = views.shape[:2]
        outputs = model(views.flatten(0, 1).to(device)).cpu().numpy()
        logits.append(outputs.reshape(batch, view_count, -1))
        targets.extend(batch_targets.numpy().tolist())
        quality_passed.extend(bool(item) for item in batch_passed)
        quality_reasons.extend(
            [reason for reason in item.split("|") if reason] for item in batch_reasons
        )
    return np.asarray(targets), np.concatenate(logits), quality_passed, quality_reasons


@dataclass(frozen=True)
class SplitOutputs:
    """Model outputs for one manifest split, in manifest order."""

    targets: np.ndarray
    view_logits: np.ndarray
    view_probabilities: np.ndarray
    quality_passed: list[bool]
    quality_reasons: list[list[str]]
    patient_ids: np.ndarray


def run_split(
    bundle: ModelBundle, manifest: pd.DataFrame, split: str, device: torch.device
) -> SplitOutputs:
    dataset = PolicyEvaluationDataset(
        manifest,
        split,
        inference_transform(bundle.image_size),
        bundle.classes,
        load_yaml(settings.quality_rules_path),
    )
    loader = DataLoader(dataset, batch_size=32, shuffle=False)
    targets, view_logits, quality_passed, quality_reasons = collect_view_logits(
        bundle.model, loader, device
    )
    return SplitOutputs(
        targets=targets,
        view_logits=view_logits,
        # Every probability uses the checkpoint's fitted temperature, as serving does.
        view_probabilities=softmax(view_logits, bundle.temperature),
        quality_passed=quality_passed,
        quality_reasons=quality_reasons,
        patient_ids=dataset.frame["patient_id"].astype(str).to_numpy(),
    )


def policy_decisions(
    engine: TriageEngine,
    outputs: SplitOutputs,
    classes: list[str],
    non_pathology_classes: set[str],
) -> list[TriageDecision]:
    """Apply the same quality gate, aggregation, and rules as the serving path to each image."""
    return [
        decide_from_views(
            engine,
            views[np.newaxis],
            classes,
            non_pathology_classes,
            QualityAssessment(passed=passed, reasons=reasons, metrics={}),
        )[0]
        for views, passed, reasons in zip(
            outputs.view_probabilities,
            outputs.quality_passed,
            outputs.quality_reasons,
            strict=True,
        )
    ]


def plot_confusion(matrix: list[list[int]], classes: list[str], path: Path) -> None:
    values = np.asarray(matrix)
    fig, axis = plt.subplots(figsize=(6, 5))
    axis.imshow(values, cmap="Blues")
    axis.set(xticks=range(len(classes)), yticks=range(len(classes)), xticklabels=classes,
             yticklabels=classes, xlabel="Predicted", ylabel="True", title="Test confusion matrix")
    for row in range(values.shape[0]):
        for column in range(values.shape[1]):
            axis.text(column, row, str(values[row, column]), ha="center", va="center")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_calibration(y_true: np.ndarray, probabilities: np.ndarray, path: Path) -> None:
    confidence = probabilities.max(axis=1)
    correct = probabilities.argmax(axis=1) == y_true
    edges = np.linspace(0.0, 1.0, 11)
    xs: list[float] = []
    ys: list[float] = []
    for lower, upper in zip(edges[:-1], edges[1:], strict=True):
        mask = (confidence > lower) & (confidence <= upper)
        if mask.any():
            xs.append(float(confidence[mask].mean()))
            ys.append(float(correct[mask].mean()))
    fig, axis = plt.subplots(figsize=(5, 5))
    axis.plot([0, 1], [0, 1], "--", color="gray", label="Perfect calibration")
    axis.plot(xs, ys, marker="o", label="Model")
    axis.set(xlim=(0, 1), ylim=(0, 1), xlabel="Confidence", ylabel="Observed accuracy",
             title="Reliability diagram")
    axis.legend()
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def evaluate(
    manifest_path: Path,
    checkpoint_path: Path,
    output_dir: Path,
    device_name: str,
) -> dict[str, object]:
    device = torch.device(device_name)
    bundle = load_bundle(checkpoint_path, device)
    manifest = pd.read_csv(manifest_path)
    assert_no_patient_leakage(manifest)
    model_config = load_yaml(settings.model_config_path)
    urgent_classes = set(model_config["urgent_classes"])
    non_pathology_classes = set(model_config.get("non_pathology_classes", ["NORMAL"]))
    engine = TriageEngine(settings.triage_rules_path)
    outputs = run_split(bundle, manifest, "test", device)
    y_true = outputs.targets

    # Classifier metrics use the un-augmented view of every image, quality gate aside.
    probabilities = outputs.view_probabilities[:, 0, :]
    metrics = classification_metrics(y_true, probabilities, bundle.classes, urgent_classes)
    metrics["expected_calibration_error_uncalibrated"] = expected_calibration_error(
        y_true, softmax(outputs.view_logits[:, 0, :])
    )

    decisions = policy_decisions(engine, outputs, bundle.classes, non_pathology_classes)
    tiers = np.asarray([decision.triage_tier for decision in decisions])
    metrics["triage_policy"] = triage_policy_metrics(
        y_true,
        tiers.tolist(),
        [decision.escalation_reasons for decision in decisions],
        bundle.classes,
        urgent_classes,
    )

    # Images from one patient are correlated, so resample patients rather than images.
    true_urgent = np.isin(y_true, [bundle.classes.index(label) for label in urgent_classes])
    predicted_urgent = np.isin(
        probabilities.argmax(axis=1), [bundle.classes.index(label) for label in urgent_classes]
    )
    triaged_urgent = tiers == "URGENT"
    metrics["confidence_intervals"] = patient_bootstrap_intervals(
        outputs.patient_ids,
        {
            "accuracy": (probabilities.argmax(axis=1) == y_true, np.ones(len(y_true))),
            "urgent_recall": (true_urgent & predicted_urgent, true_urgent),
            "urgent_specificity": (~true_urgent & ~predicted_urgent, ~true_urgent),
            "urgent_tier_sensitivity": (true_urgent & triaged_urgent, true_urgent),
            "non_urgent_cases_triaged_urgent_rate": (~true_urgent & triaged_urgent, ~true_urgent),
        },
    )
    metrics["provenance"] = {
        "model_version": bundle.model_version,
        "temperature": bundle.temperature,
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "manifest_sha256": sha256_file(manifest_path),
        "triage_rules_sha256": sha256_file(settings.triage_rules_path),
        "quality_rules_sha256": sha256_file(settings.quality_rules_path),
        "split": "test",
        "images": int(len(y_true)),
        "patients": int(len(np.unique(outputs.patient_ids))),
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "test_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    plot_confusion(metrics["confusion_matrix"], bundle.classes, output_dir / "confusion_matrix.png")
    plot_calibration(y_true, probabilities, output_dir / "calibration.png")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate an OCT triage checkpoint")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=settings.project_root / "data/manifests/kermany.csv",
    )
    parser.add_argument("--checkpoint", type=Path, default=settings.checkpoint_path)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=settings.project_root / "reports/metrics",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    metrics = evaluate(args.manifest, args.checkpoint, args.output_dir, args.device)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
