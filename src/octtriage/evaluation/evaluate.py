from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from octtriage.config import load_yaml
from octtriage.data.dataset import OCTManifestDataset
from octtriage.data.manifest import assert_no_patient_leakage
from octtriage.data.transforms import inference_transform
from octtriage.evaluation.metrics import classification_metrics
from octtriage.models.classifier import load_bundle
from octtriage.settings import settings


@torch.inference_mode()
def collect_probabilities(model, loader, device) -> tuple[np.ndarray, np.ndarray]:
    targets: list[int] = []
    probabilities: list[np.ndarray] = []
    for inputs, batch_targets in loader:
        outputs = torch.softmax(model(inputs.to(device)), dim=1).cpu().numpy()
        probabilities.append(outputs)
        targets.extend(batch_targets.numpy().tolist())
    return np.asarray(targets), np.concatenate(probabilities)


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
    dataset = OCTManifestDataset(
        manifest, "test", inference_transform(bundle.image_size), bundle.classes
    )
    loader = DataLoader(dataset, batch_size=32, shuffle=False)
    y_true, probabilities = collect_probabilities(bundle.model, loader, device)
    model_config = load_yaml(settings.model_config_path)
    metrics = classification_metrics(
        y_true, probabilities, bundle.classes, set(model_config["urgent_classes"])
    )
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
