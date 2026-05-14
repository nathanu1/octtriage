"""
Evaluation script: report metrics that go beyond plain accuracy.

Why multiple metrics?
- **Precision / recall / F1** summarize trade-offs when classes are imbalanced.
- **Confusion matrix** shows *which* classes are mistaken for which — crucial for error analysis.
- **Classification report** gives per-class precision/recall/F1 in one table.

All metrics here describe model behavior on a specific test folder — not clinical performance in the wild.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from torch import nn

from src import config
from src.dataset import get_test_loader
from src.model import build_model, load_trained_weights
from src.transforms import get_val_transforms


@torch.inference_mode()
def collect_predictions(model: nn.Module, loader, device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    ys: list[int] = []
    preds: list[int] = []
    for inputs, targets in loader:
        inputs = inputs.to(device, non_blocking=True)
        logits = model(inputs)
        batch_preds = logits.argmax(dim=1).cpu().numpy().tolist()
        preds.extend(batch_preds)
        ys.extend(targets.numpy().tolist())
    return np.array(ys), np.array(preds)


def plot_confusion(cm: np.ndarray, class_names: list[str], out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=class_names,
        yticklabels=class_names,
        ylabel="True label",
        xlabel="Predicted label",
        title="Confusion matrix (test set)",
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    thresh = cm.max() / 2.0 if cm.size else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], "d"), ha="center", va="center", color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate OCT triage checkpoint (educational).")
    parser.add_argument("--checkpoint", type=str, default=str(config.CHECKPOINT_PATH))
    parser.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    parser.add_argument("--num-workers", type=int, default=config.NUM_WORKERS)
    parser.add_argument("--device", type=str, default=config.DEVICE)
    args = parser.parse_args()

    device = torch.device(args.device)
    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.is_file():
        raise FileNotFoundError(
            f"No checkpoint at {ckpt_path}. Train first with `python -m src.train` "
            f"or pass --checkpoint explicitly."
        )

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    classes: list[str] = list(ckpt.get("classes", list(config.CLASSES)))
    num_classes = len(classes)

    val_tf = get_val_transforms(config.IMG_SIZE)
    test_loader, test_ds = get_test_loader(
        transform=val_tf,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    if list(test_ds.classes) != classes:
        print(
            "Warning: test set class names/order differ from checkpoint metadata.\n"
            f"  Checkpoint: {classes}\n"
            f"  Test set:   {test_ds.classes}"
        )

    model = build_model(num_classes=num_classes, pretrained=False)
    load_trained_weights(model, ckpt_path, map_location=device)
    model.to(device)

    y_true, y_pred = collect_predictions(model, test_loader, device)

    acc = accuracy_score(y_true, y_pred)
    precision_macro = precision_score(y_true, y_pred, average="macro", zero_division=0)
    recall_macro = recall_score(y_true, y_pred, average="macro", zero_division=0)
    f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)

    report = classification_report(y_true, y_pred, target_names=classes, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(classes))))

    print("\n=== Test metrics (educational model; not clinical validation) ===")
    print(f"Accuracy:           {acc:.4f}")
    print(f"Precision (macro): {precision_macro:.4f}")
    print(f"Recall (macro):    {recall_macro:.4f}")
    print(f"F1 (macro):         {f1_macro:.4f}")
    print("\nClassification report:\n")
    print(report)
    print("Confusion matrix (rows=true, cols=pred):")
    print(cm)

    reports_dir = config.PROJECT_ROOT / "reports"
    figures_dir = reports_dir / "figures"
    metrics_dir = reports_dir / "metrics"

    cm_path = figures_dir / "confusion_matrix_test.png"
    plot_confusion(cm, classes, cm_path)
    print(f"\nSaved confusion matrix figure to {cm_path}")

    metrics_payload = {
        "checkpoint": str(ckpt_path),
        "accuracy": acc,
        "precision_macro": precision_macro,
        "recall_macro": recall_macro,
        "f1_macro": f1_macro,
        "classes": classes,
        "confusion_matrix": cm.tolist(),
    }
    metrics_dir.mkdir(parents=True, exist_ok=True)
    metrics_json = metrics_dir / "test_metrics.json"
    metrics_json.write_text(json.dumps(metrics_payload, indent=2), encoding="utf-8")
    print(f"Saved metrics JSON to {metrics_json}")


if __name__ == "__main__":
    main()
