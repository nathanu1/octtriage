from __future__ import annotations

import argparse
import random
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import recall_score
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader

from octtriage.config import load_yaml
from octtriage.data.dataset import OCTManifestDataset
from octtriage.data.manifest import assert_no_patient_leakage
from octtriage.data.transforms import inference_transform, training_transform
from octtriage.models.classifier import build_classifier
from octtriage.models.registry import register_model
from octtriage.settings import settings


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def urgent_recall(y_true: list[int], y_pred: list[int], urgent_indices: set[int]) -> float:
    true_binary = [int(item in urgent_indices) for item in y_true]
    pred_binary = [int(item in urgent_indices) for item in y_pred]
    return float(recall_score(true_binary, pred_binary, zero_division=0))


def class_weights(frame: pd.DataFrame, classes: list[str], device: torch.device) -> torch.Tensor:
    counts = frame[frame["split"] == "train"]["label"].value_counts()
    weights = [1.0 / max(int(counts.get(label, 0)), 1) for label in classes]
    weights = np.asarray(weights, dtype=np.float32)
    weights /= weights.mean()
    return torch.tensor(weights, device=device)


def run_epoch(model, loader, loss_fn, device, optimizer=None) -> tuple[float, list[int], list[int]]:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    y_true: list[int] = []
    y_pred: list[int] = []
    context = torch.enable_grad() if training else torch.inference_mode()
    with context:
        for inputs, targets in loader:
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            if training:
                optimizer.zero_grad(set_to_none=True)
            logits = model(inputs)
            loss = loss_fn(logits, targets)
            if training:
                loss.backward()
                optimizer.step()
            total_loss += float(loss.item()) * inputs.shape[0]
            y_true.extend(targets.cpu().tolist())
            y_pred.extend(logits.argmax(dim=1).cpu().tolist())
    return total_loss / max(len(loader.dataset), 1), y_true, y_pred


def train(manifest_path: Path, output_path: Path, config_path: Path, device_name: str) -> Path:
    config = load_yaml(config_path)
    seed_everything(int(config["seed"]))
    frame = pd.read_csv(manifest_path)
    assert_no_patient_leakage(frame)
    classes = list(config["classes"])
    device = torch.device(device_name)
    image_size = int(config["image_size"])
    train_dataset = OCTManifestDataset(
        frame, "train", training_transform(image_size), classes
    )
    val_dataset = OCTManifestDataset(frame, "val", inference_transform(image_size), classes)
    loader_args = {
        "batch_size": int(config["batch_size"]),
        "num_workers": int(config["num_workers"]),
        "pin_memory": device.type == "cuda",
    }
    train_loader = DataLoader(train_dataset, shuffle=True, **loader_args)
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_args)

    model = build_classifier(
        len(classes),
        backbone=str(config["backbone"]),
        pretrained=bool(config["pretrained_imagenet"]),
    ).to(device)
    loss_fn = nn.CrossEntropyLoss(weight=class_weights(frame, classes, device))
    optimizer = AdamW(
        model.parameters(),
        lr=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
    )
    urgent_indices = {classes.index(label) for label in config["urgent_classes"]}
    version = f"octtriage-{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}"
    best_score = -1.0
    best_loss = float("inf")
    best_metrics: dict[str, float] = {}
    output_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, int(config["epochs"]) + 1):
        train_loss, _, _ = run_epoch(model, train_loader, loss_fn, device, optimizer)
        val_loss, y_true, y_pred = run_epoch(model, val_loader, loss_fn, device)
        score = urgent_recall(y_true, y_pred, urgent_indices)
        accuracy = float(np.mean(np.asarray(y_true) == np.asarray(y_pred)))
        print(
            f"epoch={epoch:02d} train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
            f"urgent_recall={score:.4f} accuracy={accuracy:.4f}"
        )
        if score > best_score or (score == best_score and val_loss < best_loss):
            best_score, best_loss = score, val_loss
            best_metrics = {
                "urgent_recall": score,
                "validation_loss": val_loss,
                "validation_accuracy": accuracy,
            }
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "classes": classes,
                    "image_size": image_size,
                    "backbone": config["backbone"],
                    "model_version": version,
                    "epoch": epoch,
                    "validation_metrics": best_metrics,
                },
                output_path,
            )
    register_model(
        settings.model_registry_path,
        model_version=version,
        checkpoint_path=output_path,
        metrics=best_metrics,
        training_manifest=manifest_path,
    )
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the OCT2017 decision-support baseline")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=settings.project_root / "data" / "manifests" / "kermany.csv",
    )
    parser.add_argument("--output", type=Path, default=settings.checkpoint_path)
    parser.add_argument("--config", type=Path, default=settings.model_config_path)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    args = parser.parse_args()
    train(args.manifest, args.output, args.config, args.device)


if __name__ == "__main__":
    main()
