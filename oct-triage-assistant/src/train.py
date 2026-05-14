"""
Training script: supervised fine-tuning of the ResNet18 head (and full network).

Core ML ideas implemented here:
- **Epoch**: one full pass over the training set.
- **Loss** (`CrossEntropyLoss`): measures how wrong logits are vs true class; minimized by SGD/Adam.
- **Accuracy**: fraction of correct argmax predictions on a split — easy to read but incomplete for imbalanced data (we address richer metrics in `evaluate.py`).
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.optim import Adam

from src import config
from src.dataset import get_dataloaders
from src.model import build_model
from src.transforms import get_train_transforms, get_val_transforms


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train_one_epoch(model, loader, criterion, optimizer, device) -> float:
    model.train()
    running_loss = 0.0
    for inputs, targets in loader:
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits = model(inputs)
        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
    return running_loss / max(1, len(loader.dataset))


@torch.inference_mode()
def validate(model, loader, criterion, device) -> tuple[float, float]:
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    for inputs, targets in loader:
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        logits = model(inputs)
        loss = criterion(logits, targets)
        running_loss += loss.item() * inputs.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == targets).sum().item()
        total += targets.size(0)
    avg_loss = running_loss / max(1, total)
    acc = correct / max(1, total)
    return avg_loss, acc


def main() -> None:
    parser = argparse.ArgumentParser(description="Train OCT triage baseline (educational).")
    parser.add_argument("--epochs", type=int, default=config.EPOCHS)
    parser.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=config.LEARNING_RATE)
    parser.add_argument("--num-workers", type=int, default=config.NUM_WORKERS)
    parser.add_argument("--device", type=str, default=config.DEVICE)
    args = parser.parse_args()

    set_seed(config.SEED)
    device = torch.device(args.device)

    train_tf = get_train_transforms(config.IMG_SIZE)
    val_tf = get_val_transforms(config.IMG_SIZE)
    train_loader, val_loader, train_ds, val_ds = get_dataloaders(
        train_transform=train_tf,
        val_transform=val_tf,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    # Sanity check: folder names should match our configured vocabulary.
    if tuple(train_ds.classes) != config.CLASSES:
        print(
            "Warning: ImageFolder class order/names differ from src.config.CLASSES.\n"
            f"  Found: {train_ds.classes}\n"
            f"  Config expects: {list(config.CLASSES)}"
        )

    model = build_model(num_classes=len(train_ds.classes), pretrained=True).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=config.WEIGHT_DECAY)

    config.SAVED_MODELS_DIR.mkdir(parents=True, exist_ok=True)

    best_val_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        print(
            f"Epoch {epoch:02d}/{args.epochs} | "
            f"train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | val_acc={val_acc:.4f}"
        )

        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            ckpt = {
                "model_state_dict": model.state_dict(),
                "classes": train_ds.classes,
                "class_to_idx": train_ds.class_to_idx,
                "val_acc": val_acc,
                "epoch": epoch,
                "img_size": config.IMG_SIZE,
            }
            torch.save(ckpt, config.CHECKPOINT_PATH)
            print(f"  Saved new best checkpoint to {config.CHECKPOINT_PATH} (val_acc={val_acc:.4f})")

    print("Training complete.")
    print(f"Best validation accuracy (during this run): {best_val_acc:.4f}")
    print("Next step: `python -m src.evaluate` on the held-out test split.")


if __name__ == "__main__":
    main()
