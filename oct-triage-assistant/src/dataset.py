"""
Dataset utilities: map folder names to labels and build PyTorch DataLoaders.

We use `torchvision.datasets.ImageFolder`, which expects:

    root/
        class_a/
        class_b/

Each image file in a folder inherits that folder's label. This matches the
medical imaging convention "one folder per diagnosis category" and avoids
maintaining a separate CSV for the baseline.
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

from torch.utils.data import DataLoader
from torchvision import datasets

from src import config


def _assert_split_exists(split_dir: Path, split_name: str) -> None:
    if not split_dir.is_dir():
        raise FileNotFoundError(
            f"Missing {split_name} directory: {split_dir}. "
            "Create it and add class subfolders (CNV, DME, DRUSEN, NORMAL) with images inside."
        )


def get_dataloaders(
    train_dir: Path | None = None,
    val_dir: Path | None = None,
    train_transform=None,
    val_transform=None,
    batch_size: int | None = None,
    num_workers: int | None = None,
) -> Tuple[DataLoader, DataLoader, datasets.ImageFolder, datasets.ImageFolder]:
    """
    Build train and validation DataLoaders.

    Returns `(train_loader, val_loader, train_dataset, val_dataset)` so callers
    can inspect `dataset.classes` / `dataset.class_to_idx` if needed.
    """
    train_dir = train_dir or config.TRAIN_DIR
    val_dir = val_dir or config.VAL_DIR
    batch_size = batch_size if batch_size is not None else config.BATCH_SIZE
    num_workers = num_workers if num_workers is not None else config.NUM_WORKERS

    _assert_split_exists(train_dir, "train")
    _assert_split_exists(val_dir, "val")

    train_ds = datasets.ImageFolder(str(train_dir), transform=train_transform)
    val_ds = datasets.ImageFolder(str(val_dir), transform=val_transform)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    return train_loader, val_loader, train_ds, val_ds


def get_test_loader(
    test_dir: Path | None = None,
    transform=None,
    batch_size: int | None = None,
    num_workers: int | None = None,
) -> Tuple[DataLoader, datasets.ImageFolder]:
    test_dir = test_dir or config.TEST_DIR
    batch_size = batch_size if batch_size is not None else config.BATCH_SIZE
    num_workers = num_workers if num_workers is not None else config.NUM_WORKERS

    _assert_split_exists(test_dir, "test")

    test_ds = datasets.ImageFolder(str(test_dir), transform=transform)
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    return test_loader, test_ds
