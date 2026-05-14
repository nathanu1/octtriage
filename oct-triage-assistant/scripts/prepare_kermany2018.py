#!/usr/bin/env python3
"""
Download Kermany et al. 2018 OCT (Kaggle: paultimothymooney/kermany2018) via kagglehub
and populate `data/raw/{train,val,test}/{CNV,DME,DRUSEN,NORMAL}/`.

Typical upstream layout (Kaggle extract):

    OCT2017/train/<CLASS>/*.jpeg
    OCT2017/val/<CLASS>/*.jpeg   (small hold-out in the published split)
    OCT2017/test/<CLASS>/*.jpeg

Strategy
--------
- If **train**, **val**, and **test** siblings exist (four classes each): symlink/copy
  those directly into `data/raw/` (publisher split). Skips `__MACOSX` noise from zips.
- Else if only **train** + **test**: official test → `data/raw/test/`; hold out
  `--val-fraction` of official train for `data/raw/val/`.
- Else: split a single **train** tree three ways using `--val-fraction` and `--test-fraction`.

By default we create **symlinks** (almost no extra disk). Use `--mode copy`
only if you need a portable copy (will duplicate many gigabytes).

Auth: configure Kaggle / kagglehub credentials as documented in the kagglehub README.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from sklearn.model_selection import train_test_split  # noqa: E402

from src import config  # noqa: E402

IMG_EXTS = {".jpeg", ".jpg", ".png", ".tif", ".tiff", ".bmp"}
CANONICAL_CLASSES = list(config.CLASSES)
_SKIP_PATH_PARTS = frozenset({"__MACOSX"})


def _path_is_skipped(p: Path) -> bool:
    return any(part in _SKIP_PATH_PARTS for part in p.parts)


def _canonical_class(name: str) -> str | None:
    u = name.upper()
    return u if u in set(CANONICAL_CLASSES) else None


def _list_images(class_dir: Path) -> list[Path]:
    if not class_dir.is_dir():
        return []
    return sorted(
        p for p in class_dir.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS
    )


def _has_all_class_dirs(split_dir: Path) -> bool:
    found = 0
    for child in split_dir.iterdir():
        if child.is_dir() and _canonical_class(child.name):
            found += 1
    return found == 4


def resolve_class_dir(split_root: Path, cls: str) -> Path:
    direct = split_root / cls
    if direct.is_dir():
        return direct
    for ch in split_root.iterdir():
        if ch.is_dir() and _canonical_class(ch.name) == cls:
            return ch
    raise FileNotFoundError(f"No folder for class {cls} under {split_root}")


def _train_image_count(train_root: Path) -> int:
    n = 0
    for cls in CANONICAL_CLASSES:
        n += len(_list_images(resolve_class_dir(train_root, cls)))
    return n


def find_official_train_val_test(download_root: Path) -> tuple[Path, Path, Path] | None:
    """
    Prefer a parent folder that contains official train/, val/, and test/ with four classes each.
    When multiple candidates exist (e.g. duplicate tree), pick the one with the most train images.
    """
    download_root = download_root.resolve()
    best: tuple[Path, Path, Path] | None = None
    best_score = -1

    for candidate_train in download_root.rglob("train"):
        if not candidate_train.is_dir() or _path_is_skipped(candidate_train):
            continue
        parent = candidate_train.parent
        val_dir = parent / "val"
        test_dir = parent / "test"
        if not val_dir.is_dir() or not test_dir.is_dir():
            continue
        if not (
            _has_all_class_dirs(candidate_train)
            and _has_all_class_dirs(val_dir)
            and _has_all_class_dirs(test_dir)
        ):
            continue
        score = _train_image_count(candidate_train)
        if score > best_score:
            best_score = score
            best = (candidate_train, val_dir, test_dir)

    return best


def find_train_test_only(download_root: Path) -> tuple[Path, Path] | None:
    """train/ + test/ siblings, four classes each; ignore __MACOSX; pick largest train."""
    download_root = download_root.resolve()
    best: tuple[Path, Path] | None = None
    best_score = -1

    for candidate_train in download_root.rglob("train"):
        if not candidate_train.is_dir() or _path_is_skipped(candidate_train):
            continue
        parent = candidate_train.parent
        candidate_test = parent / "test"
        if not candidate_test.is_dir():
            continue
        if not (_has_all_class_dirs(candidate_train) and _has_all_class_dirs(candidate_test)):
            continue
        if (parent / "val").is_dir() and _has_all_class_dirs(parent / "val"):
            continue  # prefer find_official_train_val_test for triple layouts
        score = _train_image_count(candidate_train)
        if score > best_score:
            best_score = score
            best = (candidate_train, candidate_test)

    return best


def find_train_only(download_root: Path) -> Path | None:
    download_root = download_root.resolve()
    best: Path | None = None
    best_score = -1
    for candidate_train in download_root.rglob("train"):
        if not candidate_train.is_dir() or _path_is_skipped(candidate_train):
            continue
        if not _has_all_class_dirs(candidate_train):
            continue
        score = _train_image_count(candidate_train)
        if score > best_score:
            best_score = score
            best = candidate_train
    return best


def clear_project_raw_classes() -> None:
    for split in (config.TRAIN_DIR, config.VAL_DIR, config.TEST_DIR):
        for cls in CANONICAL_CLASSES:
            d = split / cls
            d.mkdir(parents=True, exist_ok=True)
            for p in list(d.iterdir()):
                if p.is_file() or p.is_symlink():
                    p.unlink()


def _link_or_copy(src: Path, dst: Path, mode: str) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    src = src.resolve()
    if mode == "copy":
        shutil.copy2(src, dst)
        return
    try:
        os.symlink(src, dst, target_is_directory=False)
    except OSError:
        shutil.copy2(src, dst)


def _place_files(paths: list[Path], dest_class_dir: Path, mode: str) -> None:
    for src in paths:
        _link_or_copy(src, dest_class_dir / src.name, mode)


def prepare_official_three_way(kaggle_train: Path, kaggle_val: Path, kaggle_test: Path, mode: str) -> None:
    for cls in CANONICAL_CLASSES:
        _place_files(_list_images(resolve_class_dir(kaggle_train, cls)), config.TRAIN_DIR / cls, mode)
        _place_files(_list_images(resolve_class_dir(kaggle_val, cls)), config.VAL_DIR / cls, mode)
        _place_files(_list_images(resolve_class_dir(kaggle_test, cls)), config.TEST_DIR / cls, mode)


def prepare_from_kaggle_train_test(
    kaggle_train: Path,
    kaggle_test: Path,
    *,
    val_fraction: float,
    seed: int,
    mode: str,
) -> None:
    for cls in CANONICAL_CLASSES:
        tr_dir = resolve_class_dir(kaggle_train, cls)
        te_dir = resolve_class_dir(kaggle_test, cls)
        imgs = _list_images(tr_dir)
        if len(imgs) < 2:
            raise FileNotFoundError(f"Not enough train images for class {cls} under {tr_dir}")
        train_f, val_f = train_test_split(
            imgs,
            test_size=val_fraction,
            random_state=seed,
            shuffle=True,
        )
        _place_files(train_f, config.TRAIN_DIR / cls, mode)
        _place_files(val_f, config.VAL_DIR / cls, mode)
        _place_files(_list_images(te_dir), config.TEST_DIR / cls, mode)


def prepare_train_only_splits(
    kaggle_train: Path,
    *,
    val_fraction: float,
    test_fraction: float,
    seed: int,
    mode: str,
) -> None:
    for cls in CANONICAL_CLASSES:
        tr_dir = resolve_class_dir(kaggle_train, cls)
        imgs = _list_images(tr_dir)
        if len(imgs) < 3:
            raise FileNotFoundError(f"Not enough images to split for class {cls} under {tr_dir}")
        trainval_f, test_f = train_test_split(
            imgs,
            test_size=test_fraction,
            random_state=seed,
            shuffle=True,
        )
        val_of_trainval = val_fraction / max(1e-8, (1.0 - test_fraction))
        val_of_trainval = min(max(val_of_trainval, 1e-6), 0.99)
        train_f, val_f = train_test_split(
            trainval_f,
            test_size=val_of_trainval,
            random_state=seed,
            shuffle=True,
        )
        _place_files(train_f, config.TRAIN_DIR / cls, mode)
        _place_files(val_f, config.VAL_DIR / cls, mode)
        _place_files(test_f, config.TEST_DIR / cls, mode)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare Kermany 2018 Kaggle data under data/raw/.")
    parser.add_argument(
        "--source",
        type=str,
        default=None,
        help="Path to an already-downloaded/extracted dataset root (skips kagglehub download).",
    )
    parser.add_argument(
        "--val-fraction",
        type=float,
        default=0.15,
        help="Fraction of official TRAIN held out for val when only train+test exist (ignored if official val/ exists).",
    )
    parser.add_argument(
        "--test-fraction",
        type=float,
        default=0.15,
        help="Test fraction when only a single train/ tree exists.",
    )
    parser.add_argument("--seed", type=int, default=config.SEED)
    parser.add_argument(
        "--mode",
        choices=("symlink", "copy"),
        default="symlink",
        help="symlink: points into kagglehub cache (default). copy: duplicate files (huge).",
    )
    parser.add_argument(
        "--no-clean",
        action="store_true",
        help="Do not remove existing files in data/raw/{train,val,test} class folders first.",
    )
    parser.add_argument(
        "--download-only",
        action="store_true",
        help="Only download via kagglehub and print the path; do not modify data/raw.",
    )
    args = parser.parse_args()

    if args.source:
        root = Path(args.source).expanduser().resolve()
        if not root.is_dir():
            raise FileNotFoundError(f"--source is not a directory: {root}")
    else:
        import kagglehub  # noqa: WPS433

        print(
            "Downloading / resolving dataset via kagglehub.\n"
            "Note: this dataset is large (~11 GB). First download can take a long time; "
            "kagglehub caches it under your user cache."
        )
        root = Path(kagglehub.dataset_download("paultimothymooney/kermany2018")).resolve()
        print("Kagglehub dataset root:", root)

    if args.download_only:
        return

    if not (0.0 < args.val_fraction < 0.5):
        raise ValueError("--val-fraction should be between 0 and 0.5.")
    if not (0.0 < args.test_fraction < 0.5):
        raise ValueError("--test-fraction should be between 0 and 0.5.")

    triple = find_official_train_val_test(root)
    pair = find_train_test_only(root) if triple is None else None
    train_only = find_train_only(root) if triple is None and pair is None else None

    if triple is not None:
        kt, kv, kte = triple
        print("Using official train / val / test split under:", kt.parent)
        print("  train:", kt)
        print("  val:  ", kv, "(note: published val can be small)")
        print("  test: ", kte)
    elif pair is not None:
        kt, kte = pair
        print("Using official train + test; holding out val from train:", kt.parent)
        print("  train:", kt)
        print("  test: ", kte)
    elif train_only is not None:
        kt = train_only
        print("Only train/ found; splitting into train/val/test:", kt.parent)
    else:
        raise FileNotFoundError(
            f"Could not find OCT2017-style folders under {root}. "
            "Pass --source pointing at the extracted archive root."
        )

    if not args.no_clean:
        clear_project_raw_classes()

    if triple is not None:
        prepare_official_three_way(*triple, args.mode)
    elif pair is not None:
        prepare_from_kaggle_train_test(
            pair[0],
            pair[1],
            val_fraction=args.val_fraction,
            seed=args.seed,
            mode=args.mode,
        )
    else:
        assert train_only is not None
        prepare_train_only_splits(
            train_only,
            val_fraction=args.val_fraction,
            test_fraction=args.test_fraction,
            seed=args.seed,
            mode=args.mode,
        )

    print("\nDone. Layout is ready for `torchvision.datasets.ImageFolder`:")
    print(f"  {config.TRAIN_DIR}")
    print(f"  {config.VAL_DIR}")
    print(f"  {config.TEST_DIR}")
    print("\nNext:")
    print("  python3 -m src.train --num-workers 0")
    print("  python3 -m src.evaluate")


if __name__ == "__main__":
    main()
