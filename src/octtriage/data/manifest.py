from __future__ import annotations

import re
from collections.abc import Iterable
from pathlib import Path

import pandas as pd

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
KERMANY_FILENAME = re.compile(
    r"^(?P<label>CNV|DME|DRUSEN|NORMAL)-(?P<patient>[^-]+)(?:-|\.)",
    re.IGNORECASE,
)


def parse_kermany_patient_id(path: str | Path) -> str:
    item = Path(path)
    match = KERMANY_FILENAME.match(item.name)
    if not match:
        raise ValueError(
            f"Cannot derive a patient identifier from {item.name!r}; "
            "provide an explicit patient-ID manifest instead of using an image-level split."
        )
    return match.group("patient")


def discover_kermany_images(root: str | Path) -> pd.DataFrame:
    root = Path(root).resolve()
    records: list[dict[str, str]] = []
    seen: set[Path] = set()
    for path in root.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        label = next((part.upper() for part in reversed(path.parts) if part.upper() in {
            "CNV", "DME", "DRUSEN", "NORMAL"
        }), None)
        if label is None:
            continue
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        records.append(
            {
                "path": str(resolved),
                "label": label,
                "patient_id": parse_kermany_patient_id(path),
            }
        )
    if not records:
        raise FileNotFoundError(f"No Kermany OCT images found below {root}")
    return (
        pd.DataFrame.from_records(records)
        .sort_values(["patient_id", "path"])
        .reset_index(drop=True)
    )


def _allocate_groups(
    groups: list[str], val_fraction: float, test_fraction: float
) -> dict[str, str]:
    count = len(groups)
    if count < 3:
        raise ValueError("Each class needs at least three patients for train/val/test splitting")
    test_count = max(1, round(count * test_fraction))
    val_count = max(1, round(count * val_fraction))
    if test_count + val_count >= count:
        test_count = 1
        val_count = 1
    assignment: dict[str, str] = {}
    for patient in groups[:test_count]:
        assignment[patient] = "test"
    for patient in groups[test_count : test_count + val_count]:
        assignment[patient] = "val"
    for patient in groups[test_count + val_count :]:
        assignment[patient] = "train"
    return assignment


def patient_level_split(
    manifest: pd.DataFrame,
    *,
    val_fraction: float = 0.15,
    test_fraction: float = 0.15,
    seed: int = 42,
) -> pd.DataFrame:
    required = {"path", "label", "patient_id"}
    if not required.issubset(manifest.columns):
        raise ValueError(f"Manifest must contain columns: {sorted(required)}")
    if not 0 < val_fraction < 1 or not 0 < test_fraction < 1:
        raise ValueError("Validation and test fractions must be between zero and one")
    if val_fraction + test_fraction >= 1:
        raise ValueError("Validation plus test fraction must be below one")

    import random

    rng = random.Random(seed)
    assignments: dict[str, str] = {}
    patient_labels = manifest[["patient_id", "label"]].drop_duplicates()
    if patient_labels["patient_id"].duplicated().any():
        raise ValueError("A patient appears with more than one class label")
    for _, group in patient_labels.groupby("label"):
        patients = sorted(group["patient_id"].astype(str).tolist())
        rng.shuffle(patients)
        assignments.update(_allocate_groups(patients, val_fraction, test_fraction))

    result = manifest.copy()
    result["split"] = result["patient_id"].map(assignments)
    assert_no_patient_leakage(result)
    return result.sort_values(["split", "label", "patient_id", "path"]).reset_index(drop=True)


def assert_no_patient_leakage(manifest: pd.DataFrame) -> None:
    counts = manifest.groupby("patient_id")["split"].nunique()
    leaking = counts[counts > 1]
    if not leaking.empty:
        examples = ", ".join(leaking.index.astype(str).tolist()[:5])
        raise ValueError(f"Patient leakage detected across splits: {examples}")


def write_manifest(manifest: pd.DataFrame, output: str | Path) -> Path:
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(path, index=False)
    return path


def subset_manifest(manifest: pd.DataFrame, splits: Iterable[str]) -> pd.DataFrame:
    allowed = set(splits)
    return manifest[manifest["split"].isin(allowed)].copy()
