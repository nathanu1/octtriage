from __future__ import annotations

import re
import warnings
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
    seen: set[tuple[str, str]] = set()
    mismatched: list[str] = []
    # Sorted traversal makes the retained copy of a duplicated file deterministic.
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        label = next((part.upper() for part in reversed(path.parent.parts) if part.upper() in {
            "CNV", "DME", "DRUSEN", "NORMAL"
        }), None)
        if label is None:
            continue
        filename_match = KERMANY_FILENAME.match(path.name)
        if filename_match and filename_match.group("label").upper() != label:
            mismatched.append(str(path))
            continue
        # Kaggle extractions can repeat the same Kermany file under several folders; the
        # filename encodes label, patient, and image number, so it identifies the image.
        key = (label, path.name.lower())
        if key in seen:
            continue
        seen.add(key)
        records.append(
            {
                "path": str(path.resolve()),
                "label": label,
                "patient_id": parse_kermany_patient_id(path),
            }
        )
    if mismatched:
        warnings.warn(
            f"Skipped {len(mismatched)} image(s) whose filename label disagrees with their "
            f"folder, e.g. {mismatched[0]}",
            stacklevel=2,
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
    result = manifest.copy()
    result["patient_id"] = result["patient_id"].astype(str)
    strata = primary_patient_labels(result)
    for _, group in strata.groupby(strata):
        patients = sorted(group.index.tolist())
        rng.shuffle(patients)
        assignments.update(_allocate_groups(patients, val_fraction, test_fraction))

    result["split"] = result["patient_id"].map(assignments)
    assert_no_patient_leakage(result)
    return result.sort_values(["split", "label", "patient_id", "path"]).reset_index(drop=True)


def primary_patient_labels(manifest: pd.DataFrame) -> pd.Series:
    """Return one stratification label per patient.

    A patient can contribute images under more than one class (for example drusen in one eye
    and CNV in the other). Every image from that patient must still land in a single split, so
    patients are stratified by their most frequent label, with ties broken alphabetically.
    """
    counts = manifest.groupby(["patient_id", "label"]).size().reset_index(name="images")
    counts = counts.sort_values(["patient_id", "images", "label"], ascending=[True, False, True])
    return counts.drop_duplicates("patient_id").set_index("patient_id")["label"]


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
