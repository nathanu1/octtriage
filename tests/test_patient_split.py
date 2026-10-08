from __future__ import annotations

import pandas as pd
import pytest
from PIL import Image

from octtriage.data.manifest import (
    assert_no_patient_leakage,
    discover_kermany_images,
    parse_kermany_patient_id,
    patient_level_split,
    primary_patient_labels,
)


def test_kermany_patient_id_parser() -> None:
    assert parse_kermany_patient_id("CNV-9911627-16.jpeg") == "9911627"


def test_patient_split_has_no_overlap_and_preserves_images() -> None:
    rows = []
    for label in ["CNV", "DME", "DRUSEN", "NORMAL"]:
        for patient in range(10):
            for image in range(2):
                rows.append(
                    {
                        "path": f"/{label}/{label}-{patient}-{image}.jpeg",
                        "label": label,
                        "patient_id": f"{label}:{patient}",
                    }
                )
    manifest = pd.DataFrame(rows)
    result = patient_level_split(manifest, seed=7)
    assert len(result) == len(manifest)
    assert set(result["split"]) == {"train", "val", "test"}
    assert_no_patient_leakage(result)
    assert (result.groupby("patient_id")["split"].nunique() == 1).all()


def test_leakage_assertion_rejects_patient_in_two_splits() -> None:
    manifest = pd.DataFrame(
        [
            {"patient_id": "CNV:1", "split": "train"},
            {"patient_id": "CNV:1", "split": "test"},
        ]
    )
    with pytest.raises(ValueError, match="Patient leakage"):
        assert_no_patient_leakage(manifest)


def test_patient_with_images_in_several_classes_stays_in_one_split() -> None:
    rows = []
    for offset, label in enumerate(["CNV", "DME", "DRUSEN", "NORMAL"]):
        for patient in range(offset * 10, offset * 10 + 10):
            rows.append(
                {"path": f"/{label}/{label}-{patient}-0.jpeg", "label": label,
                 "patient_id": str(patient)}
            )
    # Kermany patient 3 also contributes drusen images alongside a CNV image.
    rows += [
        {"path": f"/DRUSEN/DRUSEN-3-{image}.jpeg", "label": "DRUSEN", "patient_id": "3"}
        for image in range(1, 3)
    ]
    result = patient_level_split(pd.DataFrame(rows), seed=11)
    assert len(result) == len(rows)
    assert result.loc[result["patient_id"] == "3", "split"].nunique() == 1
    assert_no_patient_leakage(result)


def test_primary_label_prefers_the_most_frequent_class() -> None:
    manifest = pd.DataFrame(
        [
            {"patient_id": "7", "label": "DRUSEN"},
            {"patient_id": "7", "label": "CNV"},
            {"patient_id": "7", "label": "DRUSEN"},
            {"patient_id": "8", "label": "NORMAL"},
            {"patient_id": "8", "label": "DME"},
        ]
    )
    assert primary_patient_labels(manifest).to_dict() == {"7": "DRUSEN", "8": "DME"}


def test_discovery_deduplicates_copies_and_skips_mislabeled_files(tmp_path) -> None:
    for folder in ["OCT2017/train/CNV", "duplicate/OCT2017/train/CNV", "OCT2017/test/DME"]:
        (tmp_path / folder).mkdir(parents=True)
    Image.new("L", (8, 8)).save(tmp_path / "OCT2017/train/CNV/CNV-100-1.jpeg")
    Image.new("L", (8, 8)).save(tmp_path / "duplicate/OCT2017/train/CNV/CNV-100-1.jpeg")
    Image.new("L", (8, 8)).save(tmp_path / "OCT2017/test/DME/DME-200-1.jpeg")
    Image.new("L", (8, 8)).save(tmp_path / "OCT2017/test/DME/CNV-300-1.jpeg")
    with pytest.warns(UserWarning, match="Skipped 1 image"):
        manifest = discover_kermany_images(tmp_path)
    assert sorted(manifest["patient_id"]) == ["100", "200"]
    assert dict(zip(manifest["patient_id"], manifest["label"], strict=True)) == {
        "100": "CNV",
        "200": "DME",
    }
