from __future__ import annotations

import pandas as pd
import pytest

from octtriage.data.manifest import (
    assert_no_patient_leakage,
    parse_kermany_patient_id,
    patient_level_split,
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
