from __future__ import annotations

import numpy as np
import pytest

from octtriage.triage.aggregation import aggregate_views, strongest_frame

CLASSES = ["CNV", "DME", "DRUSEN", "NORMAL"]
REASSURING = {"NORMAL"}


def views(*frames: list[list[float]]) -> np.ndarray:
    return np.asarray(frames, dtype=float)


def test_reassuring_class_never_takes_the_most_optimistic_view() -> None:
    evidence = aggregate_views(
        views([[0.05, 0.05, 0.05, 0.85], [0.10, 0.10, 0.10, 0.70]]), CLASSES, REASSURING
    )
    assert evidence.predicted_label == "NORMAL"
    assert evidence.confidence == pytest.approx(0.70)
    assert evidence.probabilities["CNV"] == pytest.approx(0.10)


def test_focal_pathology_in_one_bscan_drives_the_volume_label() -> None:
    normal_frame = [[0.01, 0.01, 0.01, 0.97], [0.01, 0.01, 0.01, 0.97]]
    cnv_frame = [[0.90, 0.03, 0.02, 0.05], [0.88, 0.04, 0.03, 0.05]]
    evidence = aggregate_views(views(*([normal_frame] * 31), cnv_frame), CLASSES, REASSURING)
    assert evidence.predicted_label == "CNV"
    assert evidence.probabilities["CNV"] == pytest.approx(0.90)
    assert evidence.probabilities["NORMAL"] == pytest.approx(0.05)


def test_agreement_measures_flip_consistency_not_frame_heterogeneity() -> None:
    drusen_frame = [[0.05, 0.05, 0.85, 0.05], [0.05, 0.05, 0.80, 0.10]]
    normal_frame = [[0.02, 0.02, 0.06, 0.90], [0.02, 0.02, 0.10, 0.86]]
    heterogeneous = aggregate_views(views(drusen_frame, normal_frame), CLASSES, REASSURING)
    assert heterogeneous.tta_agreement == pytest.approx(1.0)

    flip_disagrees = [[0.05, 0.05, 0.60, 0.30], [0.05, 0.05, 0.30, 0.60]]
    mixed = aggregate_views(views(drusen_frame, flip_disagrees), CLASSES, REASSURING)
    assert mixed.tta_agreement == pytest.approx(0.75)


def test_strongest_frame_finds_the_bscan_with_most_evidence() -> None:
    probabilities = views(
        [[0.20, 0.0, 0.0, 0.80], [0.10, 0.0, 0.0, 0.90]],
        [[0.30, 0.0, 0.0, 0.70], [0.65, 0.0, 0.0, 0.35]],
        [[0.40, 0.0, 0.0, 0.60], [0.40, 0.0, 0.0, 0.60]],
    )
    assert strongest_frame(probabilities, CLASSES.index("CNV")) == 1
    assert strongest_frame(probabilities, CLASSES.index("NORMAL")) == 0


@pytest.mark.parametrize(
    "shape", [(2, 4), (0, 2, 4), (1, 0, 4), (1, 2, 3)]
)
def test_malformed_probability_arrays_are_rejected(shape) -> None:
    with pytest.raises(ValueError):
        aggregate_views(np.zeros(shape), CLASSES, REASSURING)
