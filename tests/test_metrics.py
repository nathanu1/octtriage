from __future__ import annotations

import numpy as np
import pytest

from octtriage.evaluation.metrics import (
    classification_metrics,
    patient_bootstrap_intervals,
    triage_policy_metrics,
)

CLASSES = ["CNV", "DME", "DRUSEN", "NORMAL"]
URGENT = {"CNV", "DME"}


def test_classification_metrics_report_urgent_binary_performance() -> None:
    y_true = np.array([0, 1, 2, 3])
    probabilities = np.array(
        [
            [0.9, 0.05, 0.03, 0.02],  # CNV correct
            [0.1, 0.10, 0.10, 0.70],  # DME missed as NORMAL
            [0.6, 0.10, 0.20, 0.10],  # DRUSEN flagged as CNV
            [0.0, 0.00, 0.10, 0.90],  # NORMAL correct
        ]
    )
    metrics = classification_metrics(y_true, probabilities, CLASSES, URGENT)
    assert metrics["urgent_recall"] == pytest.approx(0.5)
    assert metrics["urgent_specificity"] == pytest.approx(0.5)
    assert metrics["urgent_precision"] == pytest.approx(0.5)
    assert metrics["per_class"]["CNV"]["sensitivity"] == pytest.approx(1.0)
    assert metrics["confusion_matrix"][1][3] == 1


def test_policy_metrics_describe_the_review_queue() -> None:
    y_true = np.array([0, 1, 2, 3, 3])
    tiers = ["URGENT", "SEMI_URGENT", "URGENT", "ROUTINE", "URGENT"]
    reasons = [[], [], ["low_model_confidence"], [], ["quality_gate_failed"]]
    metrics = triage_policy_metrics(y_true, tiers, reasons, CLASSES, URGENT)
    assert metrics["urgent_tier_sensitivity"] == pytest.approx(0.5)
    assert metrics["urgent_cases_below_urgent_tier"] == 1
    assert metrics["non_urgent_cases_triaged_urgent_rate"] == pytest.approx(2 / 3)
    assert metrics["urgent_tier_fraction"] == pytest.approx(0.6)
    assert metrics["escalated_fraction"] == pytest.approx(0.4)
    assert metrics["tier_counts_by_true_label"]["NORMAL"] == {
        "URGENT": 1,
        "SEMI_URGENT": 0,
        "ROUTINE": 1,
    }
    assert metrics["escalation_reason_counts"] == {
        "low_model_confidence": 1,
        "quality_gate_failed": 1,
    }


def test_policy_metrics_reject_misaligned_inputs() -> None:
    with pytest.raises(ValueError, match="align"):
        triage_policy_metrics(np.array([0, 1]), ["URGENT"], [[], []], CLASSES, URGENT)


def test_bootstrap_interval_brackets_the_estimate() -> None:
    rng = np.random.default_rng(1)
    correct = rng.random(400) < 0.8
    intervals = patient_bootstrap_intervals(
        np.arange(400).astype(str), {"accuracy": (correct, np.ones(400))}, resamples=500
    )
    accuracy = intervals["accuracy"]
    assert accuracy["lower"] <= accuracy["estimate"] <= accuracy["upper"]
    assert accuracy["estimate"] == pytest.approx(correct.mean())
    assert "patient-level" in intervals["method"]


def test_bootstrap_resamples_patients_not_images() -> None:
    # Ten patients with 50 images each; half the patients are entirely misclassified.
    patients = np.repeat(np.arange(10).astype(str), 50)
    correct = np.repeat([True, False] * 5, 50)
    by_patient = patient_bootstrap_intervals(
        patients, {"accuracy": (correct, np.ones(500))}, resamples=1000
    )["accuracy"]
    by_image = patient_bootstrap_intervals(
        np.arange(500).astype(str), {"accuracy": (correct, np.ones(500))}, resamples=1000
    )["accuracy"]
    assert by_patient["upper"] - by_patient["lower"] > 2 * (by_image["upper"] - by_image["lower"])


def test_bootstrap_reports_undefined_metrics_as_none() -> None:
    intervals = patient_bootstrap_intervals(
        ["a", "b"], {"urgent_recall": (np.zeros(2), np.zeros(2))}, resamples=50
    )
    assert intervals["urgent_recall"] == {"estimate": None, "lower": None, "upper": None}
