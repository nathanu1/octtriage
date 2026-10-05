from __future__ import annotations

import numpy as np
import pytest

from octtriage.evaluation.metrics import classification_metrics, triage_policy_metrics

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
