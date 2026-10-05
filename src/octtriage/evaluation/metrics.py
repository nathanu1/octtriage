from __future__ import annotations

from collections import Counter
from collections.abc import Sequence

import numpy as np
from sklearn.metrics import confusion_matrix, roc_auc_score


def expected_calibration_error(
    y_true: np.ndarray, probabilities: np.ndarray, bins: int = 10
) -> float:
    confidence = probabilities.max(axis=1)
    predicted = probabilities.argmax(axis=1)
    correct = predicted == y_true
    edges = np.linspace(0.0, 1.0, bins + 1)
    error = 0.0
    for lower, upper in zip(edges[:-1], edges[1:], strict=True):
        mask = (confidence > lower) & (confidence <= upper)
        if mask.any():
            accuracy_gap = abs(
                float(correct[mask].mean()) - float(confidence[mask].mean())
            )
            error += float(mask.mean()) * accuracy_gap
    return error


def classification_metrics(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    classes: list[str],
    urgent_classes: set[str],
) -> dict[str, object]:
    predicted = probabilities.argmax(axis=1)
    matrix = confusion_matrix(y_true, predicted, labels=np.arange(len(classes)))
    per_class: dict[str, dict[str, float | None]] = {}
    for index, label in enumerate(classes):
        true_positive = int(matrix[index, index])
        false_negative = int(matrix[index, :].sum() - true_positive)
        false_positive = int(matrix[:, index].sum() - true_positive)
        true_negative = int(matrix.sum() - true_positive - false_negative - false_positive)
        sensitivity = true_positive / max(true_positive + false_negative, 1)
        specificity = true_negative / max(true_negative + false_positive, 1)
        binary_true = (y_true == index).astype(int)
        try:
            auroc: float | None = float(roc_auc_score(binary_true, probabilities[:, index]))
        except ValueError:
            auroc = None
        per_class[label] = {
            "sensitivity": sensitivity,
            "specificity": specificity,
            "auroc": auroc,
        }
    urgent_indices = {classes.index(label) for label in urgent_classes}
    true_urgent = np.isin(y_true, list(urgent_indices))
    predicted_urgent = np.isin(predicted, list(urgent_indices))
    urgent_recall = float(np.sum(true_urgent & predicted_urgent) / max(np.sum(true_urgent), 1))
    urgent_specificity = float(
        np.sum(~true_urgent & ~predicted_urgent) / max(np.sum(~true_urgent), 1)
    )
    urgent_precision = float(
        np.sum(true_urgent & predicted_urgent) / max(np.sum(predicted_urgent), 1)
    )
    return {
        "accuracy": float(np.mean(predicted == y_true)),
        "urgent_recall": urgent_recall,
        "urgent_specificity": urgent_specificity,
        "urgent_precision": urgent_precision,
        "expected_calibration_error": expected_calibration_error(y_true, probabilities),
        "per_class": per_class,
        "confusion_matrix": matrix.tolist(),
    }


def triage_policy_metrics(
    y_true: np.ndarray,
    tiers: Sequence[str],
    escalation_reasons: Sequence[Sequence[str]],
    classes: list[str],
    urgent_classes: set[str],
) -> dict[str, object]:
    """Summarize the deployed triage policy rather than the classifier's argmax.

    Urgent recall of the raw classifier ignores the quality gate, candidate-probability
    promotion, and uncertainty escalation. These metrics describe what a reviewer's queue
    would actually look like: how many urgent cases reach the URGENT tier, and how much of
    the queue the URGENT tier consumes.
    """
    labels = np.asarray(y_true)
    tier_array = np.asarray(tiers)
    if len(labels) != len(tier_array) or len(labels) != len(escalation_reasons):
        raise ValueError("Labels, tiers, and escalation reasons must align")
    urgent_indices = [classes.index(label) for label in urgent_classes]
    true_urgent = np.isin(labels, urgent_indices)
    triaged_urgent = tier_array == "URGENT"
    by_label = {
        label: {
            tier: int(np.sum((labels == index) & (tier_array == tier)))
            for tier in ("URGENT", "SEMI_URGENT", "ROUTINE")
        }
        for index, label in enumerate(classes)
    }
    reasons = Counter(reason for case in escalation_reasons for reason in case)
    return {
        "cases": int(len(labels)),
        "urgent_tier_sensitivity": float(
            np.sum(true_urgent & triaged_urgent) / max(np.sum(true_urgent), 1)
        ),
        "urgent_cases_below_urgent_tier": int(np.sum(true_urgent & ~triaged_urgent)),
        "non_urgent_cases_triaged_urgent_rate": float(
            np.sum(~true_urgent & triaged_urgent) / max(np.sum(~true_urgent), 1)
        ),
        "urgent_tier_fraction": float(np.mean(triaged_urgent)) if len(labels) else 0.0,
        "escalated_fraction": float(np.mean([bool(case) for case in escalation_reasons]))
        if len(labels)
        else 0.0,
        "tier_counts_by_true_label": by_label,
        "escalation_reason_counts": dict(sorted(reasons.items())),
    }
