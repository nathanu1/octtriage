from __future__ import annotations

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
    return {
        "accuracy": float(np.mean(predicted == y_true)),
        "urgent_recall": urgent_recall,
        "expected_calibration_error": expected_calibration_error(y_true, probabilities),
        "per_class": per_class,
        "confusion_matrix": matrix.tolist(),
    }
