"""Combine per-view model outputs into the evidence the triage engine consumes.

Serving and evaluation share this module so that reported policy metrics describe exactly
the aggregation that runs in production.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import numpy as np

from octtriage.data.quality import QualityAssessment
from octtriage.triage.engine import TriageDecision, TriageEngine


@dataclass(frozen=True)
class AggregatedEvidence:
    probabilities: dict[str, float]
    predicted_label: str
    confidence: float
    tta_agreement: float


def _validated(view_probabilities: np.ndarray, classes: Sequence[str]) -> np.ndarray:
    values = np.asarray(view_probabilities, dtype=np.float64)
    if values.ndim != 3 or values.shape[0] == 0 or values.shape[1] == 0:
        raise ValueError("Expected view probabilities shaped (frames, views, classes)")
    if values.shape[2] != len(classes):
        raise ValueError(
            f"Probability vectors have {values.shape[2]} classes but {len(classes)} were named"
        )
    return values


def aggregate_views(
    view_probabilities: np.ndarray,
    classes: Sequence[str],
    non_pathology_classes: Iterable[str],
) -> AggregatedEvidence:
    """Aggregate softmax outputs shaped ``(frames, views, classes)``.

    Pathology classes take the maximum over every frame and view, so focal evidence in a
    single B-scan is not averaged away. Non-pathology (reassuring) classes take the minimum,
    so reassurance must hold in every view. Test-time-augmentation agreement is measured
    within each frame, because different B-scans of one volume legitimately differ.
    """
    values = _validated(view_probabilities, classes)
    reassuring = set(non_pathology_classes)
    flat = values.reshape(-1, len(classes))
    is_reassuring = np.array([label in reassuring for label in classes])
    aggregated = np.where(is_reassuring, flat.min(axis=0), flat.max(axis=0))
    predicted_label = classes[int(aggregated.argmax())]

    votes = values.argmax(axis=2)
    per_frame_agreement = [
        np.bincount(frame_votes, minlength=len(classes)).max() / frame_votes.size
        for frame_votes in votes
    ]
    probabilities = {label: float(aggregated[index]) for index, label in enumerate(classes)}
    return AggregatedEvidence(
        probabilities=probabilities,
        predicted_label=predicted_label,
        confidence=probabilities[predicted_label],
        tta_agreement=float(np.mean(per_frame_agreement)),
    )


def strongest_frame(view_probabilities: np.ndarray, class_index: int) -> int:
    """Return the frame whose best view assigns the highest probability to ``class_index``."""
    values = np.asarray(view_probabilities, dtype=np.float64)
    return int(values[:, :, class_index].max(axis=1).argmax())


def decide_from_views(
    engine: TriageEngine,
    view_probabilities: np.ndarray | None,
    classes: Sequence[str],
    non_pathology_classes: Iterable[str],
    quality: QualityAssessment,
) -> tuple[TriageDecision, AggregatedEvidence | None]:
    """Apply the triage policy to per-view outputs, or to the quality gate alone."""
    if not quality.passed or view_probabilities is None:
        decision = engine.decide(
            predicted_label=None,
            confidence=None,
            probabilities={},
            tta_agreement=None,
            quality=quality,
        )
        return decision, None
    evidence = aggregate_views(view_probabilities, classes, non_pathology_classes)
    decision = engine.decide(
        predicted_label=evidence.predicted_label,
        confidence=evidence.confidence,
        probabilities=evidence.probabilities,
        tta_agreement=evidence.tta_agreement,
        quality=quality,
        severity="UNKNOWN",
    )
    return decision, evidence
