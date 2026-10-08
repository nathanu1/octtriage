from __future__ import annotations

import numpy as np
import pytest

from octtriage.config import load_yaml
from octtriage.evaluation.evaluate import SplitOutputs
from octtriage.evaluation.thresholds import recommend_threshold, sweep_promotion_thresholds
from octtriage.settings import settings

CLASSES = ["CNV", "DME", "DRUSEN", "NORMAL"]
URGENT = {"CNV", "DME"}


def outputs_for(cases: list[tuple[int, list[float]]]) -> SplitOutputs:
    """Confident (no uncertainty escalation) single-B-scan cases with a given CNV probability."""
    probabilities = np.asarray([[row, row] for _, row in cases])
    return SplitOutputs(
        targets=np.asarray([target for target, _ in cases]),
        view_logits=np.log(probabilities),
        view_probabilities=probabilities,
        quality_passed=[True] * len(cases),
        quality_reasons=[[] for _ in cases],
        patient_ids=np.arange(len(cases)).astype(str),
    )


@pytest.fixture
def outputs() -> SplitOutputs:
    # Urgent CNV cases whose CNV evidence sits below the NORMAL argmax, and normal cases with
    # varying amounts of spurious CNV evidence.
    return outputs_for(
        [
            (0, [0.10, 0.0, 0.0, 0.90]),
            (0, [0.40, 0.0, 0.0, 0.60]),
            (0, [0.95, 0.0, 0.0, 0.05]),
            (3, [0.05, 0.0, 0.0, 0.95]),
            (3, [0.15, 0.0, 0.0, 0.85]),
            (3, [0.12, 0.0, 0.0, 0.88]),
        ]
    )


def sweep(outputs: SplitOutputs, thresholds: tuple[float, ...]):
    rules = load_yaml(settings.triage_rules_path)
    # Disable uncertainty escalation so the sweep isolates the promotion threshold.
    rules["uncertainty"].update(
        minimum_confidence=0.0, minimum_tta_agreement=0.0, maximum_normalized_entropy=1.0
    )
    return sweep_promotion_thresholds(rules, outputs, CLASSES, URGENT, {"NORMAL"}, thresholds)


def test_lower_thresholds_never_reduce_urgent_sensitivity(outputs) -> None:
    rows = sweep(outputs, (0.05, 0.10, 0.20, 0.50, 0.90))
    sensitivities = [row["urgent_tier_sensitivity"] for row in rows]
    assert sensitivities == sorted(sensitivities, reverse=True)
    rates = [row["non_urgent_cases_triaged_urgent_rate"] for row in rows]
    assert rates == sorted(rates, reverse=True)


def test_recommendation_is_the_least_alarming_threshold_that_meets_the_target(outputs) -> None:
    rows = sweep(outputs, (0.05, 0.10, 0.20, 0.50, 0.90))
    recommended = recommend_threshold(rows, target_sensitivity=1.0)
    assert recommended["threshold"] == pytest.approx(0.10)
    assert recommended["urgent_cases_below_urgent_tier"] == 0
    # 0.10 still flags the normal cases with 0.15 and 0.12 CNV evidence.
    assert recommended["non_urgent_cases_triaged_urgent_rate"] == pytest.approx(2 / 3)
    assert recommend_threshold(rows, target_sensitivity=2 / 3)["threshold"] == pytest.approx(0.20)


def test_no_recommendation_when_the_target_is_unreachable(outputs) -> None:
    rows = sweep(outputs, (0.50, 0.90))
    assert recommend_threshold(rows, target_sensitivity=1.0) is None
