from __future__ import annotations

import numpy as np
import pytest

from octtriage.evaluation.calibration import (
    MINIMUM_TEMPERATURE,
    fit_temperature,
    negative_log_likelihood,
    softmax,
)


def overconfident_sample(true_temperature: float, samples: int = 4000, seed: int = 0):
    """Logits whose labels follow softmax(logits / true_temperature)."""
    rng = np.random.default_rng(seed)
    logits = rng.normal(0.0, 3.0, size=(samples, 4))
    probabilities = softmax(logits, true_temperature)
    targets = np.array([rng.choice(4, p=row) for row in probabilities])
    return logits, targets


def test_fit_recovers_the_temperature_of_an_overconfident_model() -> None:
    logits, targets = overconfident_sample(true_temperature=2.0)
    temperature = fit_temperature(logits, targets)
    assert temperature == pytest.approx(2.0, rel=0.1)
    assert negative_log_likelihood(logits, targets, temperature) < negative_log_likelihood(
        logits, targets, 1.0
    )


def test_calibration_never_sharpens_confidence() -> None:
    # Perfectly separated data would otherwise drive the temperature toward zero.
    logits = np.array([[4.0, 0.0], [0.0, 4.0], [5.0, 1.0], [1.0, 6.0]])
    targets = np.array([0, 1, 0, 1])
    assert fit_temperature(logits, targets) == MINIMUM_TEMPERATURE


def test_fit_rejects_misaligned_inputs() -> None:
    with pytest.raises(ValueError):
        fit_temperature(np.zeros((3, 4)), np.zeros(2, dtype=int))


def test_softmax_is_stable_for_large_logits() -> None:
    probabilities = softmax(np.array([[1000.0, 999.0, 0.0]]), temperature=2.0)
    assert np.isfinite(probabilities).all()
    assert probabilities.sum() == pytest.approx(1.0)
