"""Post-hoc temperature scaling for softmax confidence."""

from __future__ import annotations

import math

import numpy as np

# Calibration may soften confidence but never sharpen it: a temperature below one would make the
# model more confident than its raw softmax, and on a perfectly separated validation split the
# unconstrained fit runs toward zero.
MINIMUM_TEMPERATURE = 1.0
MAXIMUM_TEMPERATURE = 10.0


def softmax(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    scaled = np.asarray(logits, dtype=np.float64) / temperature
    scaled -= scaled.max(axis=-1, keepdims=True)
    exponentials = np.exp(scaled)
    return exponentials / exponentials.sum(axis=-1, keepdims=True)


def negative_log_likelihood(logits: np.ndarray, targets: np.ndarray, temperature: float) -> float:
    scaled = np.asarray(logits, dtype=np.float64) / temperature
    scaled -= scaled.max(axis=1, keepdims=True)
    log_normalizer = np.log(np.exp(scaled).sum(axis=1))
    return float(np.mean(log_normalizer - scaled[np.arange(len(targets)), targets]))


def fit_temperature(
    logits: np.ndarray,
    targets: np.ndarray,
    *,
    bounds: tuple[float, float] = (MINIMUM_TEMPERATURE, MAXIMUM_TEMPERATURE),
    iterations: int = 80,
) -> float:
    """Return the temperature within ``bounds`` that minimizes validation negative log-likelihood.

    The likelihood is unimodal in the temperature, so a golden-section search over its
    logarithm finds the optimum without an optimizer dependency.
    """
    values = np.asarray(logits, dtype=np.float64)
    labels = np.asarray(targets, dtype=int)
    if values.ndim != 2 or len(values) == 0 or len(values) != len(labels):
        raise ValueError("Expected logits shaped (samples, classes) with one target per sample")
    low, high = math.log(bounds[0]), math.log(bounds[1])
    ratio = (math.sqrt(5.0) - 1.0) / 2.0

    def loss(log_temperature: float) -> float:
        return negative_log_likelihood(values, labels, math.exp(log_temperature))

    inner_low = high - ratio * (high - low)
    inner_high = low + ratio * (high - low)
    loss_low, loss_high = loss(inner_low), loss(inner_high)
    for _ in range(iterations):
        if loss_low <= loss_high:
            high, inner_high, loss_high = inner_high, inner_low, loss_low
            inner_low = high - ratio * (high - low)
            loss_low = loss(inner_low)
        else:
            low, inner_low, loss_low = inner_low, inner_high, loss_high
            inner_high = low + ratio * (high - low)
            loss_high = loss(inner_high)
    candidates = [bounds[0], math.exp((low + high) / 2.0), bounds[1]]
    return float(min(candidates, key=lambda value: loss(math.log(value))))
