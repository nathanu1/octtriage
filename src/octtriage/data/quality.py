from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from octtriage.config import load_yaml


@dataclass(frozen=True)
class QualityAssessment:
    passed: bool
    reasons: list[str]
    metrics: dict[str, float]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _laplacian_variance(gray: np.ndarray) -> float:
    padded = np.pad(gray.astype(np.float32), 1, mode="edge")
    center = padded[1:-1, 1:-1]
    laplacian = (
        -4.0 * center
        + padded[:-2, 1:-1]
        + padded[2:, 1:-1]
        + padded[1:-1, :-2]
        + padded[1:-1, 2:]
    )
    return float(np.var(laplacian))


def assess_quality(image: Image.Image, rules: dict[str, object] | str | Path) -> QualityAssessment:
    if not isinstance(rules, dict):
        rules = load_yaml(rules)
    gray = np.asarray(image.convert("L"), dtype=np.float32)
    height, width = gray.shape
    p01, p99 = np.percentile(gray, [1.0, 99.0])
    dynamic_range = float(p99 - p01)
    blur_variance = _laplacian_variance(gray)
    saturation_fraction = float(np.mean((gray <= 2.0) | (gray >= 253.0)))

    threshold = float(np.percentile(gray, float(rules["foreground_percentile"])))
    foreground = gray > threshold
    if foreground.any():
        ys, xs = np.nonzero(foreground)
        center_x = float(xs.mean() / max(width - 1, 1))
        center_y = float(ys.mean() / max(height - 1, 1))
        center_offset = float(((center_x - 0.5) ** 2 + (center_y - 0.5) ** 2) ** 0.5)
    else:
        center_offset = 1.0

    metrics = {
        "width": float(width),
        "height": float(height),
        "dynamic_range": dynamic_range,
        "blur_variance": blur_variance,
        "saturation_fraction": saturation_fraction,
        "center_offset": center_offset,
    }
    reasons: list[str] = []
    if width < int(rules["minimum_width"]) or height < int(rules["minimum_height"]):
        reasons.append("dimensions_below_minimum")
    if dynamic_range < float(rules["minimum_dynamic_range"]):
        reasons.append("insufficient_dynamic_range")
    if blur_variance < float(rules["minimum_blur_variance"]):
        reasons.append("excessive_blur_or_low_structure")
    if saturation_fraction > float(rules["maximum_saturation_fraction"]):
        reasons.append("excessive_saturation")
    if center_offset > float(rules["maximum_center_offset"]):
        reasons.append("foreground_off_center")
    return QualityAssessment(passed=not reasons, reasons=reasons, metrics=metrics)
