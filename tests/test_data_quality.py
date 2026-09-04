from __future__ import annotations

from pathlib import Path

from PIL import Image

from octtriage.data.quality import assess_quality

RULES = Path(__file__).resolve().parents[1] / "configs" / "quality_rules.yaml"


def test_blank_scan_is_rejected() -> None:
    result = assess_quality(Image.new("L", (224, 224), color=0), RULES)
    assert result.passed is False
    assert "insufficient_dynamic_range" in result.reasons


def test_tiny_scan_is_rejected() -> None:
    result = assess_quality(Image.new("L", (32, 32), color=120), RULES)
    assert result.passed is False
    assert "dimensions_below_minimum" in result.reasons
