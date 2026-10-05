from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))


def make_oct_like_image(width: int = 320, height: int = 240, seed: int = 0) -> Image.Image:
    """Synthetic B-scan with centered retinal-layer bands that passes the default quality gate."""
    rng = np.random.default_rng(seed)
    rows = np.arange(height)[:, None]
    pixels = np.full((height, width), 10.0)
    for center, amplitude in [(0.42, 160.0), (0.50, 90.0), (0.58, 200.0)]:
        pixels += amplitude * np.exp(-((rows - center * height) ** 2) / 30.0)
    pixels += rng.normal(0.0, 12.0, pixels.shape)
    return Image.fromarray(np.clip(pixels, 0, 255).astype(np.uint8))


@pytest.fixture
def oct_image():
    return make_oct_like_image
