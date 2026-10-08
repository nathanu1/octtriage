from __future__ import annotations

import io

import numpy as np
import pytest
from PIL import Image

from octtriage.data.io import ScanLimits, ScanTooLargeError, UnsupportedScanError, load_scan_bytes


def encoded(image: Image.Image, fmt: str, **kwargs) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format=fmt, **kwargs)
    return buffer.getvalue()


def test_sixteen_bit_scan_is_rescaled_not_saturated() -> None:
    ramp = np.linspace(0, 4000, 256 * 256).reshape(256, 256).astype(np.uint16)
    loaded = load_scan_bytes(encoded(Image.fromarray(ramp), "PNG"), "scan.png", "salt")
    pixels = np.asarray(loaded.frames[0])
    assert loaded.frames[0].mode == "RGB"
    assert float(np.mean(pixels[..., 0] == 255)) < 0.05
    assert pixels[..., 0].max() - pixels[..., 0].min() > 200


def test_eight_bit_scan_pixels_are_unchanged() -> None:
    gray = np.tile(np.arange(256, dtype=np.uint8), (128, 1))
    loaded = load_scan_bytes(encoded(Image.fromarray(gray), "PNG"), "scan.png", "salt")
    assert np.array_equal(np.asarray(loaded.frames[0])[..., 0], gray)


def test_multi_page_tiff_volume_keeps_every_bscan() -> None:
    pages = [Image.new("L", (64, 48), color=value) for value in (10, 120, 240)]
    payload = encoded(pages[0], "TIFF", save_all=True, append_images=pages[1:])
    loaded = load_scan_bytes(payload, "volume.tiff", "salt")
    assert len(loaded.frames) == 3
    assert [frame.getpixel((0, 0))[0] for frame in loaded.frames] == [10, 120, 240]


def test_undecodable_image_is_rejected() -> None:
    with pytest.raises(UnsupportedScanError, match="Unable to decode image"):
        load_scan_bytes(b"not an image", "scan.png", "salt")


def test_unsupported_extension_is_rejected() -> None:
    with pytest.raises(UnsupportedScanError, match="Unsupported file extension"):
        load_scan_bytes(b"payload", "scan.gif", "salt")


def test_volume_with_too_many_pages_is_rejected_before_decoding() -> None:
    pages = [Image.new("L", (16, 16)) for _ in range(6)]
    payload = encoded(pages[0], "TIFF", save_all=True, append_images=pages[1:])
    with pytest.raises(ScanTooLargeError, match="6 frames"):
        load_scan_bytes(payload, "volume.tiff", "salt", ScanLimits(max_frames=5))


def test_scan_exceeding_the_pixel_budget_is_rejected() -> None:
    pages = [Image.new("L", (100, 100)) for _ in range(3)]
    payload = encoded(pages[0], "TIFF", save_all=True, append_images=pages[1:])
    with pytest.raises(ScanTooLargeError, match="pixels"):
        load_scan_bytes(payload, "volume.tiff", "salt", ScanLimits(max_pixels=25_000))
    assert len(load_scan_bytes(payload, "volume.tiff", "salt", ScanLimits()).frames) == 3
