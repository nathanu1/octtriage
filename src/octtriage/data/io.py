from __future__ import annotations

import hashlib
import io
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image


class UnsupportedScanError(ValueError):
    pass


@dataclass(frozen=True)
class LoadedScan:
    frames: list[Image.Image]
    source_type: str
    patient_hash: str | None = None


def _normalize_array(array: np.ndarray) -> np.ndarray:
    pixels = np.asarray(array, dtype=np.float32)
    finite = pixels[np.isfinite(pixels)]
    if finite.size == 0:
        raise UnsupportedScanError("Scan has no finite pixel values")
    low, high = np.percentile(finite, [1.0, 99.0])
    if high <= low:
        return np.zeros(pixels.shape, dtype=np.uint8)
    pixels = np.clip((pixels - low) / (high - low), 0.0, 1.0)
    return (pixels * 255.0).astype(np.uint8)


def _array_to_frames(array: np.ndarray) -> list[Image.Image]:
    pixels = np.asarray(array)
    if pixels.ndim == 2:
        arrays = [pixels]
    elif pixels.ndim == 3 and pixels.shape[-1] in {3, 4}:
        image = Image.fromarray(_normalize_array(pixels)).convert("RGB")
        return [image]
    elif pixels.ndim == 3:
        arrays = [pixels[index] for index in range(pixels.shape[0])]
    elif pixels.ndim == 4 and pixels.shape[-1] in {3, 4}:
        return [Image.fromarray(_normalize_array(frame)).convert("RGB") for frame in pixels]
    else:
        raise UnsupportedScanError(f"Unsupported pixel-array shape: {pixels.shape}")
    return [Image.fromarray(_normalize_array(frame)).convert("RGB") for frame in arrays]


def load_dicom_bytes(payload: bytes, patient_hash_salt: str) -> LoadedScan:
    try:
        import pydicom
    except ImportError as exc:  # pragma: no cover - dependency error is deployment-specific
        raise UnsupportedScanError("DICOM support requires pydicom") from exc
    try:
        dataset = pydicom.dcmread(io.BytesIO(payload), force=False)
        pixels = dataset.pixel_array
        try:
            from pydicom.pixels import apply_modality_lut, apply_voi_lut

            pixels = apply_modality_lut(pixels, dataset)
            pixels = apply_voi_lut(pixels, dataset)
        except (ImportError, NotImplementedError, ValueError):
            pass
        if str(getattr(dataset, "PhotometricInterpretation", "")).upper() == "MONOCHROME1":
            pixels = np.max(pixels) - pixels
        frames = _array_to_frames(pixels)
    except Exception as exc:  # noqa: BLE001
        raise UnsupportedScanError(f"Unable to decode DICOM pixel data: {exc}") from exc
    patient_id = str(getattr(dataset, "PatientID", "")).strip()
    patient_hash = None
    if patient_id:
        patient_hash = hashlib.sha256(
            f"{patient_hash_salt}:{patient_id}".encode()
        ).hexdigest()
    return LoadedScan(frames=frames, source_type="dicom", patient_hash=patient_hash)


def load_image_bytes(payload: bytes) -> LoadedScan:
    try:
        image = Image.open(io.BytesIO(payload)).convert("RGB")
        image.load()
    except Exception as exc:  # noqa: BLE001
        raise UnsupportedScanError(f"Unable to decode image: {exc}") from exc
    return LoadedScan(frames=[image], source_type="image")


def load_scan_bytes(payload: bytes, filename: str, patient_hash_salt: str) -> LoadedScan:
    suffix = Path(filename).suffix.lower()
    if suffix in {".dcm", ".dicom"} or payload[:132].endswith(b"DICM"):
        return load_dicom_bytes(payload, patient_hash_salt)
    if suffix not in {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}:
        raise UnsupportedScanError(f"Unsupported file extension: {suffix or '<none>'}")
    return load_image_bytes(payload)
