from __future__ import annotations

import io

import numpy as np
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, SecondaryCaptureImageStorage, generate_uid

from octtriage.data.io import load_scan_bytes


def dicom_payload() -> bytes:
    file_meta = FileMetaDataset()
    file_meta.MediaStorageSOPClassUID = SecondaryCaptureImageStorage
    file_meta.MediaStorageSOPInstanceUID = generate_uid()
    file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    dataset = FileDataset(None, {}, file_meta=file_meta, preamble=b"\0" * 128)
    dataset.SOPClassUID = file_meta.MediaStorageSOPClassUID
    dataset.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID
    dataset.PatientID = "TEST-PATIENT-001"
    dataset.Rows = 32
    dataset.Columns = 48
    dataset.SamplesPerPixel = 1
    dataset.PhotometricInterpretation = "MONOCHROME2"
    dataset.BitsAllocated = 16
    dataset.BitsStored = 16
    dataset.HighBit = 15
    dataset.PixelRepresentation = 0
    dataset.PixelData = np.arange(32 * 48, dtype=np.uint16).reshape(32, 48).tobytes()
    buffer = io.BytesIO()
    dataset.save_as(buffer, enforce_file_format=True)
    return buffer.getvalue()


def test_dicom_pixels_and_patient_identifier_are_safely_loaded() -> None:
    loaded = load_scan_bytes(dicom_payload(), "scan.dcm", "unit-test-salt")
    assert loaded.source_type == "dicom"
    assert len(loaded.frames) == 1
    assert loaded.frames[0].size == (48, 32)
    assert loaded.patient_hash is not None
    assert loaded.patient_hash != "TEST-PATIENT-001"
