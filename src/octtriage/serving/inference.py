from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np
import torch
from PIL import Image

from octtriage.config import load_yaml
from octtriage.data.io import load_scan_bytes
from octtriage.data.quality import QualityAssessment, assess_quality
from octtriage.data.transforms import inference_transform
from octtriage.models.classifier import ModelBundle, gradcam_target_layer, load_bundle
from octtriage.models.gradcam import GradCAM, overlay_heatmap, png_bytes
from octtriage.settings import Settings, settings
from octtriage.triage.engine import TriageEngine


class ModelNotReadyError(RuntimeError):
    pass


@dataclass(frozen=True)
class InferenceResult:
    triage_tier: str
    confidence: float | None
    findings: list[str]
    class_probabilities: dict[str, float]
    escalation_reasons: list[str]
    requires_repeat_acquisition: bool
    quality: dict[str, object]
    model_version: str
    request_sha256: str
    saliency_png: bytes | None
    thumbnail_png: bytes
    disclaimer: str


class ModelInferenceService:
    def __init__(self, app_settings: Settings = settings, device: str | None = None) -> None:
        self.settings = app_settings
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.triage = TriageEngine(app_settings.triage_rules_path)
        self.quality_rules = load_yaml(app_settings.quality_rules_path)
        self.bundle: ModelBundle | None = None

    @property
    def ready(self) -> bool:
        return self.settings.checkpoint_path.is_file()

    def _bundle(self) -> ModelBundle:
        if self.bundle is None:
            if not self.ready:
                raise ModelNotReadyError(
                    "No trained checkpoint is registered. "
                    "Train the Kaggle baseline before inference."
                )
            self.bundle = load_bundle(self.settings.checkpoint_path, self.device)
        return self.bundle

    @staticmethod
    def _volume_quality(
        assessments: list[QualityAssessment], minimum_usable_fraction: float
    ) -> QualityAssessment:
        passed = [item for item in assessments if item.passed]
        usable_fraction = len(passed) / max(len(assessments), 1)
        if len(assessments) == 1:
            return assessments[0]
        metrics: dict[str, float] = {"volume_usable_fraction": usable_fraction}
        if passed:
            for key in passed[0].metrics:
                metrics[f"mean_{key}"] = float(np.mean([item.metrics[key] for item in passed]))
        if usable_fraction < minimum_usable_fraction:
            return QualityAssessment(
                passed=False,
                reasons=["insufficient_usable_volume_frames"],
                metrics=metrics,
            )
        return QualityAssessment(passed=True, reasons=[], metrics=metrics)

    def predict(self, payload: bytes, filename: str) -> InferenceResult:
        request_hash = hashlib.sha256(payload).hexdigest()
        loaded = load_scan_bytes(payload, filename, self.settings.dicom_hash_salt)
        frames = loaded.frames
        assessments = [assess_quality(frame, self.quality_rules) for frame in frames]
        quality = self._volume_quality(
            assessments, float(self.quality_rules["minimum_volume_usable_fraction"])
        )
        preview = frames[len(frames) // 2].copy()
        preview.thumbnail((512, 512))
        thumbnail = png_bytes(preview)
        if not quality.passed:
            decision = self.triage.decide(
                predicted_label=None,
                confidence=None,
                probabilities={},
                tta_agreement=None,
                quality=quality,
            )
            return InferenceResult(
                **decision.to_dict(),
                class_probabilities={},
                quality=quality.to_dict(),
                model_version="quality-gate-only",
                request_sha256=request_hash,
                saliency_png=None,
                thumbnail_png=thumbnail,
                disclaimer=self.triage.disclaimer,
            )

        bundle = self._bundle()
        transform = inference_transform(bundle.image_size)
        usable = [
            frame
            for frame, assessment in zip(frames, assessments, strict=True)
            if assessment.passed
        ]
        if len(usable) > 32:
            indices = np.linspace(0, len(usable) - 1, num=32, dtype=int)
            usable = [usable[index] for index in indices]

        view_probabilities: list[np.ndarray] = []
        view_predictions: list[int] = []
        driver: tuple[float, Image.Image, torch.Tensor, int] | None = None
        bundle.model.eval()
        with torch.inference_mode():
            for frame in usable:
                for view in (frame, frame.transpose(Image.Transpose.FLIP_LEFT_RIGHT)):
                    tensor = transform(view).unsqueeze(0).to(self.device)
                    probabilities = torch.softmax(bundle.model(tensor), dim=1)[0].cpu().numpy()
                    index = int(probabilities.argmax())
                    view_probabilities.append(probabilities)
                    view_predictions.append(index)
                    score = float(probabilities[index])
                    if driver is None or score > driver[0]:
                        driver_tensor = transform(frame).unsqueeze(0).to(self.device)
                        driver = (score, frame, driver_tensor, index)
        assert driver is not None
        stacked = np.stack(view_probabilities)
        maximum_probabilities = stacked.max(axis=0)
        predicted_index = int(maximum_probabilities.argmax())
        predicted_label = bundle.classes[predicted_index]
        probabilities_dict = {
            label: float(maximum_probabilities[index]) for index, label in enumerate(bundle.classes)
        }
        counts = np.bincount(view_predictions, minlength=len(bundle.classes))
        agreement = float(counts.max() / max(len(view_predictions), 1))
        confidence = probabilities_dict[predicted_label]
        decision = self.triage.decide(
            predicted_label=predicted_label,
            confidence=confidence,
            probabilities=probabilities_dict,
            tta_agreement=agreement,
            quality=quality,
            severity="UNKNOWN",
        )

        _, driver_image, driver_tensor, driver_index = driver
        gradcam = GradCAM(
            bundle.model,
            gradcam_target_layer(bundle.model, bundle.backbone),
        )
        try:
            heatmap = gradcam.generate(driver_tensor, driver_index)
        finally:
            gradcam.close()
        saliency = png_bytes(overlay_heatmap(driver_image, heatmap))
        return InferenceResult(
            **decision.to_dict(),
            class_probabilities=probabilities_dict,
            quality=quality.to_dict(),
            model_version=bundle.model_version,
            request_sha256=request_hash,
            saliency_png=saliency,
            thumbnail_png=thumbnail,
            disclaimer=self.triage.disclaimer,
        )
