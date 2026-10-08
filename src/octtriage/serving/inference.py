from __future__ import annotations

import hashlib
import logging
import threading
from dataclasses import dataclass

import numpy as np
import torch
from PIL import Image

from octtriage.config import load_yaml
from octtriage.data.io import ScanLimits, load_scan_bytes
from octtriage.data.quality import QualityAssessment, assess_quality
from octtriage.data.transforms import inference_transform
from octtriage.models.classifier import ModelBundle, gradcam_target_layer, load_bundle
from octtriage.models.gradcam import GradCAM, overlay_heatmap, png_bytes
from octtriage.models.registry import verify_registered_checkpoint
from octtriage.serving.errors import ModelUnavailableError
from octtriage.settings import Settings, settings
from octtriage.triage.aggregation import decide_from_views, strongest_frame
from octtriage.triage.engine import TriageEngine

logger = logging.getLogger("octtriage.serving")

MAX_FRAMES = 32
INFERENCE_BATCH_SIZE = 32


class ModelNotReadyError(ModelUnavailableError):
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
    saliency_class: str | None
    thumbnail_png: bytes
    disclaimer: str


class ModelInferenceService:
    def __init__(self, app_settings: Settings = settings, device: str | None = None) -> None:
        self.settings = app_settings
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.triage = TriageEngine(app_settings.triage_rules_path)
        self.quality_rules = load_yaml(app_settings.quality_rules_path)
        model_config = load_yaml(app_settings.model_config_path)
        self.non_pathology_classes = set(model_config.get("non_pathology_classes", ["NORMAL"]))
        self.bundle: ModelBundle | None = None
        # Grad-CAM attaches hooks to the shared model, so concurrent requests must not interleave.
        self._model_lock = threading.Lock()

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
            try:
                bundle = load_bundle(self.settings.checkpoint_path, self.device)
                registered = verify_registered_checkpoint(
                    self.settings.model_registry_path,
                    model_version=bundle.model_version,
                    checkpoint_path=bundle.checkpoint_path,
                )
            except Exception as exc:  # noqa: BLE001 - any load failure means no usable model
                raise ModelNotReadyError(f"Model checkpoint could not be loaded: {exc}") from exc
            if not registered:
                logger.warning(
                    "Serving model version %s, which has no model-registry record",
                    bundle.model_version,
                )
            self.bundle = bundle
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

    def _view_probabilities(
        self, bundle: ModelBundle, transform, frames: list[Image.Image]
    ) -> np.ndarray:
        """Return softmax outputs shaped (frames, views, classes) for original and flipped views."""
        views = [
            view
            for frame in frames
            for view in (frame, frame.transpose(Image.Transpose.FLIP_LEFT_RIGHT))
        ]
        batch = torch.stack([transform(view) for view in views])
        outputs: list[np.ndarray] = []
        bundle.model.eval()
        with torch.inference_mode():
            for start in range(0, len(batch), INFERENCE_BATCH_SIZE):
                chunk = batch[start : start + INFERENCE_BATCH_SIZE].to(self.device)
                logits = bundle.model(chunk) / bundle.temperature
                outputs.append(torch.softmax(logits, dim=1).cpu().numpy())
        return np.concatenate(outputs).reshape(len(frames), 2, len(bundle.classes))

    def _saliency(
        self, bundle: ModelBundle, transform, frame: Image.Image, class_index: int
    ) -> bytes:
        # Built outside inference_mode: Grad-CAM must backpropagate through this tensor.
        tensor = transform(frame).unsqueeze(0).to(self.device)
        gradcam = GradCAM(bundle.model, gradcam_target_layer(bundle.model, bundle.backbone))
        try:
            heatmap = gradcam.generate(tensor, class_index)
        finally:
            gradcam.close()
        return png_bytes(overlay_heatmap(frame, heatmap))

    def predict(self, payload: bytes, filename: str) -> InferenceResult:
        request_hash = hashlib.sha256(payload).hexdigest()
        loaded = load_scan_bytes(
            payload,
            filename,
            self.settings.dicom_hash_salt,
            ScanLimits(self.settings.max_scan_frames, self.settings.max_scan_pixels),
        )
        frames = loaded.frames
        assessments = [assess_quality(frame, self.quality_rules) for frame in frames]
        quality = self._volume_quality(
            assessments, float(self.quality_rules["minimum_volume_usable_fraction"])
        )
        preview = frames[len(frames) // 2].copy()
        preview.thumbnail((512, 512))
        thumbnail = png_bytes(preview)
        if not quality.passed:
            decision, _ = decide_from_views(
                self.triage, None, [], self.non_pathology_classes, quality
            )
            return InferenceResult(
                **decision.to_dict(),
                class_probabilities={},
                quality=quality.to_dict(),
                model_version="quality-gate-only",
                request_sha256=request_hash,
                saliency_png=None,
                saliency_class=None,
                thumbnail_png=thumbnail,
                disclaimer=self.triage.disclaimer,
            )

        usable = [
            frame
            for frame, assessment in zip(frames, assessments, strict=True)
            if assessment.passed
        ]
        if len(usable) > MAX_FRAMES:
            indices = np.linspace(0, len(usable) - 1, num=MAX_FRAMES, dtype=int)
            usable = [usable[index] for index in indices]

        with self._model_lock:
            bundle = self._bundle()
            transform = inference_transform(bundle.image_size)
            view_probabilities = self._view_probabilities(bundle, transform, usable)
            decision, evidence = decide_from_views(
                self.triage, view_probabilities, bundle.classes, self.non_pathology_classes, quality
            )
            assert evidence is not None
            # Explain the finding that drives the tier, on the B-scan with the strongest evidence.
            saliency_class = next(
                (finding for finding in decision.findings if finding in bundle.classes),
                evidence.predicted_label,
            )
            class_index = bundle.classes.index(saliency_class)
            driver = usable[strongest_frame(view_probabilities, class_index)]
            saliency = self._saliency(bundle, transform, driver, class_index)

        return InferenceResult(
            **decision.to_dict(),
            class_probabilities=evidence.probabilities,
            quality=quality.to_dict(),
            model_version=bundle.model_version,
            request_sha256=request_hash,
            saliency_png=saliency,
            saliency_class=saliency_class,
            thumbnail_png=thumbnail,
            disclaimer=self.triage.disclaimer,
        )
