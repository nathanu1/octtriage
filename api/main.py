from __future__ import annotations

import json
import logging
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse

from api.schemas import (
    OverrideRequest,
    OverrideResponse,
    PredictionDetailResponse,
    TriageResponse,
)
from octtriage.data.io import UnsupportedScanError
from octtriage.serving.audit import AuditStore
from octtriage.serving.errors import ModelUnavailableError
from octtriage.settings import Settings, settings

logger = logging.getLogger("octtriage.audit")
API_DISCLAIMER = (
    "Clinical decision-support prototype. This output is not a diagnosis and requires "
    "review by a qualified clinician. Do not use this prototype for patient care."
)


class LazyPredictor:
    """Keep API health and contract endpoints usable before heavyweight ML imports."""

    def __init__(self, app_settings: Settings) -> None:
        self.settings = app_settings
        self._service = None
        self._lock = threading.Lock()

    @property
    def ready(self) -> bool:
        return self.settings.checkpoint_path.is_file()

    def predict(self, payload: bytes, filename: str):
        if not self.ready:
            raise ModelUnavailableError(
                "No trained checkpoint is registered. Train the Kaggle baseline before inference."
            )
        with self._lock:
            if self._service is None:
                from octtriage.serving.inference import ModelInferenceService

                self._service = ModelInferenceService(self.settings)
        return self._service.predict(payload, filename)


def _dump(model: Any) -> dict[str, Any]:
    return model.model_dump() if hasattr(model, "model_dump") else model.dict()


def _media_url(path: str | None) -> str | None:
    return f"/v1/media/{Path(path).name}" if path else None


def create_app(
    predictor: Any | None = None,
    audit_store: AuditStore | None = None,
    app_settings: Settings = settings,
) -> FastAPI:
    app = FastAPI(
        title="OCT Triage Decision-Support API",
        version="0.1.0",
        description="Prototype decision support only; not an autonomous diagnostic device.",
    )
    app.state.predictor = predictor or LazyPredictor(app_settings)
    app.state.audit = audit_store or AuditStore(app_settings.database_path)
    app.state.settings = app_settings

    @app.exception_handler(HTTPException)
    async def http_error(_request: Request, exc: HTTPException) -> JSONResponse:
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "detail": exc.detail,
                "decision_support_only": True,
                "disclaimer": API_DISCLAIMER,
            },
            headers=exc.headers,
        )

    @app.exception_handler(RequestValidationError)
    async def validation_error(
        _request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        return JSONResponse(
            status_code=422,
            content={
                "detail": exc.errors(),
                "decision_support_only": True,
                "disclaimer": API_DISCLAIMER,
            },
        )

    @app.get("/health")
    def health() -> dict[str, object]:
        return {
            "status": "ok",
            "model_ready": bool(app.state.predictor.ready),
            "decision_support_only": True,
            "disclaimer": API_DISCLAIMER,
        }

    # A plain ``def`` endpoint runs in FastAPI's thread pool, so model inference does not block
    # the event loop (and with it /health checks) while a scan is processed.
    @app.post("/v1/triage", response_model=TriageResponse)
    def triage_scan(
        file: UploadFile = File(...),
        study_id: str | None = Form(default=None),
    ) -> TriageResponse:
        payload = file.file.read(app_settings.max_upload_bytes + 1)
        if not payload:
            raise HTTPException(status_code=422, detail="Uploaded scan is empty")
        if len(payload) > app_settings.max_upload_bytes:
            raise HTTPException(status_code=413, detail="Uploaded scan exceeds the size limit")
        try:
            result = app.state.predictor.predict(payload, file.filename or "scan")
        except UnsupportedScanError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except ModelUnavailableError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

        audit_id = str(uuid.uuid4())
        resolved_study_id = study_id.strip() if study_id and study_id.strip() else audit_id
        timestamp = datetime.now(timezone.utc).isoformat()
        media_dir = Path(app_settings.media_dir)
        media_dir.mkdir(parents=True, exist_ok=True)
        thumbnail_path = media_dir / f"{audit_id}-thumbnail.png"
        thumbnail_path.write_bytes(result.thumbnail_png)
        saliency_path: Path | None = None
        if result.saliency_png is not None:
            saliency_path = media_dir / f"{audit_id}-saliency.png"
            saliency_path.write_bytes(result.saliency_png)
        response = TriageResponse(
            audit_id=audit_id,
            study_id=resolved_study_id,
            triage_tier=result.triage_tier,
            confidence=result.confidence,
            findings=result.findings,
            class_probabilities=result.class_probabilities,
            saliency_map_url=f"/v1/media/{saliency_path.name}" if saliency_path else None,
            saliency_class=result.saliency_class,
            thumbnail_url=f"/v1/media/{thumbnail_path.name}",
            model_version=result.model_version,
            timestamp=timestamp,
            quality=result.quality,
            escalation_reasons=result.escalation_reasons,
            requires_clinician_review=True,
            requires_repeat_acquisition=result.requires_repeat_acquisition,
            disclaimer=result.disclaimer,
        )
        serialized = _dump(response)
        app.state.audit.record_prediction(
            study_id=resolved_study_id,
            result=result,
            response=serialized,
            saliency_path=str(saliency_path) if saliency_path else None,
            thumbnail_path=str(thumbnail_path),
        )
        logger.info(
            json.dumps(
                {
                    "event": "prediction_recorded",
                    "audit_id": audit_id,
                    "triage_tier": result.triage_tier,
                    "model_version": result.model_version,
                    "request_sha256": result.request_sha256,
                    "timestamp": timestamp,
                }
            )
        )
        return response

    @app.get("/v1/worklist")
    def worklist(limit: int = 100) -> dict[str, object]:
        items = app.state.audit.list_worklist(min(max(limit, 1), 500))
        for item in items:
            # Expose media URLs; server filesystem paths stay internal to the audit store.
            item["thumbnail_url"] = _media_url(item.pop("thumbnail_path"))
            item["saliency_map_url"] = _media_url(item.pop("saliency_path"))
        return {
            "items": items,
            "disclaimer": "Decision support only; every item requires clinician review.",
        }

    @app.get("/v1/predictions/{audit_id}", response_model=PredictionDetailResponse)
    def prediction_detail(audit_id: str) -> PredictionDetailResponse:
        try:
            record = app.state.audit.get_prediction(audit_id)
        except KeyError as exc:
            raise HTTPException(
                status_code=404, detail="Prediction audit record not found"
            ) from exc
        return PredictionDetailResponse(**record)

    @app.get("/v1/media/{filename}")
    def media(filename: str) -> FileResponse:
        safe_name = Path(filename).name
        path = Path(app_settings.media_dir) / safe_name
        if not path.is_file():
            raise HTTPException(status_code=404, detail="Media not found")
        return FileResponse(path, media_type="image/png")

    @app.post("/v1/overrides", response_model=OverrideResponse)
    def override(request: OverrideRequest) -> OverrideResponse:
        try:
            record = app.state.audit.record_override(**_dump(request))
        except KeyError as exc:
            raise HTTPException(
                status_code=404, detail="Prediction audit record not found"
            ) from exc
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        logger.info(
            json.dumps(
                {
                    "event": "clinician_override_recorded",
                    "audit_id": record["audit_id"],
                    "original_tier": record["original_tier"],
                    "new_tier": record["new_tier"],
                    "timestamp": record["timestamp"],
                }
            )
        )
        return OverrideResponse(**record)

    return app


app = create_app()
