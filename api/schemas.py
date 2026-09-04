from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

Tier = Literal["URGENT", "SEMI_URGENT", "ROUTINE"]


class QualityResponse(BaseModel):
    passed: bool
    reasons: list[str]
    metrics: dict[str, float]


class TriageResponse(BaseModel):
    audit_id: str
    study_id: str
    triage_tier: Tier
    confidence: float | None
    findings: list[str]
    class_probabilities: dict[str, float]
    saliency_map_url: str | None
    thumbnail_url: str
    model_version: str
    timestamp: str
    quality: QualityResponse
    escalation_reasons: list[str]
    requires_clinician_review: bool = True
    requires_repeat_acquisition: bool
    disclaimer: str


class OverrideRequest(BaseModel):
    audit_id: str
    new_tier: Tier
    clinician_id: str = Field(min_length=1, max_length=128)
    reason: str = Field(min_length=3, max_length=1000)


class OverrideResponse(BaseModel):
    override_id: str
    audit_id: str
    original_tier: Tier
    new_tier: Tier
    clinician_id: str
    reason: str
    timestamp: str
