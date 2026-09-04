from __future__ import annotations

from pathlib import Path

import pytest

from octtriage.data.quality import QualityAssessment
from octtriage.triage.engine import TriageEngine

RULES = Path(__file__).resolve().parents[1] / "configs" / "triage_rules.yaml"
PASS = QualityAssessment(passed=True, reasons=[], metrics={})


def decide(
    engine: TriageEngine,
    label: str,
    confidence: float,
    probabilities: dict[str, float],
    **kwargs,
):
    return engine.decide(
        predicted_label=label,
        confidence=confidence,
        probabilities=probabilities,
        tta_agreement=kwargs.get("tta_agreement", 1.0),
        quality=kwargs.get("quality", PASS),
        severity=kwargs.get("severity", "UNKNOWN"),
    )


def test_normal_high_confidence_can_be_routine() -> None:
    result = decide(
        TriageEngine(RULES),
        "NORMAL",
        0.96,
        {"CNV": 0.01, "DME": 0.01, "DRUSEN": 0.02, "NORMAL": 0.96},
    )
    assert result.triage_tier == "ROUTINE"
    assert result.escalation_reasons == []


@pytest.mark.parametrize("label", ["CNV", "DME"])
def test_urgent_or_unknown_severity_findings_are_urgent(label: str) -> None:
    result = decide(
        TriageEngine(RULES),
        label,
        0.95,
        {"CNV": float(label == "CNV") * 0.94, "DME": float(label == "DME") * 0.94,
         "DRUSEN": 0.03, "NORMAL": 0.03},
    )
    assert result.triage_tier == "URGENT"


def test_low_confidence_never_downgrades() -> None:
    result = decide(
        TriageEngine(RULES),
        "NORMAL",
        0.55,
        {"CNV": 0.15, "DME": 0.15, "DRUSEN": 0.15, "NORMAL": 0.55},
    )
    assert result.triage_tier == "URGENT"
    assert "low_model_confidence" in result.escalation_reasons


def test_tta_disagreement_forces_urgent_review() -> None:
    result = decide(
        TriageEngine(RULES),
        "DRUSEN",
        0.91,
        {"CNV": 0.02, "DME": 0.02, "DRUSEN": 0.91, "NORMAL": 0.05},
        tta_agreement=0.5,
    )
    assert result.triage_tier == "URGENT"
    assert "test_time_augmentation_disagreement" in result.escalation_reasons


def test_urgent_candidate_probability_promotes_normal_argmax() -> None:
    result = decide(
        TriageEngine(RULES),
        "NORMAL",
        0.82,
        {"CNV": 0.27, "DME": 0.02, "DRUSEN": 0.03, "NORMAL": 0.82},
    )
    assert result.triage_tier == "URGENT"
    assert "CNV" in result.findings


def test_quality_failure_forces_urgent_repeat() -> None:
    failed = QualityAssessment(
        passed=False,
        reasons=["insufficient_dynamic_range"],
        metrics={"dynamic_range": 0.0},
    )
    result = decide(
        TriageEngine(RULES),
        "NORMAL",
        0.99,
        {"NORMAL": 0.99},
        quality=failed,
    )
    assert result.triage_tier == "URGENT"
    assert result.findings == ["UNINTERPRETABLE_SCAN"]
    assert result.requires_repeat_acquisition is True


def test_rules_refuse_automatic_downgrade() -> None:
    engine = TriageEngine(RULES)
    unsafe = dict(engine.rules)
    unsafe["allow_automatic_downgrade"] = True
    with pytest.raises(ValueError, match="allow_automatic_downgrade"):
        TriageEngine(unsafe)
