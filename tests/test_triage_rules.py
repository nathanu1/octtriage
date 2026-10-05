from __future__ import annotations

import copy
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


def test_driving_finding_is_listed_first() -> None:
    result = decide(
        TriageEngine(RULES),
        "NORMAL",
        0.82,
        {"CNV": 0.27, "DME": 0.02, "DRUSEN": 0.03, "NORMAL": 0.82},
    )
    assert result.findings == ["CNV", "NORMAL"]


def _rules_with(mutate) -> dict:
    rules = copy.deepcopy(TriageEngine(RULES).rules)
    mutate(rules)
    return rules


def test_rules_refuse_inverted_priority_order() -> None:
    def invert(rules: dict) -> None:
        rules["priority_order"] = {"ROUTINE": 5, "SEMI_URGENT": 1, "URGENT": 2}

    with pytest.raises(ValueError, match="ROUTINE < SEMI_URGENT < URGENT"):
        TriageEngine(_rules_with(invert))


@pytest.mark.parametrize(
    "mutate",
    [
        lambda rules: rules["findings"]["CNV"].update(default_tier="URGNET"),
        lambda rules: rules["findings"]["DME"]["severity_tiers"].update(MILD="SOON"),
        lambda rules: rules["uncertainty"].update(escalation_tier="HIGH"),
        lambda rules: rules["quality_failure"].update(tier="STAT"),
    ],
)
def test_rules_refuse_unknown_tier_names(mutate) -> None:
    with pytest.raises(ValueError, match="Unknown triage tier"):
        TriageEngine(_rules_with(mutate))


@pytest.mark.parametrize(
    "mutate",
    [
        lambda rules: rules["uncertainty"].update(minimum_confidence=80),
        lambda rules: rules["uncertainty"].pop("minimum_tta_agreement"),
        lambda rules: rules["finding_probability_thresholds"].update(CNV=-0.1),
    ],
)
def test_rules_refuse_out_of_range_thresholds(mutate) -> None:
    with pytest.raises(ValueError, match="between 0 and 1"):
        TriageEngine(_rules_with(mutate))


def test_unknown_model_class_routes_to_urgent() -> None:
    result = decide(
        TriageEngine(RULES),
        "MACULAR_HOLE",
        0.97,
        {"MACULAR_HOLE": 0.97, "NORMAL": 0.03},
    )
    assert result.triage_tier == "URGENT"
