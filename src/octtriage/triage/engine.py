from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from pathlib import Path

from octtriage.config import ConfigurationError, load_yaml
from octtriage.data.quality import QualityAssessment


@dataclass(frozen=True)
class TriageDecision:
    triage_tier: str
    confidence: float | None
    findings: list[str]
    escalation_reasons: list[str]
    requires_repeat_acquisition: bool

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def normalized_entropy(probabilities: dict[str, float]) -> float:
    positive = [max(float(value), 0.0) for value in probabilities.values()]
    total = sum(positive)
    if total <= 0 or len(positive) <= 1:
        return 1.0
    normalized = [value / total for value in positive]
    entropy = -sum(value * math.log(value) for value in normalized if value > 0)
    return entropy / math.log(len(normalized))


class TriageEngine:
    def __init__(self, rules: dict[str, object] | str | Path) -> None:
        self.rules = rules if isinstance(rules, dict) else load_yaml(rules)
        self._validate()

    @property
    def disclaimer(self) -> str:
        return str(self.rules["disclaimer"])

    def _validate(self) -> None:
        required = {
            "disclaimer",
            "priority_order",
            "findings",
            "uncertainty",
            "quality_failure",
            "combination_rule",
            "allow_automatic_downgrade",
        }
        missing = required - set(self.rules)
        if missing:
            raise ConfigurationError(f"Triage rules missing fields: {sorted(missing)}")
        priority = self.rules["priority_order"]
        if not isinstance(priority, dict) or set(priority) != {"ROUTINE", "SEMI_URGENT", "URGENT"}:
            raise ConfigurationError("priority_order must define ROUTINE, SEMI_URGENT, and URGENT")
        if self.rules["combination_rule"] != "HIGHEST_URGENCY_WINS":
            raise ConfigurationError("Only the fail-safe HIGHEST_URGENCY_WINS rule is supported")
        if self.rules["allow_automatic_downgrade"] is not False:
            raise ConfigurationError("allow_automatic_downgrade must remain false")

    def _tier_for_finding(self, finding: str, severity: str) -> str:
        table = self.rules["findings"]
        if not isinstance(table, dict) or finding not in table:
            return "URGENT"
        rule = table[finding]
        if not isinstance(rule, dict):
            return "URGENT"
        if "severity_tiers" in rule:
            severity_tiers = rule["severity_tiers"]
            if not isinstance(severity_tiers, dict):
                return "URGENT"
            return str(severity_tiers.get(severity, severity_tiers.get("UNKNOWN", "URGENT")))
        return str(rule.get("default_tier", "URGENT"))

    def _highest(self, tiers: list[str]) -> str:
        priorities = self.rules["priority_order"]
        assert isinstance(priorities, dict)
        return max(tiers, key=lambda tier: int(priorities.get(tier, priorities["URGENT"])))

    def decide(
        self,
        *,
        predicted_label: str | None,
        confidence: float | None,
        probabilities: dict[str, float],
        tta_agreement: float | None,
        quality: QualityAssessment,
        severity: str = "UNKNOWN",
    ) -> TriageDecision:
        if not quality.passed:
            rule = self.rules["quality_failure"]
            assert isinstance(rule, dict)
            return TriageDecision(
                triage_tier=str(rule.get("tier", "URGENT")),
                confidence=None,
                findings=[str(rule.get("finding", "UNINTERPRETABLE_SCAN"))],
                escalation_reasons=[*quality.reasons, "quality_gate_failed"],
                requires_repeat_acquisition=bool(rule.get("require_repeat_acquisition", True)),
            )

        if predicted_label is None or confidence is None:
            return TriageDecision(
                triage_tier="URGENT",
                confidence=None,
                findings=["MODEL_OUTPUT_UNAVAILABLE"],
                escalation_reasons=["missing_model_output"],
                requires_repeat_acquisition=False,
            )

        findings = [predicted_label]
        thresholds = self.rules.get("finding_probability_thresholds", {})
        if isinstance(thresholds, dict):
            for finding, threshold in thresholds.items():
                promotes = probabilities.get(str(finding), 0.0) >= float(threshold)
                if promotes and finding not in findings:
                    findings.append(str(finding))
        tiers = [self._tier_for_finding(finding, severity) for finding in findings]
        base_tier = self._highest(tiers)
        reasons: list[str] = []
        uncertainty = self.rules["uncertainty"]
        assert isinstance(uncertainty, dict)
        if confidence < float(uncertainty["minimum_confidence"]):
            reasons.append("low_model_confidence")
        if tta_agreement is None or tta_agreement < float(uncertainty["minimum_tta_agreement"]):
            reasons.append("test_time_augmentation_disagreement")
        if normalized_entropy(probabilities) > float(uncertainty["maximum_normalized_entropy"]):
            reasons.append("high_predictive_entropy")
        if reasons:
            base_tier = self._highest([base_tier, str(uncertainty["escalation_tier"])])
        return TriageDecision(
            triage_tier=base_tier,
            confidence=float(confidence),
            findings=findings,
            escalation_reasons=reasons,
            requires_repeat_acquisition=False,
        )
