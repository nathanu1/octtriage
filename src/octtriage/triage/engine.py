from __future__ import annotations

import math
from collections.abc import Iterator
from dataclasses import asdict, dataclass
from pathlib import Path

from octtriage.config import ConfigurationError, load_yaml
from octtriage.data.quality import QualityAssessment

TIERS = ("ROUTINE", "SEMI_URGENT", "URGENT")


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


def _require_probability(value: object, location: str) -> None:
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        raise ConfigurationError(f"{location} must be a number between 0 and 1") from exc
    if not 0.0 <= number <= 1.0:
        raise ConfigurationError(f"{location} must be between 0 and 1, got {number}")


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
        if not isinstance(priority, dict) or set(priority) != set(TIERS):
            raise ConfigurationError("priority_order must define ROUTINE, SEMI_URGENT, and URGENT")
        try:
            ranks = [int(priority[tier]) for tier in TIERS]
        except (TypeError, ValueError) as exc:
            raise ConfigurationError("priority_order values must be integers") from exc
        # HIGHEST_URGENCY_WINS is only fail-safe if URGENT actually outranks everything else.
        if not ranks[0] < ranks[1] < ranks[2]:
            raise ConfigurationError("priority_order must rank ROUTINE < SEMI_URGENT < URGENT")
        if self.rules["combination_rule"] != "HIGHEST_URGENCY_WINS":
            raise ConfigurationError("Only the fail-safe HIGHEST_URGENCY_WINS rule is supported")
        if self.rules["allow_automatic_downgrade"] is not False:
            raise ConfigurationError("allow_automatic_downgrade must remain false")

        uncertainty = self.rules["uncertainty"]
        if not isinstance(uncertainty, dict) or "escalation_tier" not in uncertainty:
            raise ConfigurationError("uncertainty must be a mapping with an escalation_tier")
        for key in ("minimum_confidence", "minimum_tta_agreement", "maximum_normalized_entropy"):
            _require_probability(uncertainty.get(key), f"uncertainty.{key}")
        thresholds = self.rules.get("finding_probability_thresholds", {})
        if not isinstance(thresholds, dict):
            raise ConfigurationError("finding_probability_thresholds must be a mapping")
        for finding, threshold in thresholds.items():
            _require_probability(threshold, f"finding_probability_thresholds.{finding}")
        if not isinstance(self.rules["quality_failure"], dict):
            raise ConfigurationError("quality_failure must be a mapping")

        # A misspelled tier would otherwise surface as an invalid API response at request time.
        for location, tier in self._referenced_tiers():
            if tier not in TIERS:
                raise ConfigurationError(f"Unknown triage tier {tier!r} at {location}")

    def _referenced_tiers(self) -> Iterator[tuple[str, object]]:
        findings = self.rules["findings"]
        if not isinstance(findings, dict):
            raise ConfigurationError("findings must be a mapping")
        for finding, rule in findings.items():
            if not isinstance(rule, dict):
                raise ConfigurationError(f"findings.{finding} must be a mapping")
            if "severity_tiers" in rule:
                severity_tiers = rule["severity_tiers"]
                if not isinstance(severity_tiers, dict):
                    raise ConfigurationError(f"findings.{finding}.severity_tiers must be a mapping")
                for severity, tier in severity_tiers.items():
                    yield f"findings.{finding}.severity_tiers.{severity}", tier
            elif "default_tier" in rule:
                yield f"findings.{finding}.default_tier", rule["default_tier"]
        uncertainty = self.rules["uncertainty"]
        assert isinstance(uncertainty, dict)
        yield "uncertainty.escalation_tier", uncertainty["escalation_tier"]
        quality_failure = self.rules["quality_failure"]
        assert isinstance(quality_failure, dict)
        if "tier" in quality_failure:
            yield "quality_failure.tier", quality_failure["tier"]

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

    def _priority(self, tier: str) -> int:
        priorities = self.rules["priority_order"]
        assert isinstance(priorities, dict)
        return int(priorities.get(tier, priorities["URGENT"]))

    def _highest(self, tiers: list[str]) -> str:
        return max(tiers, key=self._priority)

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
        tiers = {finding: self._tier_for_finding(finding, severity) for finding in findings}
        # List the finding that drives the tier first so reviewers and saliency maps see it.
        findings.sort(
            key=lambda finding: (-self._priority(tiers[finding]), -probabilities.get(finding, 0.0))
        )
        base_tier = self._highest(list(tiers.values()))
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
