"""Choose finding-promotion thresholds on the validation split.

The README requires promotion thresholds to be selected through validation and then locked. This
tool sweeps a shared threshold for the urgent findings through the full serving policy on the
validation split and recommends the highest threshold that still meets a target urgent-tier
sensitivity: among thresholds that are safe enough, it flags the fewest non-urgent cases. It never
edits the rules file; lock the chosen value deliberately and confirm it with ``oct-evaluate`` on
the untouched test split.
"""

from __future__ import annotations

import argparse
import copy
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from octtriage.config import load_yaml
from octtriage.data.manifest import assert_no_patient_leakage
from octtriage.evaluation.evaluate import SplitOutputs, policy_decisions, run_split
from octtriage.evaluation.metrics import triage_policy_metrics
from octtriage.models.classifier import load_bundle
from octtriage.models.registry import sha256_file
from octtriage.settings import settings
from octtriage.triage.engine import TriageEngine

DEFAULT_THRESHOLDS = tuple(round(value, 2) for value in np.arange(0.05, 0.951, 0.05))


def sweep_promotion_thresholds(
    rules: dict[str, object],
    outputs: SplitOutputs,
    classes: list[str],
    urgent_classes: set[str],
    non_pathology_classes: set[str],
    thresholds: tuple[float, ...] = DEFAULT_THRESHOLDS,
) -> list[dict[str, float]]:
    """Return policy metrics for each shared urgent-finding promotion threshold."""
    rows: list[dict[str, float]] = []
    for threshold in thresholds:
        candidate = copy.deepcopy(rules)
        configured = dict(candidate.get("finding_probability_thresholds") or {})
        configured.update({finding: float(threshold) for finding in sorted(urgent_classes)})
        candidate["finding_probability_thresholds"] = configured
        decisions = policy_decisions(
            TriageEngine(candidate), outputs, classes, non_pathology_classes
        )
        policy = triage_policy_metrics(
            outputs.targets,
            [decision.triage_tier for decision in decisions],
            [decision.escalation_reasons for decision in decisions],
            classes,
            urgent_classes,
        )
        rows.append(
            {
                "threshold": float(threshold),
                "urgent_tier_sensitivity": policy["urgent_tier_sensitivity"],
                "urgent_cases_below_urgent_tier": policy["urgent_cases_below_urgent_tier"],
                "non_urgent_cases_triaged_urgent_rate": policy[
                    "non_urgent_cases_triaged_urgent_rate"
                ],
                "urgent_tier_fraction": policy["urgent_tier_fraction"],
            }
        )
    return rows


def recommend_threshold(
    rows: list[dict[str, float]], target_sensitivity: float
) -> dict[str, float] | None:
    """Return the highest threshold meeting the target, or None when none does.

    Raising the threshold can only remove promotions, so the highest eligible threshold is also
    the one that flags the fewest non-urgent cases as URGENT.
    """
    eligible = [row for row in rows if row["urgent_tier_sensitivity"] >= target_sensitivity]
    return max(eligible, key=lambda row: row["threshold"]) if eligible else None


def tune(
    manifest_path: Path,
    checkpoint_path: Path,
    output_path: Path,
    device_name: str,
    target_sensitivity: float,
    split: str = "val",
) -> dict[str, object]:
    if not 0.0 < target_sensitivity <= 1.0:
        raise ValueError("target_sensitivity must be in (0, 1]")
    if split == "test":
        raise ValueError("Tune thresholds on the validation split; the test split confirms them")
    device = torch.device(device_name)
    bundle = load_bundle(checkpoint_path, device)
    manifest = pd.read_csv(manifest_path)
    assert_no_patient_leakage(manifest)
    model_config = load_yaml(settings.model_config_path)
    urgent_classes = set(model_config["urgent_classes"])
    non_pathology_classes = set(model_config.get("non_pathology_classes", ["NORMAL"]))
    rules = load_yaml(settings.triage_rules_path)
    outputs = run_split(bundle, manifest, split, device)
    rows = sweep_promotion_thresholds(
        rules, outputs, bundle.classes, urgent_classes, non_pathology_classes
    )
    recommended = recommend_threshold(rows, target_sensitivity)
    report: dict[str, object] = {
        "split": split,
        "target_urgent_tier_sensitivity": target_sensitivity,
        "recommended": recommended,
        "swept_findings": sorted(urgent_classes),
        "current_thresholds": rules.get("finding_probability_thresholds", {}),
        "sweep": rows,
        "provenance": {
            "model_version": bundle.model_version,
            "temperature": bundle.temperature,
            "checkpoint_sha256": sha256_file(checkpoint_path),
            "manifest_sha256": sha256_file(manifest_path),
            "triage_rules_sha256": sha256_file(settings.triage_rules_path),
            "images": int(len(outputs.targets)),
            "patients": int(len(np.unique(outputs.patient_ids))),
            "tuned_at": datetime.now(timezone.utc).isoformat(),
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Recommend urgent-finding promotion thresholds from the validation split"
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=settings.project_root / "data/manifests/kermany.csv",
    )
    parser.add_argument("--checkpoint", type=Path, default=settings.checkpoint_path)
    parser.add_argument(
        "--output",
        type=Path,
        default=settings.project_root / "reports/metrics/threshold_sweep.json",
    )
    parser.add_argument("--target-sensitivity", type=float, default=0.97)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    report = tune(
        args.manifest, args.checkpoint, args.output, args.device, args.target_sensitivity
    )
    for row in report["sweep"]:
        print(
            f"threshold={row['threshold']:.2f} "
            f"urgent_tier_sensitivity={row['urgent_tier_sensitivity']:.3f} "
            f"non_urgent_flagged_urgent={row['non_urgent_cases_triaged_urgent_rate']:.3f}"
        )
    recommended = report["recommended"]
    if recommended is None:
        print(
            f"No threshold reaches urgent-tier sensitivity {args.target_sensitivity:.2f}; "
            "keep the current rules and investigate the misses."
        )
        return
    print(
        "\nRecommended (lock deliberately in configs/triage_rules.yaml, then confirm with "
        "oct-evaluate on the test split):\nfinding_probability_thresholds:"
    )
    for finding in report["swept_findings"]:
        print(f"  {finding}: {recommended['threshold']:.2f}")
    print(f"Report written to {args.output}")


if __name__ == "__main__":
    main()
