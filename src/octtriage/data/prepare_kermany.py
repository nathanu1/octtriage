from __future__ import annotations

import argparse
from pathlib import Path

from octtriage.data.manifest import (
    assert_no_patient_leakage,
    discover_kermany_images,
    patient_level_split,
    write_manifest,
)
from octtriage.settings import settings

KAGGLE_DATASET = "paultimothymooney/kermany2018"


def prepare(
    source: str | Path | None,
    output: str | Path,
    *,
    val_fraction: float,
    test_fraction: float,
    seed: int,
) -> Path:
    if source is None:
        try:
            import kagglehub
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("Install kagglehub before downloading the dataset") from exc
        source = kagglehub.dataset_download(KAGGLE_DATASET)
    source_path = Path(source).expanduser().resolve()
    manifest = discover_kermany_images(source_path)
    manifest = patient_level_split(
        manifest,
        val_fraction=val_fraction,
        test_fraction=test_fraction,
        seed=seed,
    )
    assert_no_patient_leakage(manifest)
    destination = write_manifest(manifest, output)
    summary = manifest.groupby(["split", "label"]).size().unstack(fill_value=0)
    print(f"Kaggle source: {source_path}")
    print(f"Patient-safe manifest: {destination}")
    print(summary.to_string())
    print(f"Unique patients: {manifest['patient_id'].nunique()}")
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create a patient-level Kermany OCT2017 train/val/test manifest."
    )
    parser.add_argument("--source", help="Existing Kaggle extraction path; otherwise download")
    parser.add_argument(
        "--output",
        default=str(settings.project_root / "data" / "manifests" / "kermany.csv"),
    )
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--test-fraction", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    prepare(
        args.source,
        args.output,
        val_fraction=args.val_fraction,
        test_fraction=args.test_fraction,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
