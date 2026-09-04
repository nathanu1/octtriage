from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

_SOURCE_ROOT = Path(__file__).resolve().parents[2]
_WORKING_ROOT = Path.cwd()
PROJECT_ROOT = Path(
    os.getenv(
        "OCT_PROJECT_ROOT",
        _WORKING_ROOT if (_WORKING_ROOT / "configs").is_dir() else _SOURCE_ROOT,
    )
).resolve()


@dataclass(frozen=True)
class Settings:
    project_root: Path = PROJECT_ROOT
    triage_rules_path: Path = PROJECT_ROOT / "configs" / "triage_rules.yaml"
    quality_rules_path: Path = PROJECT_ROOT / "configs" / "quality_rules.yaml"
    model_config_path: Path = PROJECT_ROOT / "configs" / "model.yaml"
    model_registry_path: Path = PROJECT_ROOT / "model_registry" / "registry.json"
    runtime_dir: Path = Path(os.getenv("OCT_RUNTIME_DIR", PROJECT_ROOT / "runtime"))
    checkpoint_path: Path = Path(
        os.getenv(
            "OCT_CHECKPOINT_PATH",
            PROJECT_ROOT / "model_registry" / "checkpoints" / "best.pt",
        )
    )
    database_path: Path = Path(
        os.getenv("OCT_DATABASE_PATH", PROJECT_ROOT / "runtime" / "audit.db")
    )
    dicom_hash_salt: str = os.getenv("OCT_DICOM_HASH_SALT", "prototype-only-change-me")
    max_upload_bytes: int = int(os.getenv("OCT_MAX_UPLOAD_BYTES", str(64 * 1024 * 1024)))

    @property
    def media_dir(self) -> Path:
        return self.runtime_dir / "media"


settings = Settings()
