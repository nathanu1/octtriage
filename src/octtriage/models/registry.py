from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def register_model(
    registry_path: str | Path,
    *,
    model_version: str,
    checkpoint_path: str | Path,
    metrics: dict[str, float],
    training_manifest: str | Path,
) -> dict[str, Any]:
    path = Path(registry_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {"models": []}
    if path.is_file():
        payload = json.loads(path.read_text(encoding="utf-8"))
    record = {
        "model_version": model_version,
        "checkpoint_path": str(Path(checkpoint_path)),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "registered_at": datetime.now(timezone.utc).isoformat(),
        "training_manifest": str(Path(training_manifest)),
        "training_manifest_sha256": sha256_file(training_manifest),
        "metrics": metrics,
    }
    payload.setdefault("models", []).append(record)
    payload["active_model_version"] = model_version
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return record


def verify_registered_checkpoint(
    registry_path: str | Path, *, model_version: str, checkpoint_path: str | Path
) -> bool:
    """Check a checkpoint against the registry record for its model version.

    Returns True when the file matches its record and False when the version is not
    registered. Raises ValueError when the version is registered with a different hash, which
    means the checkpoint was replaced or corrupted after registration.
    """
    path = Path(registry_path)
    if not path.is_file():
        return False
    payload = json.loads(path.read_text(encoding="utf-8"))
    records = [
        record
        for record in payload.get("models", [])
        if record.get("model_version") == model_version
    ]
    if not records:
        return False
    digest = sha256_file(checkpoint_path)
    if any(record.get("checkpoint_sha256") == digest for record in records):
        return True
    raise ValueError(
        f"Checkpoint {Path(checkpoint_path).name} does not match the SHA-256 registered for "
        f"model version {model_version}"
    )
