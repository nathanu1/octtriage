from __future__ import annotations

import json
import sqlite3
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


class AuditStore:
    def __init__(self, database_path: str | Path) -> None:
        self.database_path = Path(database_path)
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self.initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database_path)
        connection.row_factory = sqlite3.Row
        return connection

    def initialize(self) -> None:
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS predictions (
                    audit_id TEXT PRIMARY KEY,
                    study_id TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    triage_tier TEXT NOT NULL,
                    confidence REAL,
                    findings_json TEXT NOT NULL,
                    quality_json TEXT NOT NULL,
                    model_version TEXT NOT NULL,
                    request_sha256 TEXT NOT NULL,
                    saliency_path TEXT,
                    thumbnail_path TEXT NOT NULL,
                    response_json TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS overrides (
                    override_id TEXT PRIMARY KEY,
                    audit_id TEXT NOT NULL,
                    original_tier TEXT NOT NULL,
                    new_tier TEXT NOT NULL,
                    clinician_id TEXT NOT NULL,
                    reason TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(audit_id) REFERENCES predictions(audit_id)
                );
                """
            )

    def record_prediction(
        self,
        *,
        study_id: str,
        result: Any,
        response: dict[str, Any],
        saliency_path: str | None,
        thumbnail_path: str,
    ) -> str:
        audit_id = str(response["audit_id"])
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO predictions VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    audit_id,
                    study_id,
                    response["timestamp"],
                    result.triage_tier,
                    result.confidence,
                    json.dumps(result.findings),
                    json.dumps(result.quality),
                    result.model_version,
                    result.request_sha256,
                    saliency_path,
                    thumbnail_path,
                    json.dumps(response),
                ),
            )
        return audit_id

    def list_worklist(self, limit: int = 100) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT p.*,
                       (SELECT new_tier FROM overrides o WHERE o.audit_id = p.audit_id
                        ORDER BY o.created_at DESC LIMIT 1) AS override_tier
                FROM predictions p
                ORDER BY CASE COALESCE(override_tier, triage_tier)
                    WHEN 'URGENT' THEN 2 WHEN 'SEMI_URGENT' THEN 1 ELSE 0 END DESC,
                    created_at ASC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        worklist: list[dict[str, Any]] = []
        for row in rows:
            item = dict(row)
            item["findings"] = json.loads(item.pop("findings_json"))
            item["quality"] = json.loads(item.pop("quality_json"))
            item.pop("response_json", None)
            item["effective_tier"] = item["override_tier"] or item["triage_tier"]
            worklist.append(item)
        return worklist

    def record_override(
        self,
        *,
        audit_id: str,
        new_tier: str,
        clinician_id: str,
        reason: str,
    ) -> dict[str, str]:
        if new_tier not in {"URGENT", "SEMI_URGENT", "ROUTINE"}:
            raise ValueError("Invalid triage tier")
        if not clinician_id.strip() or not reason.strip():
            raise ValueError("Clinician identifier and override reason are required")
        with self._connect() as connection:
            row = connection.execute(
                "SELECT triage_tier FROM predictions WHERE audit_id = ?", (audit_id,)
            ).fetchone()
            if row is None:
                raise KeyError(audit_id)
            latest = connection.execute(
                "SELECT new_tier FROM overrides WHERE audit_id = ? "
                "ORDER BY created_at DESC LIMIT 1",
                (audit_id,),
            ).fetchone()
            original = str(latest["new_tier"] if latest else row["triage_tier"])
            record = {
                "override_id": str(uuid.uuid4()),
                "audit_id": audit_id,
                "original_tier": original,
                "new_tier": new_tier,
                "clinician_id": clinician_id.strip(),
                "reason": reason.strip(),
                "timestamp": utc_now(),
            }
            connection.execute(
                "INSERT INTO overrides VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    record["override_id"],
                    audit_id,
                    original,
                    new_tier,
                    record["clinician_id"],
                    record["reason"],
                    record["timestamp"],
                ),
            )
        return record
