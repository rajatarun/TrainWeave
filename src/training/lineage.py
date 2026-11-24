"""SQLite WAL-backed adapter lineage registry for TrainWeave.

Links sanitized training dataset SHA-256 digests -> LoRA/GGUF artifact manifest
SHA-256 digests -> optional ``chaos-cert-v1`` certification digests in a durable
local SQLite database configured with Write-Ahead Logging (WAL).
"""

from __future__ import annotations

import sqlite3
from contextlib import closing
from datetime import UTC, datetime
from pathlib import Path
from types import TracebackType
from typing import Any, Self

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS adapter_lineage (
    adapter_id TEXT PRIMARY KEY,
    adapter_name TEXT NOT NULL,
    base_model TEXT NOT NULL,
    dataset_sha256 TEXT NOT NULL,
    manifest_sha256 TEXT NOT NULL,
    chaos_cert_sha256 TEXT,
    certified INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_adapter_lineage_base_model
    ON adapter_lineage (base_model);

CREATE INDEX IF NOT EXISTS idx_adapter_lineage_certified
    ON adapter_lineage (certified);
"""


def _utc_now_iso() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _row_to_dict(row: sqlite3.Row) -> dict[str, Any]:
    return {
        "adapter_id": str(row["adapter_id"]),
        "adapter_name": str(row["adapter_name"]),
        "base_model": str(row["base_model"]),
        "dataset_sha256": str(row["dataset_sha256"]),
        "manifest_sha256": str(row["manifest_sha256"]),
        "chaos_cert_sha256": row["chaos_cert_sha256"],
        "certified": bool(row["certified"]),
        "created_at": str(row["created_at"]),
    }


class AdapterLineageRegistry:
    """Durable SQLite WAL registry for adapter dataset-to-certification lineage."""

    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._journal_mode = self._initialize()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path), timeout=30.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        return conn

    def _initialize(self) -> str:
        with closing(self._connect()) as conn:
            mode_row = conn.execute("PRAGMA journal_mode=WAL;").fetchone()
            journal_mode = str(mode_row[0]).lower() if mode_row else "unknown"
            conn.executescript(_SCHEMA_SQL)
            conn.commit()
            return journal_mode

    @property
    def journal_mode(self) -> str:
        """Return the active SQLite journal mode reported on initialization."""
        return self._journal_mode

    def record(
        self,
        adapter_id: str,
        adapter_name: str,
        base_model: str,
        dataset_sha256: str,
        manifest_sha256: str,
        chaos_cert_sha256: str | None = None,
        certified: bool = False,
        created_at: str | None = None,
    ) -> dict[str, Any]:
        """Insert or update an adapter lineage entry and return the stored record."""
        for field_name, value in (
            ("adapter_id", adapter_id),
            ("adapter_name", adapter_name),
            ("base_model", base_model),
            ("dataset_sha256", dataset_sha256),
            ("manifest_sha256", manifest_sha256),
        ):
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{field_name} must be a non-empty string")

        timestamp = created_at if created_at is not None else _utc_now_iso()
        certified_int = 1 if certified else 0

        with closing(self._connect()) as conn:
            conn.execute(
                """
                INSERT INTO adapter_lineage (
                    adapter_id,
                    adapter_name,
                    base_model,
                    dataset_sha256,
                    manifest_sha256,
                    chaos_cert_sha256,
                    certified,
                    created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(adapter_id) DO UPDATE SET
                    adapter_name = excluded.adapter_name,
                    base_model = excluded.base_model,
                    dataset_sha256 = excluded.dataset_sha256,
                    manifest_sha256 = excluded.manifest_sha256,
                    chaos_cert_sha256 = excluded.chaos_cert_sha256,
                    certified = excluded.certified,
                    created_at = excluded.created_at
                """,
                (
                    adapter_id,
                    adapter_name,
                    base_model,
                    dataset_sha256,
                    manifest_sha256,
                    chaos_cert_sha256,
                    certified_int,
                    timestamp,
                ),
            )
            conn.commit()
            row = conn.execute(
                "SELECT * FROM adapter_lineage WHERE adapter_id = ?",
                (adapter_id,),
            ).fetchone()

        assert row is not None
        return _row_to_dict(row)

    def get(self, adapter_id: str) -> dict[str, Any] | None:
        """Fetch a single adapter lineage record by ``adapter_id``, or ``None``."""
        with closing(self._connect()) as conn:
            row = conn.execute(
                "SELECT * FROM adapter_lineage WHERE adapter_id = ?",
                (adapter_id,),
            ).fetchone()
        return _row_to_dict(row) if row is not None else None

    def list_lineage(
        self,
        base_model: str | None = None,
        certified_only: bool = False,
    ) -> list[dict[str, Any]]:
        """List adapter lineage records with optional ``base_model`` and certification filters."""
        clauses: list[str] = []
        params: list[Any] = []

        if base_model is not None:
            clauses.append("base_model = ?")
            params.append(base_model)
        if certified_only:
            clauses.append("certified = 1")

        where_sql = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        query = (
            f"SELECT * FROM adapter_lineage {where_sql} "
            "ORDER BY created_at DESC, adapter_id ASC"
        )

        with closing(self._connect()) as conn:
            rows = conn.execute(query, params).fetchall()
        return [_row_to_dict(row) for row in rows]

    def close(self) -> None:
        """No-op hook for API parity with persistent-connection registries."""

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()
