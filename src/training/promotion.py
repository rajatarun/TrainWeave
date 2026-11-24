"""Atomic SQLite WAL-backed adapter promotion and rollback manager for TrainWeave.

Governs promotion of fine-tuned LoRA/GGUF adapters registered in
:class:`AdapterLineageRegistry`, enforcing ``chaos-cert-v1`` certification gates
and maintaining a durable, auditable promotion/rollback history per ``base_model``.
"""

from __future__ import annotations

import sqlite3
import uuid
from contextlib import closing
from datetime import UTC, datetime
from types import TracebackType
from typing import Any, Self

try:
    from training.lineage import AdapterLineageRegistry
except ImportError:  # pragma: no cover - flat pythonpath fallback (pytest.ini)
    from lineage import AdapterLineageRegistry  # type: ignore[no-redef]

_PROMOTION_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS adapter_promotions (
    promotion_id TEXT PRIMARY KEY,
    base_model TEXT NOT NULL,
    adapter_id TEXT NOT NULL,
    previous_adapter_id TEXT,
    action TEXT NOT NULL CHECK (action IN ('promote', 'rollback')),
    reason TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_adapter_promotions_base_model
    ON adapter_promotions (base_model);
"""


class PromotionError(ValueError):
    """Raised when an adapter promotion or rollback gate fails."""


def _utc_now_iso() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _promotion_row_to_dict(row: sqlite3.Row) -> dict[str, Any]:
    return {
        "promotion_id": str(row["promotion_id"]),
        "base_model": str(row["base_model"]),
        "adapter_id": str(row["adapter_id"]),
        "previous_adapter_id": (
            str(row["previous_adapter_id"]) if row["previous_adapter_id"] is not None else None
        ),
        "action": str(row["action"]),
        "reason": str(row["reason"]),
        "created_at": str(row["created_at"]),
    }


class AdapterPromotionManager:
    """Atomic promotion gate and rollback manager backed by SQLite WAL."""

    def __init__(self, registry: AdapterLineageRegistry) -> None:
        self.registry = registry
        self.db_path = registry.db_path
        self._journal_mode = self._initialize()

    def _connect(self, *, isolation_level: str | None = "") -> sqlite3.Connection:
        conn = sqlite3.connect(
            str(self.db_path),
            timeout=30.0,
            isolation_level=isolation_level,
        )
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        return conn

    def _initialize(self) -> str:
        with closing(self._connect()) as conn:
            mode_row = conn.execute("PRAGMA journal_mode=WAL;").fetchone()
            journal_mode = str(mode_row[0]).lower() if mode_row else "unknown"
            conn.executescript(_PROMOTION_SCHEMA_SQL)
            conn.commit()
            return journal_mode

    @property
    def journal_mode(self) -> str:
        """Return the active SQLite journal mode reported on initialization."""
        return self._journal_mode

    def promote(
        self,
        adapter_id: str,
        *,
        require_certified: bool = True,
        reason: str = "",
        candidate_metrics: dict[str, Any] | None = None,
        baseline_metrics: dict[str, Any] | None = None,
        promotion_gate: Any | None = None,
        min_score_delta: float = 0.01,
        max_latency_regression_pct: float = 10.0,
    ) -> dict[str, Any]:
        """Atomically promote a registered adapter to active for its ``base_model``.

        When ``require_certified=True`` (default), the adapter must have
        ``certified is True`` and a non-empty ``chaos_cert_sha256`` digest in
        :class:`AdapterLineageRegistry`.

        When ``candidate_metrics`` and ``baseline_metrics`` are both supplied,
        evaluates ``promotion_gate`` (defaulting to
        :func:`trainweave.adapter_bundle.evaluate_promotion_gate`) and raises
        :class:`PromotionError` if the candidate is not approved.
        """
        if not isinstance(adapter_id, str) or not adapter_id.strip():
            raise ValueError("adapter_id must be a non-empty string")
        if not isinstance(reason, str):
            raise TypeError("reason must be a string")

        adapter = self.registry.get(adapter_id)
        if adapter is None:
            raise PromotionError(
                f"Adapter {adapter_id!r} is not registered in AdapterLineageRegistry"
            )

        if require_certified:
            if not adapter.get("certified"):
                raise PromotionError(
                    f"Adapter {adapter_id!r} is not certified (certified=False); "
                    "cannot promote when require_certified=True"
                )
            chaos_sha = adapter.get("chaos_cert_sha256")
            if not isinstance(chaos_sha, str) or not chaos_sha.strip():
                raise PromotionError(
                    f"Adapter {adapter_id!r} is missing chaos_cert_sha256; "
                    "cannot promote when require_certified=True"
                )

        gate_result: dict[str, Any] | None = None
        if candidate_metrics is not None or baseline_metrics is not None:
            if candidate_metrics is None or baseline_metrics is None:
                raise PromotionError(
                    "Both candidate_metrics and baseline_metrics must be provided "
                    "when evaluating the promotion gate"
                )
            gate_fn = promotion_gate
            if gate_fn is None:
                from trainweave.adapter_bundle import evaluate_promotion_gate

                gate_fn = evaluate_promotion_gate
            gate_result = gate_fn(
                candidate_metrics,
                baseline_metrics,
                min_score_delta=min_score_delta,
                max_latency_regression_pct=max_latency_regression_pct,
            )
            if not gate_result.get("approved"):
                reasons_str = "; ".join(gate_result.get("reasons", [])) or "gate rejected"
                raise PromotionError(
                    f"Adapter {adapter_id!r} failed quantitative promotion gate: {reasons_str}"
                )

        base_model = str(adapter["base_model"])
        promotion_id = f"promo-{uuid.uuid4().hex}"
        created_at = _utc_now_iso()

        with closing(self._connect(isolation_level=None)) as conn:
            conn.execute("BEGIN IMMEDIATE")
            try:
                latest = conn.execute(
                    """
                    SELECT adapter_id
                    FROM adapter_promotions
                    WHERE base_model = ?
                    ORDER BY rowid DESC
                    LIMIT 1
                    """,
                    (base_model,),
                ).fetchone()
                previous_adapter_id = str(latest["adapter_id"]) if latest is not None else None

                if previous_adapter_id == adapter_id:
                    raise PromotionError(
                        f"Adapter {adapter_id!r} is already active for base_model {base_model!r}"
                    )

                conn.execute(
                    """
                    INSERT INTO adapter_promotions (
                        promotion_id,
                        base_model,
                        adapter_id,
                        previous_adapter_id,
                        action,
                        reason,
                        created_at
                    )
                    VALUES (?, ?, ?, ?, 'promote', ?, ?)
                    """,
                    (
                        promotion_id,
                        base_model,
                        adapter_id,
                        previous_adapter_id,
                        reason,
                        created_at,
                    ),
                )
                conn.execute("COMMIT")
            except BaseException:
                conn.execute("ROLLBACK")
                raise

        return {
            "promoted": True,
            "promotion_id": promotion_id,
            "base_model": base_model,
            "active_adapter_id": adapter_id,
            "previous_adapter_id": previous_adapter_id,
            "action": "promote",
            "reason": reason,
            "created_at": created_at,
        }

    def rollback(
        self,
        base_model: str,
        *,
        reason: str = "",
    ) -> dict[str, Any]:
        """Atomically restore ``previous_adapter_id`` as active for ``base_model``.

        Logs a ``"rollback"`` audit record in ``adapter_promotions`` and fails
        cleanly with :class:`PromotionError` if no previous adapter exists.
        """
        if not isinstance(base_model, str) or not base_model.strip():
            raise ValueError("base_model must be a non-empty string")
        if not isinstance(reason, str):
            raise TypeError("reason must be a string")

        promotion_id = f"promo-{uuid.uuid4().hex}"
        created_at = _utc_now_iso()

        with closing(self._connect(isolation_level=None)) as conn:
            conn.execute("BEGIN IMMEDIATE")
            try:
                latest = conn.execute(
                    """
                    SELECT adapter_id, previous_adapter_id
                    FROM adapter_promotions
                    WHERE base_model = ?
                    ORDER BY rowid DESC
                    LIMIT 1
                    """,
                    (base_model,),
                ).fetchone()

                if latest is None:
                    raise PromotionError(
                        f"No active adapter promotion found for base_model {base_model!r}"
                    )

                target_adapter_id = latest["previous_adapter_id"]
                if target_adapter_id is None:
                    raise PromotionError(
                        f"No previous adapter exists to roll back to for base_model {base_model!r}"
                    )

                current_adapter_id = str(latest["adapter_id"])
                target_adapter_id_str = str(target_adapter_id)

                if self.registry.get(target_adapter_id_str) is None:
                    raise PromotionError(
                        f"Previous adapter {target_adapter_id_str!r} is missing from "
                        "AdapterLineageRegistry"
                    )

                conn.execute(
                    """
                    INSERT INTO adapter_promotions (
                        promotion_id,
                        base_model,
                        adapter_id,
                        previous_adapter_id,
                        action,
                        reason,
                        created_at
                    )
                    VALUES (?, ?, ?, ?, 'rollback', ?, ?)
                    """,
                    (
                        promotion_id,
                        base_model,
                        target_adapter_id_str,
                        current_adapter_id,
                        reason,
                        created_at,
                    ),
                )
                conn.execute("COMMIT")
            except BaseException:
                conn.execute("ROLLBACK")
                raise

        return {
            "rolled_back": True,
            "promotion_id": promotion_id,
            "base_model": base_model,
            "active_adapter_id": target_adapter_id_str,
            "previous_adapter_id": current_adapter_id,
            "action": "rollback",
            "reason": reason,
            "created_at": created_at,
        }

    def get_active(self, base_model: str) -> dict[str, Any] | None:
        """Return the currently active adapter promotion record for ``base_model``."""
        if not isinstance(base_model, str) or not base_model.strip():
            raise ValueError("base_model must be a non-empty string")

        with closing(self._connect()) as conn:
            row = conn.execute(
                """
                SELECT *
                FROM adapter_promotions
                WHERE base_model = ?
                ORDER BY rowid DESC
                LIMIT 1
                """,
                (base_model,),
            ).fetchone()

        if row is None:
            return None

        record = _promotion_row_to_dict(row)
        lineage = self.registry.get(record["adapter_id"])
        if lineage is not None:
            record["adapter_name"] = lineage["adapter_name"]
            record["dataset_sha256"] = lineage["dataset_sha256"]
            record["manifest_sha256"] = lineage["manifest_sha256"]
            record["chaos_cert_sha256"] = lineage["chaos_cert_sha256"]
            record["certified"] = lineage["certified"]
        return record

    def list_history(self, base_model: str | None = None) -> list[dict[str, Any]]:
        """Return promotion and rollback audit records, newest first."""
        params: list[Any] = []
        where_sql = ""
        if base_model is not None:
            where_sql = "WHERE base_model = ?"
            params.append(base_model)

        query = f"SELECT * FROM adapter_promotions {where_sql} ORDER BY rowid DESC"
        with closing(self._connect()) as conn:
            rows = conn.execute(query, params).fetchall()
        return [_promotion_row_to_dict(row) for row in rows]

    def close(self) -> None:
        """No-op hook for API parity with persistent-connection managers."""

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()
