"""Unit tests for TrainWeave AdapterPromotionManager and flywheel CLI promote/rollback."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from training.flywheel_cli import main as flywheel_main
from training.lineage import AdapterLineageRegistry
from training.promotion import AdapterPromotionManager, PromotionError

SHA_A = "a" * 64
SHA_B = "b" * 64
SHA_C = "c" * 64
CHAOS_SHA_1 = "1" * 64
CHAOS_SHA_2 = "2" * 64
BASE_MODEL = "Qwen/Qwen2.5-3B-Instruct"


def _register_adapter(
    registry: AdapterLineageRegistry,
    *,
    adapter_name: str,
    base_model: str = BASE_MODEL,
    dataset_sha256: str = SHA_A,
    manifest_sha256: str = SHA_B,
    chaos_cert_sha256: str | None = CHAOS_SHA_1,
    certified: bool = True,
) -> dict:
    return registry.record(
        adapter_id=f"adp-{adapter_name}",
        adapter_name=adapter_name,
        base_model=base_model,
        dataset_sha256=dataset_sha256,
        manifest_sha256=manifest_sha256,
        chaos_cert_sha256=chaos_cert_sha256,
        certified=certified,
    )


def test_promote_rejects_uncertified_and_missing_chaos_sha(tmp_path: Path) -> None:
    db_path = tmp_path / "lineage.sqlite3"
    with AdapterLineageRegistry(db_path) as registry:
        uncert = _register_adapter(
            registry,
            adapter_name="qwen-uncert",
            chaos_cert_sha256=CHAOS_SHA_1,
            certified=False,
        )
        missing_sha = _register_adapter(
            registry,
            adapter_name="qwen-no-sha",
            chaos_cert_sha256=None,
            certified=True,
        )

        manager = AdapterPromotionManager(registry)
        assert manager.journal_mode == "wal"

        with pytest.raises(PromotionError, match="not registered"):
            manager.promote("missing-adapter-id")

        with pytest.raises(PromotionError, match="not certified"):
            manager.promote(uncert["adapter_id"], require_certified=True)

        with pytest.raises(PromotionError, match="missing chaos_cert_sha256"):
            manager.promote(missing_sha["adapter_id"], require_certified=True)

        assert manager.get_active(BASE_MODEL) is None
        assert manager.list_history(BASE_MODEL) == []

        # Explicitly allowing uncertified succeeds
        res = manager.promote(
            uncert["adapter_id"],
            require_certified=False,
            reason="dev-override",
        )
        assert res["promoted"] is True
        assert res["active_adapter_id"] == uncert["adapter_id"]
        assert res["previous_adapter_id"] is None


def test_promote_v1_to_v2_rollback_and_persistence(tmp_path: Path) -> None:
    db_path = tmp_path / "lineage.sqlite3"
    with AdapterLineageRegistry(db_path) as registry:
        v1 = _register_adapter(
            registry,
            adapter_name="qwen-v1",
            chaos_cert_sha256=CHAOS_SHA_1,
            certified=True,
        )
        v2 = _register_adapter(
            registry,
            adapter_name="qwen-v2",
            dataset_sha256=SHA_B,
            manifest_sha256=SHA_C,
            chaos_cert_sha256=CHAOS_SHA_2,
            certified=True,
        )

        manager = AdapterPromotionManager(registry)

        # Rollback before any promotion fails cleanly
        with pytest.raises(PromotionError, match="No active adapter"):
            manager.rollback(BASE_MODEL)

        # Promote v1
        p1 = manager.promote(v1["adapter_id"], reason="initial release")
        assert p1["promoted"] is True
        assert p1["base_model"] == BASE_MODEL
        assert p1["active_adapter_id"] == v1["adapter_id"]
        assert p1["previous_adapter_id"] is None

        # Cannot rollback when there is no previous adapter
        with pytest.raises(PromotionError, match="No previous adapter"):
            manager.rollback(BASE_MODEL)

        # Re-promoting already-active v1 is rejected
        with pytest.raises(PromotionError, match="already active"):
            manager.promote(v1["adapter_id"])

        # Promote v2 over v1
        p2 = manager.promote(v2["adapter_id"], reason="cycle-4 upgrade")
        assert p2["promoted"] is True
        assert p2["active_adapter_id"] == v2["adapter_id"]
        assert p2["previous_adapter_id"] == v1["adapter_id"]

        active = manager.get_active(BASE_MODEL)
        assert active is not None
        assert active["adapter_id"] == v2["adapter_id"]
        assert active["adapter_name"] == "qwen-v2"
        assert active["certified"] is True

        # Rollback restores v1 atomically
        rb = manager.rollback(BASE_MODEL, reason="regression detected")
        assert rb["rolled_back"] is True
        assert rb["base_model"] == BASE_MODEL
        assert rb["active_adapter_id"] == v1["adapter_id"]
        assert rb["previous_adapter_id"] == v2["adapter_id"]
        assert rb["action"] == "rollback"

    # Reopen SQLite DB to verify WAL persistence
    with AdapterLineageRegistry(db_path) as reopened_registry:
        reopened_mgr = AdapterPromotionManager(reopened_registry)
        active_after = reopened_mgr.get_active(BASE_MODEL)
        assert active_after is not None
        assert active_after["adapter_id"] == v1["adapter_id"]
        assert active_after["action"] == "rollback"

        history = reopened_mgr.list_history(BASE_MODEL)
        assert len(history) == 3
        assert [h["action"] for h in history] == ["rollback", "promote", "promote"]
        assert [h["adapter_id"] for h in history] == [
            v1["adapter_id"],
            v2["adapter_id"],
            v1["adapter_id"],
        ]


def test_flywheel_cli_promote_and_rollback_json(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    db_path = tmp_path / "cli_lineage.sqlite3"
    with AdapterLineageRegistry(db_path) as registry:
        uncert = _register_adapter(
            registry,
            adapter_name="cli-uncert",
            chaos_cert_sha256=None,
            certified=False,
        )
        v1 = _register_adapter(
            registry,
            adapter_name="cli-v1",
            chaos_cert_sha256=CHAOS_SHA_1,
            certified=True,
        )
        v2 = _register_adapter(
            registry,
            adapter_name="cli-v2",
            chaos_cert_sha256=CHAOS_SHA_2,
            certified=True,
        )

    # Uncertified promote returns exit code 2
    rc = flywheel_main(
        ["promote", "--db", str(db_path), "--adapter-id", uncert["adapter_id"], "--json"]
    )
    assert rc == 2
    err_out = json.loads(capsys.readouterr().out)
    assert err_out["promoted"] is False
    assert err_out["status"] == "rejected"

    # Promote v1 returns exit code 0
    rc = flywheel_main(
        [
            "promote",
            "--db",
            str(db_path),
            "--adapter-id",
            v1["adapter_id"],
            "--reason",
            "v1 rollout",
            "--json",
        ]
    )
    assert rc == 0
    out_v1 = json.loads(capsys.readouterr().out)
    assert out_v1["promoted"] is True
    assert out_v1["active_adapter_id"] == v1["adapter_id"]

    # Promote v2 returns exit code 0
    rc = flywheel_main(
        [
            "promote",
            "--db",
            str(db_path),
            "--adapter-id",
            v2["adapter_id"],
            "--reason",
            "v2 rollout",
            "--json",
        ]
    )
    assert rc == 0
    out_v2 = json.loads(capsys.readouterr().out)
    assert out_v2["active_adapter_id"] == v2["adapter_id"]
    assert out_v2["previous_adapter_id"] == v1["adapter_id"]

    # Rollback returns exit code 0 and restores v1
    rc = flywheel_main(
        [
            "rollback",
            "--db",
            str(db_path),
            "--base-model",
            BASE_MODEL,
            "--reason",
            "slo breach",
            "--json",
        ]
    )
    assert rc == 0
    out_rb = json.loads(capsys.readouterr().out)
    assert out_rb["rolled_back"] is True
    assert out_rb["active_adapter_id"] == v1["adapter_id"]
    assert out_rb["previous_adapter_id"] == v2["adapter_id"]
