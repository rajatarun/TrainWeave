"""Tests for TrainWeave portable LoRA adapter bundle (node-snapshot-v1) and promotion gate."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator

from training.lineage import AdapterLineageRegistry
from training.promotion import AdapterPromotionManager, PromotionError
from trainweave.adapter_bundle import (
    compute_components_checksum,
    evaluate_promotion_gate,
    export_adapter_bundle,
    verify_adapter_bundle,
)

NODE_SNAPSHOT_SCHEMA_PATH = (
    Path(__file__).resolve().parents[2] / "weave" / "contracts" / "node-snapshot-v1.schema.json"
)


def _sample_artifact_manifest() -> dict:
    return {
        "schema": "artifact-manifest-v1",
        "adapter_id": "lora-qwen25-home-v5",
        "base_model": "Qwen/Qwen2.5-7B-Instruct",
        "created_at": "2026-11-24T12:00:00Z",
        "format": "gguf",
        "sha256": "a" * 64,
    }


def _make_adapter_dir(tmp_path: Path) -> Path:
    adapter_dir = tmp_path / "trained_adapter"
    adapter_dir.mkdir()
    (adapter_dir / "adapter_model.safetensors").write_bytes(b"FAKE-LORA-WEIGHTS-V5" * 64)
    (adapter_dir / "adapter_config.json").write_text(
        json.dumps({"r": 16, "lora_alpha": 32, "base_model": "Qwen/Qwen2.5-7B-Instruct"}),
        encoding="utf-8",
    )
    return adapter_dir


def test_export_and_verify_adapter_bundle_conforms_to_schema(tmp_path: Path) -> None:
    adapter_dir = _make_adapter_dir(tmp_path)
    out_dir = tmp_path / "bundle_out"

    snapshot = export_adapter_bundle(
        adapter_dir,
        _sample_artifact_manifest(),
        out_dir,
        node_id="pi5-living-room",
    )

    assert snapshot["schema"] == "node-snapshot-v1"
    assert snapshot["node_id"] == "pi5-living-room"
    assert len(snapshot["components"]) == 3
    kinds = {c["kind"] for c in snapshot["components"]}
    assert "lora_adapter" in kinds
    assert snapshot["checksum_sha256"] == compute_components_checksum(snapshot["components"])

    if NODE_SNAPSHOT_SCHEMA_PATH.is_file():
        schema = json.loads(NODE_SNAPSHOT_SCHEMA_PATH.read_text(encoding="utf-8"))
        Draft202012Validator(schema).validate(snapshot)

    manifest_path = out_dir / "snapshot-manifest.json"
    assert manifest_path.is_file()
    verification = verify_adapter_bundle(manifest_path)
    assert verification["valid"] is True
    assert verification["errors"] == []
    assert verification["component_count"] == 3


def test_verify_adapter_bundle_rejects_tampered_weights(tmp_path: Path) -> None:
    adapter_dir = _make_adapter_dir(tmp_path)
    out_dir = tmp_path / "bundle_tamper"
    export_adapter_bundle(
        adapter_dir,
        _sample_artifact_manifest(),
        out_dir,
        node_id="pi5-office",
    )

    weights_path = out_dir / "adapter" / "adapter_model.safetensors"
    weights_path.write_bytes(b"TAMPERED-WEIGHTS-DATA")

    verification = verify_adapter_bundle(out_dir / "snapshot-manifest.json")
    assert verification["valid"] is False
    assert any("SHA-256 mismatch" in err or "size mismatch" in err for err in verification["errors"])


def test_verify_adapter_bundle_rejects_tampered_checksum(tmp_path: Path) -> None:
    adapter_dir = _make_adapter_dir(tmp_path)
    out_dir = tmp_path / "bundle_bad_checksum"
    export_adapter_bundle(
        adapter_dir,
        _sample_artifact_manifest(),
        out_dir,
        node_id="pi5-office",
    )

    manifest_path = out_dir / "snapshot-manifest.json"
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    data["checksum_sha256"] = "0" * 64
    manifest_path.write_text(json.dumps(data), encoding="utf-8")

    verification = verify_adapter_bundle(manifest_path)
    assert verification["valid"] is False
    assert any("checksum_sha256 mismatch" in err for err in verification["errors"])


def test_evaluate_promotion_gate_pass_and_fail_cases() -> None:
    approved = evaluate_promotion_gate(
        {"score": 0.89, "latency_ms": 105.0},
        {"score": 0.85, "latency_ms": 100.0},
    )
    assert approved["approved"] is True
    assert approved["score_delta"] == pytest.approx(0.04)
    assert approved["latency_delta_pct"] == pytest.approx(5.0)
    assert approved["reasons"] == []

    score_regression = evaluate_promotion_gate(
        {"score": 0.84, "latency_ms": 95.0},
        {"score": 0.85, "latency_ms": 100.0},
        min_score_delta=0.01,
    )
    assert score_regression["approved"] is False
    assert any("Score delta" in r for r in score_regression["reasons"])

    latency_regression = evaluate_promotion_gate(
        {"score": 0.90, "latency_ms": 125.0},
        {"score": 0.85, "latency_ms": 100.0},
        max_latency_regression_pct=10.0,
    )
    assert latency_regression["approved"] is False
    assert any("Latency regression" in r for r in latency_regression["reasons"])


def test_adapter_promotion_manager_with_optional_promotion_gate(tmp_path: Path) -> None:
    db_path = tmp_path / "lineage.sqlite3"
    registry = AdapterLineageRegistry(db_path)
    registry.record(
        adapter_id="adapter-v2",
        adapter_name="qwen-v2",
        base_model="qwen2.5-7b",
        dataset_sha256="a" * 64,
        manifest_sha256="b" * 64,
        certified=True,
        chaos_cert_sha256="c" * 64,
    )
    manager = AdapterPromotionManager(registry)

    # Blocked when quantitative gate fails
    with pytest.raises(PromotionError, match="quantitative promotion gate"):
        manager.promote(
            "adapter-v2",
            candidate_metrics={"score": 0.70, "latency_ms": 150.0},
            baseline_metrics={"score": 0.80, "latency_ms": 100.0},
        )
    assert manager.get_active("qwen2.5-7b") is None

    # Succeeds when quantitative gate passes
    res = manager.promote(
        "adapter-v2",
        candidate_metrics={"score": 0.86, "latency_ms": 104.0},
        baseline_metrics={"score": 0.80, "latency_ms": 100.0},
    )
    assert res["promoted"] is True
    assert manager.get_active("qwen2.5-7b")["adapter_id"] == "adapter-v2"
