"""Unit and CLI tests for trainweave-flywheel (src/training/flywheel_cli.py)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import flywheel_cli
import pytest


def _write_jsonl(path: Path, rows: list[dict[str, str]]) -> Path:
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    return path


def _make_chaos_cert(
    *,
    subject_id: str = "edge-v1",
    certified: bool = True,
    schema: str = "chaos-cert-v1",
    observed_resilience_score: float = 0.92,
    tamper_digest: bool = False,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "schema": schema,
        "subject_id": subject_id,
        "certified": certified,
        "min_resilience_score": 0.75,
        "observed_resilience_score": observed_resilience_score,
        "abort_count": 0,
        "max_aborts": 0,
        "baseline_pass_rate": 1.0,
        "chaos_pass_rate": 0.92,
        "reasons": [] if certified else ["resilience score below threshold"],
        "generated_at": "2026-11-23T12:00:00Z",
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    if tamper_digest:
        digest = ("0" if digest[0] != "0" else "1") + digest[1:]
    payload["content_sha256"] = digest
    return payload


def _clean_rows() -> list[dict[str, str]]:
    return [
        {
            "prompt": "Turn off the bedroom lamp at 11pm.",
            "chosen": "Scheduled bedroom lamp off at 23:00 via DeviceWeave.",
            "rejected": "I cannot control lights.",
        },
        {
            "prompt": "Summarize today's agent run telemetry.",
            "chosen": "3 runs succeeded with 0 policy blocks and average latency 180ms.",
            "rejected": "No data.",
        },
    ]


def test_flywheel_cli_full_clean_ed25519_and_chaos_cert(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    gw_integrity = pytest.importorskip("gatewayweave.integrity")
    pytest.importorskip("cryptography")

    dataset_path = _write_jsonl(tmp_path / "prefs.jsonl", _clean_rows())
    adapter_path = tmp_path / "adapter_model.safetensors"
    adapter_path.write_bytes(b"synthetic-lora-weights-v2")

    gguf_path = tmp_path / "home-agent-q4.gguf"
    gguf_bytes = b"GGUF\x03\x00\x00\x00" + b"\x42" * 512
    gguf_path.write_bytes(gguf_bytes)
    gguf_sha = hashlib.sha256(gguf_bytes).hexdigest()
    expected_version_id = f"gguf:home-agent-v2@sha256:{gguf_sha}"

    priv_pem, pub_pem = gw_integrity.generate_keypair()
    pub_key_path = tmp_path / "ed25519_pub.pem"
    pub_key_path.write_text(pub_pem, encoding="utf-8")

    passport_record = gw_integrity.apply_ed25519_integrity(
        {
            "record_version": 1,
            "verdict": "PASS",
            "model_version_id": expected_version_id,
            "expires_at": "2099-01-01T00:00:00Z",
        },
        priv_pem,
    )
    passport_path = tmp_path / "passport.json"
    passport_path.write_text(json.dumps(passport_record), encoding="utf-8")

    chaos_cert_path = tmp_path / "chaos_cert.json"
    chaos_cert_path.write_text(
        json.dumps(_make_chaos_cert(subject_id=expected_version_id, certified=True)),
        encoding="utf-8",
    )

    out_dir = tmp_path / "bundle_out"
    rc = flywheel_cli.main(
        [
            "package",
            "--dataset",
            str(dataset_path),
            "--adapter",
            str(adapter_path),
            "--base-model",
            "Qwen/Qwen2.5-3B-Instruct",
            "--adapter-name",
            "home-agent-v2",
            "--out-dir",
            str(out_dir),
            "--gguf",
            str(gguf_path),
            "--passport",
            str(passport_path),
            "--public-key",
            str(pub_key_path),
            "--chaos-cert",
            str(chaos_cert_path),
            "--json",
        ]
    )

    assert rc == 0
    stdout_payload = json.loads(capsys.readouterr().out)
    assert stdout_payload["status"] == "ready"
    assert stdout_payload["model_version_id"] == expected_version_id
    assert stdout_payload["dataset"]["passed"] is True
    assert stdout_payload["dataset"]["valid_rows"] == 2
    assert stdout_payload["passport"]["status"] == "verified"
    assert stdout_payload["passport"]["signature_checked"] is True
    assert stdout_payload["chaos_cert"]["status"] == "verified"
    assert stdout_payload["chaos_cert"]["certified"] is True

    manifest_file = out_dir / "artifact_manifest.json"
    bundle_file = out_dir / "flywheel_bundle.json"
    assert manifest_file.is_file()
    assert bundle_file.is_file()

    manifest_data = json.loads(manifest_file.read_text(encoding="utf-8"))
    assert manifest_data["model_version_id"] == expected_version_id
    assert manifest_data["passport"]["verified"] is True

    bundle_data = json.loads(bundle_file.read_text(encoding="utf-8"))
    assert bundle_data["schema"] == "flywheel-bundle-v1"
    assert bundle_data["manifest"]["sha256"] == hashlib.sha256(manifest_file.read_bytes()).hexdigest()


def test_flywheel_package_lora_without_optional_gates(tmp_path: Path) -> None:
    dataset_path = _write_jsonl(tmp_path / "prefs.jsonl", _clean_rows())
    adapter_path = tmp_path / "adapter.bin"
    adapter_path.write_bytes(b"lora-only-weights")
    out_dir = tmp_path / "lora_out"

    bundle = flywheel_cli.run_flywheel_package(
        dataset_path=dataset_path,
        adapter_path=adapter_path,
        base_model="Qwen/Qwen2.5-3B-Instruct",
        adapter_name="lora-only",
        output_dir=out_dir,
    )

    assert bundle["artifact_format"] == "lora"
    assert bundle["model_version_id"].startswith("lora:lora-only@sha256:")
    assert bundle["passport"]["status"] == "skipped"
    assert bundle["chaos_cert"]["status"] == "skipped"
    assert (out_dir / "artifact_manifest.json").is_file()
    assert (out_dir / "flywheel_bundle.json").is_file()


def test_flywheel_cli_dirty_duplicate_dataset_fails_preflight(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    dup_row = _clean_rows()[0]
    dataset_path = _write_jsonl(tmp_path / "dirty.jsonl", [dup_row, dup_row])
    adapter_path = tmp_path / "adapter.bin"
    adapter_path.write_bytes(b"weights")
    out_dir = tmp_path / "should_not_exist"

    rc = flywheel_cli.main(
        [
            "package",
            "--dataset",
            str(dataset_path),
            "--adapter",
            str(adapter_path),
            "--base-model",
            "Qwen/Qwen2.5-3B-Instruct",
            "--adapter-name",
            "dup-test",
            "--out-dir",
            str(out_dir),
            "--json",
        ]
    )

    assert rc != 0
    err_payload = json.loads(capsys.readouterr().out)
    assert err_payload["status"] == "failed"
    assert err_payload["stage"] == "preflight"
    assert err_payload["report"]["duplicate_rows"] == 1
    assert not out_dir.exists()


def test_flywheel_rejects_uncertified_chaos_cert(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    dataset_path = _write_jsonl(tmp_path / "prefs.jsonl", _clean_rows())
    adapter_path = tmp_path / "adapter.bin"
    adapter_path.write_bytes(b"weights")

    uncertified_path = tmp_path / "uncertified.json"
    uncertified_path.write_text(
        json.dumps(_make_chaos_cert(certified=False, observed_resilience_score=0.45)),
        encoding="utf-8",
    )

    out_dir = tmp_path / "out_uncert"
    rc = flywheel_cli.main(
        [
            "package",
            "--dataset",
            str(dataset_path),
            "--adapter",
            str(adapter_path),
            "--base-model",
            "Qwen/Qwen2.5-3B-Instruct",
            "--adapter-name",
            "uncert-test",
            "--out-dir",
            str(out_dir),
            "--chaos-cert",
            str(uncertified_path),
            "--json",
        ]
    )

    assert rc == 2
    err_payload = json.loads(capsys.readouterr().out)
    assert err_payload["stage"] == "chaos_cert"
    assert "certified is False" in err_payload["error"]
    assert not (out_dir / "flywheel_bundle.json").exists()


def test_flywheel_rejects_tampered_chaos_cert_digest(tmp_path: Path) -> None:
    dataset_path = _write_jsonl(tmp_path / "prefs.jsonl", _clean_rows())
    adapter_path = tmp_path / "adapter.bin"
    adapter_path.write_bytes(b"weights")

    tampered_path = tmp_path / "tampered_cert.json"
    tampered_path.write_text(
        json.dumps(_make_chaos_cert(certified=True, tamper_digest=True)),
        encoding="utf-8",
    )

    with pytest.raises(flywheel_cli.FlywheelError, match="content_sha256 mismatch") as exc_info:
        flywheel_cli.run_flywheel_package(
            dataset_path=dataset_path,
            adapter_path=adapter_path,
            base_model="Qwen/Qwen2.5-3B-Instruct",
            adapter_name="tamper-test",
            output_dir=tmp_path / "out_tamper",
            chaos_cert_path=tampered_path,
        )
    assert exc_info.value.stage == "chaos_cert"


def test_flywheel_rejects_wrong_ed25519_public_key(tmp_path: Path) -> None:
    gw_integrity = pytest.importorskip("gatewayweave.integrity")
    pytest.importorskip("cryptography")

    dataset_path = _write_jsonl(tmp_path / "prefs.jsonl", _clean_rows())
    adapter_path = tmp_path / "adapter.bin"
    adapter_bytes = b"weights-ed25519"
    adapter_path.write_bytes(adapter_bytes)
    expected_vid = f"lora:key-test@sha256:{hashlib.sha256(adapter_bytes).hexdigest()}"

    priv_pem, _ = gw_integrity.generate_keypair()
    _, wrong_pub_pem = gw_integrity.generate_keypair()

    wrong_pub_path = tmp_path / "wrong_pub.pem"
    wrong_pub_path.write_text(wrong_pub_pem, encoding="utf-8")

    passport_record = gw_integrity.apply_ed25519_integrity(
        {
            "record_version": 1,
            "verdict": "PASS",
            "model_version_id": expected_vid,
            "expires_at": "2099-01-01T00:00:00Z",
        },
        priv_pem,
    )
    passport_path = tmp_path / "passport.json"
    passport_path.write_text(json.dumps(passport_record), encoding="utf-8")

    with pytest.raises(flywheel_cli.FlywheelError, match="passport verification failed") as exc_info:
        flywheel_cli.run_flywheel_package(
            dataset_path=dataset_path,
            adapter_path=adapter_path,
            base_model="Qwen/Qwen2.5-3B-Instruct",
            adapter_name="key-test",
            output_dir=tmp_path / "out_bad_key",
            passport_path=passport_path,
            public_key_path=wrong_pub_path,
        )
    assert exc_info.value.stage == "passport"
