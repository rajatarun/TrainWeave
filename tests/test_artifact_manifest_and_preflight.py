"""Tests for artifact_manifest.py and preflight_validator.py."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import artifact_manifest as am
import preflight_validator as pv
import pytest


def _write_jsonl(path: Path, rows: list[object]) -> Path:
    lines = [row if isinstance(row, str) else json.dumps(row) for row in rows]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


# ── Artifact Manifest tests ───────────────────────────────────────────────────


def test_build_artifact_manifest_lora_file_sha256_and_roundtrip(tmp_path: Path) -> None:
    adapter_file = tmp_path / "adapter_model.safetensors"
    payload = b"synthetic-lora-weights-" * 4096
    adapter_file.write_bytes(payload)
    expected_sha = hashlib.sha256(payload).hexdigest()

    manifest = am.build_artifact_manifest(
        adapter_path=adapter_file,
        base_model="Qwen/Qwen2.5-3B-Instruct",
        adapter_name="nightshift-home-v1",
        created_at="2026-11-23T00:00:00Z",
        metadata={"job_id": "job-42"},
    )

    assert manifest["schema_version"] == am.MANIFEST_SCHEMA_VERSION
    assert manifest["artifact_format"] == "lora"
    assert manifest["adapter"]["sha256"] == expected_sha
    assert manifest["adapter"]["size_bytes"] == len(payload)
    assert manifest["gguf"] is None
    assert manifest["model_version_id"] == f"lora:nightshift-home-v1@sha256:{expected_sha}"

    out_path = tmp_path / "artifact_manifest.json"
    am.write_artifact_manifest(manifest, out_path)
    loaded = am.load_artifact_manifest(out_path)
    assert loaded == manifest


def test_build_artifact_manifest_gguf_format_and_directory_adapter(tmp_path: Path) -> None:
    adapter_dir = tmp_path / "lora_adapter"
    adapter_dir.mkdir()
    (adapter_dir / "adapter_config.json").write_text('{"r": 16}', encoding="utf-8")
    (adapter_dir / "adapter_model.bin").write_bytes(b"weights-v1")

    gguf_file = tmp_path / "home-assistant-q4.gguf"
    gguf_bytes = b"GGUF\x03\x00\x00\x00" + b"\xab" * 1024
    gguf_file.write_bytes(gguf_bytes)
    expected_gguf_sha = hashlib.sha256(gguf_bytes).hexdigest()

    manifest = am.build_artifact_manifest(
        adapter_path=adapter_dir,
        base_model="Qwen/Qwen2.5-3B-Instruct",
        adapter_name="pi5-assistant",
        gguf_path=gguf_file,
    )

    assert manifest["artifact_format"] == "gguf"
    assert manifest["gguf"]["sha256"] == expected_gguf_sha
    assert manifest["gguf"]["size_bytes"] == len(gguf_bytes)
    assert manifest["model_version_id"] == f"gguf:pi5-assistant@sha256:{expected_gguf_sha}"


def test_build_artifact_manifest_with_gatewayweave_passport_hook(tmp_path: Path) -> None:
    adapter_file = tmp_path / "adapter.safetensors"
    adapter_file.write_bytes(b"adapter-data")
    gguf_file = tmp_path / "model.gguf"
    gguf_bytes = b"gguf-binary-content"
    gguf_file.write_bytes(gguf_bytes)
    gguf_sha = hashlib.sha256(gguf_bytes).hexdigest()
    expected_version = f"gguf:edge-v1@sha256:{gguf_sha}"

    passport_path = tmp_path / "passport.json"
    passport_path.write_text(
        json.dumps(
            {
                "record_version": 1,
                "verdict": "PASS",
                "model_version_id": expected_version,
                "expires_at": "2099-01-01T00:00:00Z",
            }
        ),
        encoding="utf-8",
    )

    manifest = am.build_artifact_manifest(
        adapter_path=adapter_file,
        base_model="Qwen/Qwen2.5-3B-Instruct",
        adapter_name="edge-v1",
        gguf_path=gguf_file,
        passport_path=passport_path,
    )

    assert manifest["passport"]["verified"] is True
    assert manifest["passport"]["errors"] == []
    assert manifest["passport"]["model_version_id"] == expected_version


def test_passport_hook_flags_mismatch_and_expired(tmp_path: Path) -> None:
    adapter_file = tmp_path / "adapter.bin"
    adapter_file.write_bytes(b"abc")

    bad_passport = tmp_path / "bad_passport.json"
    bad_passport.write_text(
        json.dumps(
            {
                "verdict": "FAIL",
                "model_version_id": "lora:wrong@sha256:" + "0" * 64,
                "expires_at": "2020-01-01T00:00:00Z",
            }
        ),
        encoding="utf-8",
    )

    manifest = am.build_artifact_manifest(
        adapter_path=adapter_file,
        base_model="Qwen/Qwen2.5-3B-Instruct",
        adapter_name="right-name",
        passport_path=bad_passport,
    )

    assert manifest["passport"]["verified"] is False
    assert len(manifest["passport"]["errors"]) == 3


def test_validate_artifact_manifest_rejects_tampered_version_id(tmp_path: Path) -> None:
    adapter_file = tmp_path / "adapter.bin"
    adapter_file.write_bytes(b"hello")
    manifest = am.build_artifact_manifest(
        adapter_path=adapter_file,
        base_model="Qwen/Qwen2.5-3B-Instruct",
        adapter_name="demo",
    )
    manifest["model_version_id"] = "lora:demo@sha256:" + "f" * 64
    with pytest.raises(ValueError, match="does not match expected"):
        am.validate_artifact_manifest(manifest)


def test_build_artifact_manifest_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        am.build_artifact_manifest(
            adapter_path=tmp_path / "does-not-exist.bin",
            base_model="Qwen/Qwen2.5-3B-Instruct",
            adapter_name="demo",
        )


def test_verify_gatewayweave_passport_hmac_valid_tampered_and_wrong_key(tmp_path: Path) -> None:
    gw_integrity = pytest.importorskip("gatewayweave.integrity")

    model_version_id = "lora:hmac-test@sha256:" + "a" * 64
    base_record = {
        "record_version": 1,
        "verdict": "PASS",
        "model_version_id": model_version_id,
        "expires_at": "2099-01-01T00:00:00Z",
    }
    secret = "super-secret-hmac-key-2026"
    signed = gw_integrity.apply_integrity(base_record, secret)

    valid_path = tmp_path / "hmac_valid.json"
    valid_path.write_text(json.dumps(signed), encoding="utf-8")

    ok_res = am.verify_gatewayweave_passport(
        valid_path,
        expected_model_version_id=model_version_id,
        integrity_key=secret,
    )
    assert ok_res["gatewayweave_installed"] is True
    assert ok_res["signature_checked"] is True
    assert ok_res["verified"] is True
    assert ok_res["errors"] == []

    # Wrong key must fail verification
    wrong_key_res = am.verify_gatewayweave_passport(
        valid_path,
        expected_model_version_id=model_version_id,
        integrity_key="wrong-hmac-key",
    )
    assert wrong_key_res["signature_checked"] is True
    assert wrong_key_res["verified"] is False
    assert any("does not match" in err for err in wrong_key_res["errors"])

    # Tampered payload must fail verification
    tampered = dict(signed)
    tampered["extra_tampered"] = True
    tampered_path = tmp_path / "hmac_tampered.json"
    tampered_path.write_text(json.dumps(tampered), encoding="utf-8")
    tampered_res = am.verify_gatewayweave_passport(
        tampered_path,
        expected_model_version_id=model_version_id,
        integrity_key=secret,
    )
    assert tampered_res["signature_checked"] is True
    assert tampered_res["verified"] is False
    assert any("does not match" in err for err in tampered_res["errors"])

    # Stripped integrity block when key is provided must fail verification
    unsigned_path = tmp_path / "hmac_unsigned.json"
    unsigned_path.write_text(json.dumps(base_record), encoding="utf-8")
    unsigned_res = am.verify_gatewayweave_passport(
        unsigned_path,
        expected_model_version_id=model_version_id,
        integrity_key=secret,
    )
    assert unsigned_res["signature_checked"] is True
    assert unsigned_res["verified"] is False
    assert any("missing" in err for err in unsigned_res["errors"])


def test_verify_gatewayweave_passport_ed25519_valid_tampered_and_wrong_key(
    tmp_path: Path,
) -> None:
    gw_integrity = pytest.importorskip("gatewayweave.integrity")
    pytest.importorskip("cryptography")

    priv_pem, pub_pem = gw_integrity.generate_keypair()
    _, other_pub_pem = gw_integrity.generate_keypair()

    adapter_file = tmp_path / "adapter.safetensors"
    adapter_bytes = b"ed25519-adapter-weights"
    adapter_file.write_bytes(adapter_bytes)
    adapter_sha = hashlib.sha256(adapter_bytes).hexdigest()
    model_version_id = f"lora:ed25519-test@sha256:{adapter_sha}"

    base_record = {
        "record_version": 1,
        "verdict": "PASS",
        "model_version_id": model_version_id,
        "expires_at": "2099-01-01T00:00:00Z",
    }
    signed = gw_integrity.apply_ed25519_integrity(base_record, priv_pem)
    valid_path = tmp_path / "ed25519_valid.json"
    valid_path.write_text(json.dumps(signed), encoding="utf-8")

    # End-to-end through build_artifact_manifest
    manifest = am.build_artifact_manifest(
        adapter_path=adapter_file,
        base_model="Qwen/Qwen2.5-3B-Instruct",
        adapter_name="ed25519-test",
        passport_path=valid_path,
        public_key_pem=pub_pem,
    )
    assert manifest["passport"]["gatewayweave_installed"] is True
    assert manifest["passport"]["signature_checked"] is True
    assert manifest["passport"]["verified"] is True
    assert manifest["passport"]["errors"] == []

    # Wrong Ed25519 public key must fail verification
    wrong_key_res = am.verify_gatewayweave_passport(
        valid_path,
        expected_model_version_id=model_version_id,
        public_key_pem=other_pub_pem,
    )
    assert wrong_key_res["signature_checked"] is True
    assert wrong_key_res["verified"] is False
    assert any("does not match" in err for err in wrong_key_res["errors"])

    # Tampered payload must fail Ed25519 verification
    tampered = dict(signed)
    tampered["issuer"] = "attacker"
    tampered_path = tmp_path / "ed25519_tampered.json"
    tampered_path.write_text(json.dumps(tampered), encoding="utf-8")
    tampered_res = am.verify_gatewayweave_passport(
        tampered_path,
        expected_model_version_id=model_version_id,
        public_key_pem=pub_pem,
    )
    assert tampered_res["signature_checked"] is True
    assert tampered_res["verified"] is False
    assert any("does not match" in err for err in tampered_res["errors"])


# ── Night-Shift Pre-Flight Validator tests ────────────────────────────────────


def test_preflight_passes_clean_jsonl_dataset(tmp_path: Path) -> None:
    dataset = _write_jsonl(
        tmp_path / "clean.jsonl",
        [
            {
                "prompt": "Summarize the benefits of SQLite WAL mode.",
                "chosen": "WAL mode allows concurrent readers while writing.",
                "rejected": "SQLite does not support concurrency.",
            },
            {
                "prompt": "How do you compute a SHA-256 digest in Python?",
                "chosen": "Use hashlib.sha256() and stream file chunks.",
                "rejected": "Use random.randint().",
            },
        ],
    )

    report = pv.validate_jsonl_dataset(dataset)
    assert report.passed is True
    assert report.total_lines == 2
    assert report.valid_rows == 2
    assert report.invalid_rows == 0
    assert report.duplicate_rows == 0
    assert report.issues == []
    as_dict = report.to_dict()
    assert as_dict["passed"] is True
    assert as_dict["valid_rows"] == 2


def test_preflight_catches_malformed_empty_missing_and_duplicate_rows(tmp_path: Path) -> None:
    good_row = {
        "prompt": "What is a LoRA adapter?",
        "chosen": "A low-rank weight update matrix for efficient fine-tuning.",
        "rejected": "A network cable.",
    }
    dataset = _write_jsonl(
        tmp_path / "dirty.jsonl",
        [
            good_row,
            good_row,  # duplicate
            "{not valid json",  # malformed JSON
            "",  # empty line
            {"prompt": "Only prompt", "chosen": "Missing rejected"},  # missing field
            {"prompt": "   ", "chosen": "ok", "rejected": "bad"},  # empty field
            {"prompt": "Same", "chosen": "identical", "rejected": "identical"},  # identical
        ],
    )

    report = pv.validate_jsonl_dataset(dataset, pv.PreflightConfig(scan_pii=False))
    assert report.passed is False
    assert report.total_lines == 7
    assert report.valid_rows == 1
    assert report.invalid_rows == 6
    assert report.duplicate_rows == 1

    codes = {issue.code for issue in report.issues}
    assert {
        "DUPLICATE_ROW",
        "MALFORMED_JSON",
        "EMPTY_LINE",
        "MISSING_FIELD",
        "EMPTY_FIELD",
        "IDENTICAL_PREFERENCES",
    }.issubset(codes)


def test_preflight_enforces_char_and_token_bounds(tmp_path: Path) -> None:
    dataset = _write_jsonl(
        tmp_path / "long.jsonl",
        [
            {
                "prompt": "x" * 200,
                "chosen": "y" * 50,
                "rejected": "z" * 50,
            }
        ],
    )

    cfg = pv.PreflightConfig(
        max_prompt_chars=100,
        max_response_chars=100,
        max_estimated_tokens=50,
        scan_pii=False,
    )
    report = pv.validate_jsonl_dataset(dataset, cfg)
    assert report.passed is False
    codes = {issue.code for issue in report.issues}
    assert "PROMPT_TOO_LONG" in codes
    assert "TOKENS_EXCEEDED" in codes


def test_preflight_optional_veilweave_pii_detection(tmp_path: Path) -> None:
    dataset = _write_jsonl(
        tmp_path / "pii.jsonl",
        [
            {
                "prompt": "Draft a reply to alice.smith@example.com about her order.",
                "chosen": "Sure, I can help draft a polite follow-up.",
                "rejected": "Ignore the email.",
            }
        ],
    )

    report = pv.validate_jsonl_dataset(dataset)
    if report.pii_scanner == "veilweave":
        assert report.passed is False
        assert report.pii_findings.get("EMAIL", 0) >= 1
        assert any(issue.code == "PII_DETECTED" for issue in report.issues)
    else:
        assert report.pii_scanner == "unavailable"
        assert report.passed is True
