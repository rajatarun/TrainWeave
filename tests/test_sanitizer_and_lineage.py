"""Unit tests for local dataset PII/secret sanitization and SQLite WAL adapter lineage."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import sanitizer
from lineage import AdapterLineageRegistry
from sanitizer import sanitize_jsonl_dataset


def _write_jsonl(path: Path, lines: list[str]) -> Path:
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


@pytest.mark.parametrize("use_veilweave", [True, False])
def test_sanitize_jsonl_redacts_pii_and_secrets_preserving_structure(
    tmp_path: Path,
    use_veilweave: bool,
) -> None:
    raw_records = [
        {
            "id": 1,
            "prompt": "Email alice.smith@example.com or call (415) 555-0199 about SSN 123-45-6789.",
            "chosen": "Please avoid sharing personal identifiers.",
            "metadata": {
                "nested_notes": [
                    "Key sk-proj-AbCdEf1234567890 leaked",
                    "AWS AKIAIOSFODNN7EXAMPLE found",
                    "Header Authorization: Bearer eyJhbGciOiJIUzI1NiJ9.payload.sig",
                ],
                "score": 0.98,
                "verified": True,
            },
        },
        {
            "id": 2,
            "prompt": "Explain how SQLite WAL mode improves concurrent reads on Raspberry Pi.",
            "chosen": "WAL writes changes to a separate log file so readers never block writers.",
            "metadata": {"version": "3.11.4", "date": "2026-11-24", "tokens": 512},
        },
        {
            "id": 3,
            "prompt": "Reach support at bob@weave.dev or +1-415-555-0142.",
            "chosen": "Use the official support portal instead.",
        },
    ]
    input_path = _write_jsonl(
        tmp_path / "raw.jsonl",
        [json.dumps(r) for r in raw_records],
    )
    output_path = tmp_path / "out" / "sanitized.jsonl"

    summary = sanitize_jsonl_dataset(input_path, output_path, use_veilweave=use_veilweave)

    assert summary["records_processed"] == 3
    assert summary["records_modified"] == 2
    assert summary["redactions_count"] == 8
    expected_sha = hashlib.sha256(output_path.read_bytes()).hexdigest()
    assert summary["output_sha256"] == expected_sha

    output_lines = output_path.read_text(encoding="utf-8").splitlines()
    assert len(output_lines) == 3
    parsed = [json.loads(line) for line in output_lines]

    # Sensitive values must be completely removed from the serialized output.
    serialized = output_path.read_text(encoding="utf-8")
    for sensitive in (
        "alice.smith@example.com",
        "bob@weave.dev",
        "(415) 555-0199",
        "415-555-0142",
        "123-45-6789",
        "sk-proj-AbCdEf1234567890",
        "AKIAIOSFODNN7EXAMPLE",
        "eyJhbGciOiJIUzI1NiJ9.payload.sig",
    ):
        assert sensitive not in serialized

    # Non-sensitive training text, nested structure, and scalar types are preserved.
    assert parsed[0]["id"] == 1
    assert parsed[0]["chosen"] == "Please avoid sharing personal identifiers."
    assert parsed[0]["metadata"]["score"] == 0.98
    assert parsed[0]["metadata"]["verified"] is True
    assert len(parsed[0]["metadata"]["nested_notes"]) == 3

    assert parsed[1] == raw_records[1]
    assert parsed[2]["chosen"] == "Use the official support portal instead."


@pytest.mark.parametrize("use_veilweave", [True, False])
def test_sanitize_jsonl_deterministic_output_sha256(
    tmp_path: Path,
    use_veilweave: bool,
) -> None:
    input_path = _write_jsonl(
        tmp_path / "input.jsonl",
        [
            json.dumps(
                {
                    "prompt": "Contact ops@example.org with token sk-test-1234567890",
                    "response": "Stored securely.",
                }
            ),
            json.dumps(
                {
                    "prompt": "Call 555-234-5678 and use Bearer secret_token_value_99",
                    "response": "Done.",
                }
            ),
        ],
    )
    out_a = tmp_path / "sanitized_a.jsonl"
    out_b = tmp_path / "sanitized_b.jsonl"

    summary_a = sanitize_jsonl_dataset(input_path, out_a, use_veilweave=use_veilweave)
    summary_b = sanitize_jsonl_dataset(input_path, out_b, use_veilweave=use_veilweave)

    assert summary_a == summary_b
    assert out_a.read_bytes() == out_b.read_bytes()


def test_sanitize_jsonl_fallback_when_veilweave_unavailable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(sanitizer, "_veilweave", None)

    input_path = _write_jsonl(
        tmp_path / "input.jsonl",
        [
            json.dumps(
                {
                    "text": (
                        "user@example.com 415.555.0199 123-45-6789 "
                        "sk-live-ABCDEFGH1234 AKIAIOSFODNN7EXAMPLE Bearer abcdef123456"
                    )
                }
            )
        ],
    )
    output_path = tmp_path / "fallback.jsonl"

    summary = sanitize_jsonl_dataset(input_path, output_path, use_veilweave=True)
    assert summary["records_processed"] == 1
    assert summary["records_modified"] == 1
    assert summary["redactions_count"] == 6

    cleaned_text = json.loads(output_path.read_text(encoding="utf-8").strip())["text"]
    assert "[REDACTED_EMAIL]" in cleaned_text
    assert "[REDACTED_PHONE]" in cleaned_text
    assert "[REDACTED_SSN]" in cleaned_text
    assert "[REDACTED_API_KEY]" in cleaned_text
    assert "[REDACTED_AWS_KEY]" in cleaned_text
    assert "Bearer [REDACTED_TOKEN]" in cleaned_text


def test_sanitize_jsonl_skips_blank_lines_and_handles_clean_dataset(tmp_path: Path) -> None:
    input_path = _write_jsonl(
        tmp_path / "clean.jsonl",
        [
            "",
            json.dumps({"prompt": "What is 2 + 2?", "chosen": "4"}),
            "   ",
            json.dumps({"prompt": "Define LoRA.", "chosen": "Low-Rank Adaptation."}),
        ],
    )
    output_path = tmp_path / "clean_out.jsonl"

    summary = sanitize_jsonl_dataset(input_path, output_path)
    assert summary["records_processed"] == 2
    assert summary["records_modified"] == 0
    assert summary["redactions_count"] == 0

    lines = output_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0]) == {"prompt": "What is 2 + 2?", "chosen": "4"}
    assert json.loads(lines[1]) == {"prompt": "Define LoRA.", "chosen": "Low-Rank Adaptation."}


def test_sanitize_jsonl_errors_on_missing_or_malformed_input(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        sanitize_jsonl_dataset(tmp_path / "missing.jsonl", tmp_path / "out.jsonl")

    bad_path = _write_jsonl(tmp_path / "bad.jsonl", ['{"ok": true}', "{not valid json"])
    with pytest.raises(ValueError, match="Malformed JSON on line 2"):
        sanitize_jsonl_dataset(bad_path, tmp_path / "out.jsonl")


def test_lineage_registry_wal_mode_and_persistence_across_reopen(tmp_path: Path) -> None:
    db_path = tmp_path / "lineage" / "adapters.db"

    with AdapterLineageRegistry(db_path) as registry:
        assert registry.journal_mode == "wal"
        stored = registry.record(
            adapter_id="adp-001",
            adapter_name="night-shift-home-v1",
            base_model="Qwen/Qwen2.5-3B-Instruct",
            dataset_sha256="a" * 64,
            manifest_sha256="b" * 64,
            chaos_cert_sha256="c" * 64,
            certified=True,
            created_at="2026-11-23T10:00:00Z",
        )

    assert stored == {
        "adapter_id": "adp-001",
        "adapter_name": "night-shift-home-v1",
        "base_model": "Qwen/Qwen2.5-3B-Instruct",
        "dataset_sha256": "a" * 64,
        "manifest_sha256": "b" * 64,
        "chaos_cert_sha256": "c" * 64,
        "certified": True,
        "created_at": "2026-11-23T10:00:00Z",
    }

    reopened = AdapterLineageRegistry(db_path)
    assert reopened.get("adp-001") == stored
    assert reopened.get("does-not-exist") is None


def test_lineage_registry_filters_by_base_model_and_certified_only(tmp_path: Path) -> None:
    registry = AdapterLineageRegistry(tmp_path / "lineage.db")

    registry.record(
        adapter_id="adp-qwen-cert",
        adapter_name="qwen-certified",
        base_model="Qwen/Qwen2.5-3B-Instruct",
        dataset_sha256="1" * 64,
        manifest_sha256="2" * 64,
        chaos_cert_sha256="3" * 64,
        certified=True,
        created_at="2026-11-23T10:00:00Z",
    )
    registry.record(
        adapter_id="adp-qwen-draft",
        adapter_name="qwen-draft",
        base_model="Qwen/Qwen2.5-3B-Instruct",
        dataset_sha256="4" * 64,
        manifest_sha256="5" * 64,
        chaos_cert_sha256=None,
        certified=False,
        created_at="2026-11-23T11:00:00Z",
    )
    registry.record(
        adapter_id="adp-llama-cert",
        adapter_name="llama-certified",
        base_model="meta-llama/Llama-3.2-3B-Instruct",
        dataset_sha256="6" * 64,
        manifest_sha256="7" * 64,
        chaos_cert_sha256="8" * 64,
        certified=True,
        created_at="2026-11-23T12:00:00Z",
    )

    all_rows = registry.list_lineage()
    assert [r["adapter_id"] for r in all_rows] == [
        "adp-llama-cert",
        "adp-qwen-draft",
        "adp-qwen-cert",
    ]

    qwen_only = registry.list_lineage(base_model="Qwen/Qwen2.5-3B-Instruct")
    assert [r["adapter_id"] for r in qwen_only] == ["adp-qwen-draft", "adp-qwen-cert"]

    certified_only = registry.list_lineage(certified_only=True)
    assert [r["adapter_id"] for r in certified_only] == ["adp-llama-cert", "adp-qwen-cert"]

    qwen_certified = registry.list_lineage("Qwen/Qwen2.5-3B-Instruct", True)
    assert [r["adapter_id"] for r in qwen_certified] == ["adp-qwen-cert"]


def test_lineage_registry_upsert_updates_certification(tmp_path: Path) -> None:
    registry = AdapterLineageRegistry(tmp_path / "lineage.db")

    first = registry.record(
        adapter_id="adp-upsert",
        adapter_name="edge-lora",
        base_model="Qwen/Qwen2.5-3B-Instruct",
        dataset_sha256="d" * 64,
        manifest_sha256="m" * 64,
        certified=False,
    )
    assert first["certified"] is False
    assert first["chaos_cert_sha256"] is None
    assert first["created_at"].endswith("Z")

    updated = registry.record(
        adapter_id="adp-upsert",
        adapter_name="edge-lora",
        base_model="Qwen/Qwen2.5-3B-Instruct",
        dataset_sha256="d" * 64,
        manifest_sha256="m" * 64,
        chaos_cert_sha256="e" * 64,
        certified=True,
    )
    assert updated["certified"] is True
    assert updated["chaos_cert_sha256"] == "e" * 64
    assert len(registry.list_lineage()) == 1


def test_lineage_registry_rejects_empty_required_fields(tmp_path: Path) -> None:
    registry = AdapterLineageRegistry(tmp_path / "lineage.db")
    with pytest.raises(ValueError, match="adapter_id must be a non-empty string"):
        registry.record(
            adapter_id="   ",
            adapter_name="edge-lora",
            base_model="Qwen/Qwen2.5-3B-Instruct",
            dataset_sha256="d" * 64,
            manifest_sha256="m" * 64,
        )


def test_end_to_end_sanitize_to_lineage_flow(tmp_path: Path) -> None:
    raw = _write_jsonl(
        tmp_path / "raw_logs.jsonl",
        [
            json.dumps(
                {
                    "prompt": "User jane@example.com asked to rotate key sk-prod-1234567890",
                    "chosen": "Rotated API key and notified user.",
                }
            )
        ],
    )
    sanitized = tmp_path / "sanitized_logs.jsonl"
    summary = sanitize_jsonl_dataset(raw, sanitized)

    registry = AdapterLineageRegistry(tmp_path / "flywheel_lineage.db")
    entry = registry.record(
        adapter_id="adp-e2e-1",
        adapter_name="sanitized-night-shift",
        base_model="Qwen/Qwen2.5-3B-Instruct",
        dataset_sha256=summary["output_sha256"],
        manifest_sha256=hashlib.sha256(b"manifest-payload").hexdigest(),
        chaos_cert_sha256=hashlib.sha256(b"chaos-cert-payload").hexdigest(),
        certified=True,
    )
    fetched = registry.get("adp-e2e-1")
    assert fetched == entry
    assert fetched is not None
    assert fetched["dataset_sha256"] == summary["output_sha256"]
