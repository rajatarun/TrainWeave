"""Tests for edge Qwen2.5/Qwen3 GGUF LoRA metadata in TrainWeave."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src" / "training"))

from register_adapter import build_catalog_item, is_edge_gguf_base_model


def test_is_edge_gguf_base_model() -> None:
    assert is_edge_gguf_base_model("Qwen/Qwen2.5-3B-Instruct") is True
    assert is_edge_gguf_base_model("Qwen/Qwen3-4B") is True
    assert is_edge_gguf_base_model("amazon.titan-text-express-v1") is False


def test_build_catalog_item_with_gguf_provenance() -> None:
    sha = "b" * 64
    item = build_catalog_item(
        job_id="job-pi-001",
        base_model="Qwen/Qwen2.5-3B-Instruct",
        adapter_s3_uri="s3://artifacts/adapters/job-pi-001/",
        dataset_name="tool-calling",
        gguf_artifact_uri="s3://artifacts/adapters/job-pi-001/adapter.gguf",
        gguf_sha256=sha,
        created_at=1700000000,
    )
    assert "format:gguf" in item["tags"]
    assert "target:edge-pi" in item["tags"]
    assert item["metadata"]["edge_runtime"] == "llama.cpp"
    assert item["metadata"]["gguf_sha256"] == sha
    assert item["metadata"]["pinned_version_id"] == f"job-pi-001.gguf@sha256:{sha}"
