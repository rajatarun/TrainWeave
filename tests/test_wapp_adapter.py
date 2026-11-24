"""Unit tests for TrainWeave .wapp skill-adapter sidecar packaging and validation."""

from __future__ import annotations

import copy

import pytest

from trainweave.wapp_adapter import (
    SIDECAR_SCHEMA,
    build_wapp_adapter_sidecar,
    validate_adapter_for_wapp,
)


def _sample_adapter_manifest() -> dict:
    return {
        "schema": "artifact-manifest-v1",
        "adapter_id": "lora-home-ops-v2",
        "base_model": "Qwen/Qwen2.5-7B-Instruct",
        "artifact_sha256": "a" * 64,
    }


def _sample_wapp_manifest() -> dict:
    return {
        "schema": "wapp-manifest-v1",
        "app_id": "com.weave.home-ops",
        "name": "Home Ops Agent",
        "version": "1.2.0",
        "entrypoint": "main.py",
        "permissions": ["mcp:device:read"],
        "min_os_version": "0.3.0",
        "content_sha256": "b" * 64,
    }


def test_build_and_validate_wapp_adapter_sidecar_happy_path() -> None:
    sidecar = build_wapp_adapter_sidecar(
        _sample_adapter_manifest(),
        _sample_wapp_manifest(),
        min_ram_mb=4096,
        target_base_models=["Qwen/Qwen2.5-7B-Instruct", "Qwen/Qwen2.5-3B-Instruct"],
        min_vram_mb=2048,
    )

    assert sidecar["schema"] == SIDECAR_SCHEMA
    assert sidecar["app_id"] == "com.weave.home-ops"
    assert sidecar["wapp_version"] == "1.2.0"
    assert sidecar["adapter_sha256"] == "a" * 64
    assert len(sidecar["sidecar_sha256"]) == 64
    assert sidecar["hardware_requirements"] == {"min_ram_mb": 4096, "min_vram_mb": 2048}

    result = validate_adapter_for_wapp(
        sidecar,
        _sample_wapp_manifest(),
        available_models=["qwen/qwen2.5-7b-instruct", "llama-3.2-3b"],
        available_ram_mb=8192,
    )
    assert result.compatible is True
    assert result.integrity_ok is True
    assert result.matched_base_model == "qwen/qwen2.5-7b-instruct"
    assert result.reasons == []
    assert result.to_dict()["compatible"] is True


def test_validate_adapter_rejects_app_id_mismatch() -> None:
    sidecar = build_wapp_adapter_sidecar(
        _sample_adapter_manifest(),
        _sample_wapp_manifest(),
        min_ram_mb=2048,
        target_base_models=["Qwen/Qwen2.5-7B-Instruct"],
    )
    other_wapp = _sample_wapp_manifest()
    other_wapp["app_id"] = "com.weave.other-app"

    result = validate_adapter_for_wapp(
        sidecar,
        other_wapp,
        available_models=["Qwen/Qwen2.5-7B-Instruct"],
        available_ram_mb=8192,
    )
    assert result.compatible is False
    assert any("App ID mismatch" in r for r in result.reasons)


def test_validate_adapter_rejects_insufficient_ram() -> None:
    sidecar = build_wapp_adapter_sidecar(
        _sample_adapter_manifest(),
        _sample_wapp_manifest(),
        min_ram_mb=8192,
        target_base_models=["Qwen/Qwen2.5-7B-Instruct"],
    )
    result = validate_adapter_for_wapp(
        sidecar,
        _sample_wapp_manifest(),
        available_models=["Qwen/Qwen2.5-7B-Instruct"],
        available_ram_mb=4096,
    )
    assert result.compatible is False
    assert result.required_ram_mb == 8192
    assert result.available_ram_mb == 4096
    assert any("Insufficient device RAM" in r for r in result.reasons)


def test_validate_adapter_rejects_missing_base_model() -> None:
    sidecar = build_wapp_adapter_sidecar(
        _sample_adapter_manifest(),
        _sample_wapp_manifest(),
        min_ram_mb=2048,
        target_base_models=["Qwen/Qwen2.5-7B-Instruct"],
    )
    result = validate_adapter_for_wapp(
        sidecar,
        _sample_wapp_manifest(),
        available_models=["mistral-7b-instruct"],
        available_ram_mb=8192,
    )
    assert result.compatible is False
    assert result.matched_base_model is None
    assert any("No compatible base model" in r for r in result.reasons)


def test_validate_adapter_detects_sidecar_tampering() -> None:
    sidecar = build_wapp_adapter_sidecar(
        _sample_adapter_manifest(),
        _sample_wapp_manifest(),
        min_ram_mb=4096,
        target_base_models=["Qwen/Qwen2.5-7B-Instruct"],
    )
    tampered = copy.deepcopy(sidecar)
    tampered["min_ram_mb"] = 1024

    result = validate_adapter_for_wapp(
        tampered,
        _sample_wapp_manifest(),
        available_models=["Qwen/Qwen2.5-7B-Instruct"],
        available_ram_mb=8192,
    )
    assert result.compatible is False
    assert result.integrity_ok is False
    assert any("integrity verification failed" in r for r in result.reasons)


def test_build_sidecar_validates_inputs() -> None:
    with pytest.raises(ValueError, match="min_ram_mb"):
        build_wapp_adapter_sidecar(
            _sample_adapter_manifest(),
            _sample_wapp_manifest(),
            min_ram_mb=0,
            target_base_models=["Qwen/Qwen2.5-7B-Instruct"],
        )

    with pytest.raises(ValueError, match="target_base_models"):
        build_wapp_adapter_sidecar(
            _sample_adapter_manifest(),
            _sample_wapp_manifest(),
            min_ram_mb=2048,
            target_base_models=[],
        )
