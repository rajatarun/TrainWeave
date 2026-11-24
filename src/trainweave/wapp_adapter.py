"""Skill-adapter sidecar packaging and compatibility validation for ``.wapp`` bundles.

Links a verified LoRA ``artifact-manifest-v1`` digest to a ``wapp-manifest-v1``
application identity (``app_id`` + ``version``) alongside hardware RAM and
target base-model compatibility metadata, without importing ``weaveos``.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from typing import Any

SIDECAR_SCHEMA = "wapp-skill-adapter-v1"
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def _utc_now_iso() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _canonical_sha256(payload: dict[str, Any]) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _extract_adapter_sha256(adapter_manifest: dict[str, Any]) -> str:
    for key in ("artifact_sha256", "checksum_sha256", "content_sha256", "sha256"):
        candidate = adapter_manifest.get(key)
        if isinstance(candidate, str) and _SHA256_RE.match(candidate):
            return candidate
    return _canonical_sha256(adapter_manifest)


@dataclass(frozen=True)
class AdapterCompatibilityResult:
    """Structured result of validating a skill-adapter sidecar against a target device & .wapp."""

    compatible: bool
    app_id: str
    adapter_id: str
    adapter_sha256: str
    matched_base_model: str | None = None
    required_ram_mb: int = 0
    available_ram_mb: int = 0
    integrity_ok: bool = True
    reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_wapp_adapter_sidecar(
    adapter_manifest: dict[str, Any],
    wapp_manifest: dict[str, Any],
    min_ram_mb: int,
    target_base_models: list[str],
    *,
    min_vram_mb: int | None = None,
) -> dict[str, Any]:
    """Package a verified LoRA adapter into a ``skill_adapter.json`` sidecar dict.

    Binds the ``artifact-manifest-v1`` digest to ``wapp_manifest`` ``app_id`` and
    ``version``, records hardware RAM/VRAM and base-model requirements, and signs
    the canonical sidecar body with ``sidecar_sha256``.
    """
    if not isinstance(adapter_manifest, dict):
        raise TypeError("adapter_manifest must be a dict")
    if not isinstance(wapp_manifest, dict):
        raise TypeError("wapp_manifest must be a dict")

    app_id = wapp_manifest.get("app_id")
    wapp_version = wapp_manifest.get("version")
    if not isinstance(app_id, str) or not app_id.strip():
        raise ValueError("wapp_manifest must include a non-empty 'app_id'")
    if not isinstance(wapp_version, str) or not wapp_version.strip():
        raise ValueError("wapp_manifest must include a non-empty 'version'")

    if isinstance(min_ram_mb, bool) or not isinstance(min_ram_mb, int) or min_ram_mb < 1:
        raise ValueError("min_ram_mb must be an integer >= 1")
    if min_vram_mb is not None and (
        isinstance(min_vram_mb, bool) or not isinstance(min_vram_mb, int) or min_vram_mb < 0
    ):
        raise ValueError("min_vram_mb must be an integer >= 0 when provided")

    if not isinstance(target_base_models, list) or not target_base_models:
        raise ValueError("target_base_models must be a non-empty list of model identifiers")

    cleaned_models: list[str] = []
    seen: set[str] = set()
    for m in target_base_models:
        if not isinstance(m, str) or not m.strip():
            raise ValueError("Each entry in target_base_models must be a non-empty string")
        norm = m.strip()
        if norm not in seen:
            seen.add(norm)
            cleaned_models.append(norm)

    adapter_id = str(
        adapter_manifest.get("adapter_id")
        or adapter_manifest.get("artifact_id")
        or adapter_manifest.get("name")
        or "unnamed-adapter"
    ).strip()
    adapter_sha256 = _extract_adapter_sha256(adapter_manifest)

    body: dict[str, Any] = {
        "schema": SIDECAR_SCHEMA,
        "generated_at": _utc_now_iso(),
        "app_id": app_id.strip(),
        "wapp_version": wapp_version.strip(),
        "adapter_id": adapter_id,
        "artifact_schema": str(adapter_manifest.get("schema", "artifact-manifest-v1")),
        "adapter_sha256": adapter_sha256,
        "base_model": str(adapter_manifest.get("base_model") or cleaned_models[0]),
        "target_base_models": cleaned_models,
        "hardware_requirements": {
            "min_ram_mb": min_ram_mb,
            "min_vram_mb": min_vram_mb,
        },
        "min_ram_mb": min_ram_mb,
    }
    body["sidecar_sha256"] = _canonical_sha256(body)
    return body


def validate_adapter_for_wapp(
    sidecar: dict[str, Any],
    wapp_manifest: dict[str, Any],
    available_models: list[str],
    available_ram_mb: int,
) -> AdapterCompatibilityResult:
    """Validate a skill-adapter sidecar against a ``.wapp`` manifest and device capabilities.

    Verifies:
    1. Canonical ``sidecar_sha256`` and ``adapter_sha256`` integrity.
    2. Exact ``app_id`` binding between ``sidecar`` and ``wapp_manifest``.
    3. Availability of at least one compatible base model in ``available_models``.
    4. Sufficient device RAM (``available_ram_mb >= min_ram_mb``).
    """
    reasons: list[str] = []
    integrity_ok = True

    sidecar_app_id = str(sidecar.get("app_id", "")).strip()
    target_app_id = str(wapp_manifest.get("app_id", "")).strip()
    adapter_id = str(sidecar.get("adapter_id", "")).strip()
    adapter_sha256 = str(sidecar.get("adapter_sha256", "")).strip()

    # 1. SHA-256 integrity check
    if not _SHA256_RE.match(adapter_sha256):
        integrity_ok = False
        reasons.append(
            f"Invalid adapter_sha256 digest '{adapter_sha256}': expected 64 lowercase hex characters."
        )

    recorded_sidecar_sha = sidecar.get("sidecar_sha256")
    if not isinstance(recorded_sidecar_sha, str) or not _SHA256_RE.match(recorded_sidecar_sha):
        integrity_ok = False
        reasons.append("Missing or malformed 'sidecar_sha256' integrity digest.")
    else:
        body_without_sig = {k: v for k, v in sidecar.items() if k != "sidecar_sha256"}
        expected_sha = _canonical_sha256(body_without_sig)
        if expected_sha != recorded_sidecar_sha:
            integrity_ok = False
            reasons.append(
                "Sidecar SHA-256 integrity verification failed: payload has been modified."
            )

    # 2. app_id binding
    if not sidecar_app_id or not target_app_id or sidecar_app_id != target_app_id:
        reasons.append(
            f"App ID mismatch: adapter sidecar is bound to '{sidecar_app_id}', "
            f"but .wapp manifest declares '{target_app_id}'."
        )

    # 3. Base model availability
    raw_targets = sidecar.get("target_base_models")
    target_models = (
        [m.strip() for m in raw_targets if isinstance(m, str) and m.strip()]
        if isinstance(raw_targets, list)
        else []
    )
    avail_lookup = {m.strip().lower(): m.strip() for m in available_models if isinstance(m, str)}
    matched_model: str | None = None
    for tm in target_models:
        if tm.lower() in avail_lookup:
            matched_model = avail_lookup[tm.lower()]
            break

    if matched_model is None:
        reasons.append(
            f"No compatible base model available: adapter requires one of {target_models}, "
            f"device has {list(available_models)}."
        )

    # 4. RAM headroom
    hw = sidecar.get("hardware_requirements")
    if isinstance(hw, dict) and isinstance(hw.get("min_ram_mb"), int):
        required_ram = int(hw["min_ram_mb"])
    elif isinstance(sidecar.get("min_ram_mb"), int):
        required_ram = int(sidecar["min_ram_mb"])
    else:
        required_ram = 0
        reasons.append("Sidecar is missing a valid integer 'min_ram_mb' requirement.")

    if available_ram_mb < required_ram:
        reasons.append(
            f"Insufficient device RAM: adapter requires at least {required_ram} MB, "
            f"but device only has {available_ram_mb} MB available."
        )

    return AdapterCompatibilityResult(
        compatible=len(reasons) == 0,
        app_id=sidecar_app_id,
        adapter_id=adapter_id,
        adapter_sha256=adapter_sha256,
        matched_base_model=matched_model,
        required_ram_mb=required_ram,
        available_ram_mb=available_ram_mb,
        integrity_ok=integrity_ok,
        reasons=reasons,
    )
