"""Standardized artifact manifest emitter and optional GatewayWeave passport hook.

Produces deterministic ``artifact_manifest.json`` records with streaming
SHA-256 digests and pinned ``model_version_id`` identifiers
(``gguf:<name>@sha256:<hex>`` or ``lora:<name>@sha256:<hex>``) suitable for
standalone model registries, CI/CD gates, and downstream agent runtimes.
"""

from __future__ import annotations

import hashlib
import json
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

MANIFEST_SCHEMA_VERSION = "1.0"
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_VERSION_ID_RE = re.compile(r"^(gguf|lora):[^@\s]+@sha256:[0-9a-f]{64}$")
_CHUNK_SIZE = 65_536


def sha256_file(path: str | Path, *, chunk_size: int = _CHUNK_SIZE) -> tuple[str, int]:
    """Compute streaming SHA-256 hex digest and total byte size for a file or directory.

    For a single file, streams raw bytes in ``chunk_size`` increments so large
    GGUF or safetensors artifacts never need to be loaded into memory.
    For a directory (e.g., a HuggingFace LoRA adapter directory), hashes all
    files in deterministic sorted relative-path order.
    """
    target = Path(path)
    if not target.exists():
        raise FileNotFoundError(f"Artifact path does not exist: {target}")

    hasher = hashlib.sha256()
    total_bytes = 0

    if target.is_file():
        with target.open("rb") as fh:
            while chunk := fh.read(chunk_size):
                hasher.update(chunk)
                total_bytes += len(chunk)
        return hasher.hexdigest(), total_bytes

    if target.is_dir():
        files = sorted(p for p in target.rglob("*") if p.is_file())
        if not files:
            raise ValueError(f"Artifact directory contains no files: {target}")
        for file_path in files:
            rel = file_path.relative_to(target).as_posix()
            hasher.update(rel.encode("utf-8"))
            hasher.update(b"\0")
            with file_path.open("rb") as fh:
                while chunk := fh.read(chunk_size):
                    hasher.update(chunk)
                    total_bytes += len(chunk)
            hasher.update(b"\0")
        return hasher.hexdigest(), total_bytes

    raise ValueError(f"Unsupported artifact path type: {target}")


def _parse_iso_utc(raw: str) -> datetime:
    text = raw.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    dt = datetime.fromisoformat(text)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return dt.astimezone(UTC)


def verify_gatewayweave_passport(
    passport_path: str | Path,
    *,
    expected_model_version_id: str,
    integrity_key: str | None = None,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Optionally verify a GatewayWeave passport JSON against this artifact.

    Gracefully degrades when ``gatewayweave`` is not installed: structural
    checks (verdict, model_version_id binding, expiration) always execute,
    while cryptographic signature verification runs whenever ``gatewayweave``
    is importable and an integrity block is present.
    """
    p_path = Path(passport_path)
    if not p_path.is_file():
        raise FileNotFoundError(f"Passport file does not exist: {p_path}")

    passport_sha256, _ = sha256_file(p_path)
    try:
        record = json.loads(p_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Passport JSON is malformed: {exc}") from exc

    if not isinstance(record, dict):
        raise TypeError("Passport JSON root must be an object")

    errors: list[str] = []
    verdict = str(record.get("verdict") or "")
    if verdict != "PASS":
        errors.append(f"passport verdict is {verdict!r}, expected 'PASS'")

    passport_version_id = str(record.get("model_version_id") or "").strip()
    if passport_version_id != expected_model_version_id:
        errors.append(
            f"passport model_version_id {passport_version_id!r} does not match "
            f"artifact {expected_model_version_id!r}"
        )

    expires_raw = record.get("expires_at")
    if isinstance(expires_raw, str) and expires_raw.strip():
        try:
            expires_at = _parse_iso_utc(expires_raw)
            ref_now = now if now is not None else datetime.now(UTC)
            if ref_now >= expires_at:
                errors.append(f"passport expired at {expires_raw}")
        except ValueError as exc:
            errors.append(f"invalid passport expires_at: {exc}")

    gatewayweave_installed = False
    signature_checked = False
    try:
        from gatewayweave import integrity as gw_integrity

        gatewayweave_installed = True
        if "integrity" in record:
            key = gw_integrity.resolve_key(integrity_key)
            if key:
                signature_checked = True
                try:
                    gw_integrity.verify_integrity(record, key)
                except Exception as exc:  # noqa: BLE001
                    errors.append(f"gatewayweave integrity check failed: {exc}")
    except ImportError:
        gatewayweave_installed = False

    return {
        "passport_path": str(p_path),
        "passport_sha256": passport_sha256,
        "verdict": verdict,
        "model_version_id": passport_version_id,
        "verified": len(errors) == 0,
        "gatewayweave_installed": gatewayweave_installed,
        "signature_checked": signature_checked,
        "errors": errors,
    }


def validate_artifact_manifest(manifest: dict[str, Any]) -> None:
    """Validate required schema fields and invariants of an artifact manifest."""
    if not isinstance(manifest, dict):
        raise TypeError("Manifest must be a dict")

    required_str = (
        "schema_version",
        "adapter_name",
        "base_model",
        "artifact_format",
        "model_version_id",
        "created_at",
    )
    for key in required_str:
        val = manifest.get(key)
        if not isinstance(val, str) or not val.strip():
            raise ValueError(f"Manifest field '{key}' must be a non-empty string")

    if manifest["artifact_format"] not in {"gguf", "lora"}:
        raise ValueError(
            f"Unsupported artifact_format {manifest['artifact_format']!r}; expected 'gguf' or 'lora'"
        )

    if not _VERSION_ID_RE.match(manifest["model_version_id"]):
        raise ValueError(
            f"Invalid model_version_id format: {manifest['model_version_id']!r}"
        )

    adapter = manifest.get("adapter")
    if not isinstance(adapter, dict):
        raise TypeError("Manifest field 'adapter' must be a dict")
    if not _SHA256_RE.match(str(adapter.get("sha256", ""))):
        raise ValueError("Manifest 'adapter.sha256' must be a 64-char lowercase hex digest")
    if not isinstance(adapter.get("size_bytes"), int) or adapter["size_bytes"] < 0:
        raise ValueError("Manifest 'adapter.size_bytes' must be a non-negative int")

    gguf = manifest.get("gguf")
    if manifest["artifact_format"] == "gguf":
        if not isinstance(gguf, dict):
            raise ValueError("Manifest 'gguf' section is required when artifact_format == 'gguf'")
        if not _SHA256_RE.match(str(gguf.get("sha256", ""))):
            raise ValueError("Manifest 'gguf.sha256' must be a 64-char lowercase hex digest")
        if not isinstance(gguf.get("size_bytes"), int) or gguf["size_bytes"] < 0:
            raise ValueError("Manifest 'gguf.size_bytes' must be a non-negative int")
        expected_id = f"gguf:{manifest['adapter_name']}@sha256:{gguf['sha256']}"
    else:
        expected_id = f"lora:{manifest['adapter_name']}@sha256:{adapter['sha256']}"

    if manifest["model_version_id"] != expected_id:
        raise ValueError(
            f"model_version_id {manifest['model_version_id']!r} does not match "
            f"expected {expected_id!r}"
        )


def build_artifact_manifest(
    adapter_path: str | Path,
    base_model: str,
    adapter_name: str,
    gguf_path: str | Path | None = None,
    passport_path: str | Path | None = None,
    *,
    integrity_key: str | None = None,
    created_at: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build and validate a deterministic ``artifact_manifest.json`` payload.

    Args:
        adapter_path: Path to the trained LoRA adapter file or directory.
        base_model: HuggingFace or canonical base model identifier.
        adapter_name: Logical adapter name (no whitespace or ``@``).
        gguf_path: Optional path to exported GGUF artifact for edge inference.
        passport_path: Optional path to a GatewayWeave passport JSON to verify.
        integrity_key: Optional HMAC key for GatewayWeave passport verification.
        created_at: Optional ISO-8601 UTC timestamp override.
        metadata: Optional extra provenance metadata dictionary.

    Returns:
        Validated manifest dictionary ready for JSON serialization.
    """
    clean_base = base_model.strip()
    clean_name = adapter_name.strip()
    if not clean_base:
        raise ValueError("base_model must be a non-empty string")
    if not clean_name or any(ch.isspace() for ch in clean_name) or "@" in clean_name:
        raise ValueError("adapter_name must be a non-empty string without whitespace or '@'")

    adapter_file = Path(adapter_path)
    adapter_sha, adapter_size = sha256_file(adapter_file)

    gguf_section: dict[str, Any] | None = None
    if gguf_path is not None:
        gguf_file = Path(gguf_path)
        gguf_sha, gguf_size = sha256_file(gguf_file)
        artifact_format = "gguf"
        model_version_id = f"gguf:{clean_name}@sha256:{gguf_sha}"
        gguf_section = {
            "path": str(gguf_file),
            "filename": gguf_file.name,
            "sha256": gguf_sha,
            "size_bytes": gguf_size,
        }
    else:
        artifact_format = "lora"
        model_version_id = f"lora:{clean_name}@sha256:{adapter_sha}"

    timestamp = (
        created_at
        if created_at is not None
        else datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    )

    manifest: dict[str, Any] = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "adapter_name": clean_name,
        "base_model": clean_base,
        "artifact_format": artifact_format,
        "model_version_id": model_version_id,
        "created_at": timestamp,
        "adapter": {
            "path": str(adapter_file),
            "filename": adapter_file.name,
            "sha256": adapter_sha,
            "size_bytes": adapter_size,
        },
        "gguf": gguf_section,
        "metadata": dict(metadata) if metadata else {},
    }

    if passport_path is not None:
        manifest["passport"] = verify_gatewayweave_passport(
            passport_path,
            expected_model_version_id=model_version_id,
            integrity_key=integrity_key,
        )

    validate_artifact_manifest(manifest)
    return manifest


def write_artifact_manifest(manifest: dict[str, Any], output_path: str | Path) -> Path:
    """Validate and write an artifact manifest to disk as canonical JSON."""
    validate_artifact_manifest(manifest)
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return out


def load_artifact_manifest(manifest_path: str | Path) -> dict[str, Any]:
    """Load and validate an artifact manifest JSON file."""
    path = Path(manifest_path)
    data = json.loads(path.read_text(encoding="utf-8"))
    validate_artifact_manifest(data)
    return data
