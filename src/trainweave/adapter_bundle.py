"""Portable LoRA adapter bundling (node-snapshot-v1) and offline promotion gate.

Packages trained LoRA adapter directories alongside their ``artifact-manifest-v1``
metadata into SHA-256-verified ``node-snapshot-v1`` bundles for offline edge
deployment, and gates adapter promotions on quantitative score/latency deltas.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

NODE_SNAPSHOT_SCHEMA = "node-snapshot-v1"
TRAINWEAVE_VERSION = "0.2.0"


def _utc_now_iso() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def compute_components_checksum(components: list[dict[str, Any]]) -> str:
    """Compute canonical SHA-256 hex digest covering the ``components`` array."""
    canonical = json.dumps(components, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def export_adapter_bundle(
    adapter_dir: str | Path,
    manifest: dict[str, Any],
    output_dir: str | Path,
    *,
    node_id: str,
    snapshot_id: str | None = None,
) -> dict[str, Any]:
    """Package an adapter directory and its artifact manifest as a ``node-snapshot-v1`` bundle.

    Copies all files from ``adapter_dir`` into ``output_dir/adapter/``, writes
    ``artifact-manifest.json`` into ``output_dir``, computes per-file SHA-256
    digests and the canonical ``checksum_sha256``, and writes
    ``snapshot-manifest.json``.
    """
    src_dir = Path(adapter_dir).resolve()
    if not src_dir.is_dir():
        raise FileNotFoundError(f"Adapter directory does not exist: {src_dir}")
    if not isinstance(node_id, str) or not node_id.strip():
        raise ValueError("node_id must be a non-empty string")
    if not isinstance(manifest, dict):
        raise TypeError("manifest must be a dict")

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    adapter_out = out_dir / "adapter"
    adapter_out.mkdir(parents=True, exist_ok=True)

    adapter_files = sorted(p for p in src_dir.rglob("*") if p.is_file())
    if not adapter_files:
        raise ValueError(f"Adapter directory contains no files: {src_dir}")

    components: list[dict[str, Any]] = []
    for src_file in adapter_files:
        rel_inside = src_file.relative_to(src_dir)
        dest_file = adapter_out / rel_inside
        dest_file.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src_file, dest_file)

        rel_posix = (Path("adapter") / rel_inside).as_posix()
        components.append(
            {
                "name": f"lora_adapter:{rel_inside.as_posix()}",
                "kind": "lora_adapter",
                "relative_path": rel_posix,
                "sha256": _sha256_file(dest_file),
                "size_bytes": dest_file.stat().st_size,
                "encrypted": False,
            }
        )

    artifact_manifest_path = out_dir / "artifact-manifest.json"
    artifact_manifest_bytes = (
        json.dumps(manifest, indent=2, sort_keys=True).encode("utf-8") + b"\n"
    )
    artifact_manifest_path.write_bytes(artifact_manifest_bytes)
    components.append(
        {
            "name": "artifact_manifest",
            "kind": "json_bundle",
            "relative_path": "artifact-manifest.json",
            "sha256": _sha256_file(artifact_manifest_path),
            "size_bytes": artifact_manifest_path.stat().st_size,
            "encrypted": False,
        }
    )

    snap_id = snapshot_id.strip() if isinstance(snapshot_id, str) and snapshot_id.strip() else (
        f"adapter-snap-{uuid.uuid4().hex[:12]}"
    )

    snapshot_manifest: dict[str, Any] = {
        "schema": NODE_SNAPSHOT_SCHEMA,
        "snapshot_id": snap_id,
        "node_id": node_id.strip(),
        "created_at": _utc_now_iso(),
        "weave_version": TRAINWEAVE_VERSION,
        "components": components,
        "checksum_sha256": compute_components_checksum(components),
    }

    manifest_path = out_dir / "snapshot-manifest.json"
    manifest_path.write_text(
        json.dumps(snapshot_manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return snapshot_manifest


def verify_adapter_bundle(manifest_path: str | Path) -> dict[str, Any]:
    """Verify a ``node-snapshot-v1`` adapter bundle fail-closed.

    Checks manifest schema identifier, ``checksum_sha256``, path traversal
    safety, file existence, byte size, and SHA-256 digest for every component.
    """
    mpath = Path(manifest_path).resolve()
    errors: list[str] = []

    if not mpath.is_file():
        return {
            "valid": False,
            "snapshot_id": None,
            "node_id": None,
            "component_count": 0,
            "errors": [f"Manifest file not found: {mpath}"],
        }

    try:
        data = json.loads(mpath.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return {
            "valid": False,
            "snapshot_id": None,
            "node_id": None,
            "component_count": 0,
            "errors": [f"Failed to parse manifest JSON: {exc}"],
        }

    if not isinstance(data, dict):
        return {
            "valid": False,
            "snapshot_id": None,
            "node_id": None,
            "component_count": 0,
            "errors": ["Manifest root must be a JSON object"],
        }

    if data.get("schema") != NODE_SNAPSHOT_SCHEMA:
        errors.append(f"Invalid schema: expected {NODE_SNAPSHOT_SCHEMA!r}, got {data.get('schema')!r}")

    components = data.get("components")
    if not isinstance(components, list) or len(components) == 0:
        errors.append("Manifest must contain a non-empty 'components' array")
        components = []

    expected_checksum = data.get("checksum_sha256")
    if components:
        actual_checksum = compute_components_checksum(components)
        if expected_checksum != actual_checksum:
            errors.append(
                f"Manifest checksum_sha256 mismatch: expected {expected_checksum!r}, "
                f"computed {actual_checksum!r}"
            )

    base_dir = mpath.parent.resolve()
    for idx, comp in enumerate(components):
        if not isinstance(comp, dict):
            errors.append(f"Component[{idx}] must be an object")
            continue
        rel = comp.get("relative_path")
        expected_sha = comp.get("sha256")
        expected_size = comp.get("size_bytes")
        if not isinstance(rel, str) or not rel:
            errors.append(f"Component[{idx}] missing valid relative_path")
            continue

        target = (base_dir / rel).resolve()
        try:
            target.relative_to(base_dir)
        except ValueError:
            errors.append(f"Component[{idx}] path escapes bundle directory: {rel!r}")
            continue

        if not target.is_file():
            errors.append(f"Component[{idx}] file missing: {rel!r}")
            continue

        actual_size = target.stat().st_size
        if expected_size != actual_size:
            errors.append(
                f"Component[{idx}] size mismatch for {rel!r}: "
                f"expected {expected_size}, got {actual_size}"
            )

        actual_sha = _sha256_file(target)
        if expected_sha != actual_sha:
            errors.append(
                f"Component[{idx}] SHA-256 mismatch for {rel!r}: "
                f"expected {expected_sha!r}, got {actual_sha!r}"
            )

    return {
        "valid": len(errors) == 0,
        "snapshot_id": data.get("snapshot_id"),
        "node_id": data.get("node_id"),
        "component_count": len(components),
        "errors": errors,
    }


def _extract_metric(metrics: dict[str, Any], keys: tuple[str, ...], *, required: bool) -> float:
    for key in keys:
        if key in metrics and metrics[key] is not None:
            val = metrics[key]
            if isinstance(val, bool) or not isinstance(val, (int, float)):
                raise TypeError(f"Metric {key!r} must be numeric")
            return float(val)
    if required:
        raise KeyError(f"Missing required metric; expected one of {keys}")
    return 0.0


def evaluate_promotion_gate(
    candidate_metrics: dict[str, Any],
    baseline_metrics: dict[str, Any],
    *,
    min_score_delta: float = 0.01,
    max_latency_regression_pct: float = 10.0,
) -> dict[str, Any]:
    """Evaluate quantitative score improvement and latency regression for adapter promotion.

    Returns ``{"approved": bool, "score_delta": float, "latency_delta_pct": float,
    "reasons": list[str]}``.
    """
    cand_score = _extract_metric(candidate_metrics, ("score", "eval_score", "accuracy"), required=True)
    base_score = _extract_metric(baseline_metrics, ("score", "eval_score", "accuracy"), required=True)
    cand_latency = _extract_metric(candidate_metrics, ("latency_ms", "p95_latency_ms"), required=False)
    base_latency = _extract_metric(baseline_metrics, ("latency_ms", "p95_latency_ms"), required=False)

    score_delta = round(cand_score - base_score, 6)
    if base_latency > 0.0:
        latency_delta_pct = round(((cand_latency - base_latency) / base_latency) * 100.0, 4)
    elif cand_latency > 0.0:
        latency_delta_pct = 100.0
    else:
        latency_delta_pct = 0.0

    reasons: list[str] = []
    if score_delta < min_score_delta:
        reasons.append(
            f"Score delta {score_delta:+.4f} is below required minimum {min_score_delta:+.4f}"
        )
    if latency_delta_pct > max_latency_regression_pct:
        reasons.append(
            f"Latency regression {latency_delta_pct:+.2f}% exceeds maximum allowed "
            f"{max_latency_regression_pct:.2f}%"
        )

    return {
        "approved": len(reasons) == 0,
        "score_delta": score_delta,
        "latency_delta_pct": latency_delta_pct,
        "reasons": reasons,
    }
