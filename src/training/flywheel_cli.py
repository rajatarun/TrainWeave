"""End-to-end TrainWeave flywheel packaging CLI.

Chains Night-Shift JSONL dataset pre-flight validation, streaming SHA-256
artifact manifest generation, optional GatewayWeave passport verification
(HMAC-SHA256 or Ed25519), and optional ``chaos-cert-v1`` resilience
certification verification into a deployment-ready edge bundle directory
containing ``artifact_manifest.json`` and ``flywheel_bundle.json``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

try:
    from training.artifact_manifest import (
        build_artifact_manifest,
        sha256_file,
        write_artifact_manifest,
    )
    from training.lineage import AdapterLineageRegistry
    from training.preflight_validator import PreflightConfig, validate_jsonl_dataset
    from training.promotion import AdapterPromotionManager, PromotionError
except ImportError:  # pragma: no cover - flat pythonpath fallback (pytest.ini)
    from artifact_manifest import (  # type: ignore[no-redef]
        build_artifact_manifest,
        sha256_file,
        write_artifact_manifest,
    )
    from lineage import AdapterLineageRegistry  # type: ignore[no-redef]
    from preflight_validator import (  # type: ignore[no-redef]
        PreflightConfig,
        validate_jsonl_dataset,
    )
    from promotion import AdapterPromotionManager, PromotionError  # type: ignore[no-redef]

FLYWHEEL_BUNDLE_SCHEMA = "flywheel-bundle-v1"
CHAOS_CERT_SCHEMA = "chaos-cert-v1"
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class FlywheelError(RuntimeError):
    """Raised when a flywheel packaging gate fails."""

    def __init__(
        self,
        message: str,
        *,
        stage: str,
        report: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.stage = stage
        self.report = report or {}


def _canonical_json_sha256(payload: dict[str, Any]) -> str:
    canonical = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def verify_chaos_cert(chaos_cert_path: str | Path) -> dict[str, Any]:
    """Verify a ``chaos-cert-v1`` resilience certification JSON artifact.

    Validates:
    1. ``schema == "chaos-cert-v1"``
    2. ``certified is True``
    3. ``content_sha256`` is a 64-char lowercase hex digest matching the
       canonical JSON SHA-256 over the payload excluding ``content_sha256``.
    """
    path = Path(chaos_cert_path)
    if not path.is_file():
        raise FileNotFoundError(f"Chaos certification file does not exist: {path}")

    cert_file_sha256, _ = sha256_file(path)
    try:
        cert = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise FlywheelError(
            f"Chaos certification JSON is malformed: {exc}",
            stage="chaos_cert",
            report={"chaos_cert_path": str(path), "error": str(exc)},
        ) from exc

    if not isinstance(cert, dict):
        raise FlywheelError(
            "Chaos certification JSON root must be an object",
            stage="chaos_cert",
            report={"chaos_cert_path": str(path)},
        )

    errors: list[str] = []
    schema = cert.get("schema")
    if schema != CHAOS_CERT_SCHEMA:
        errors.append(f"schema is {schema!r}, expected {CHAOS_CERT_SCHEMA!r}")

    certified = cert.get("certified")
    if certified is not True:
        errors.append(f"certified is {certified!r}, expected True")

    claimed_digest = cert.get("content_sha256")
    if not isinstance(claimed_digest, str) or not _SHA256_RE.match(claimed_digest):
        errors.append("content_sha256 must be a 64-char lowercase hex SHA-256 digest")
        computed_digest = None
    else:
        payload = {k: v for k, v in cert.items() if k != "content_sha256"}
        computed_digest = _canonical_json_sha256(payload)
        if claimed_digest != computed_digest:
            errors.append(
                f"content_sha256 mismatch: claimed {claimed_digest}, computed {computed_digest}"
            )

    result = {
        "status": "verified" if not errors else "failed",
        "verified": len(errors) == 0,
        "chaos_cert_path": str(path),
        "file_sha256": cert_file_sha256,
        "schema": schema,
        "subject_id": cert.get("subject_id"),
        "certified": certified,
        "observed_resilience_score": cert.get("observed_resilience_score"),
        "content_sha256": claimed_digest,
        "errors": errors,
    }
    if errors:
        raise FlywheelError(
            f"Chaos certification check failed: {'; '.join(errors)}",
            stage="chaos_cert",
            report=result,
        )
    return result


def run_flywheel_package(
    *,
    dataset_path: str | Path,
    adapter_path: str | Path,
    base_model: str,
    adapter_name: str,
    output_dir: str | Path,
    gguf_path: str | Path | None = None,
    passport_path: str | Path | None = None,
    public_key_path: str | Path | None = None,
    chaos_cert_path: str | Path | None = None,
    integrity_key: str | None = None,
    preflight_config: PreflightConfig | None = None,
) -> dict[str, Any]:
    """Execute the end-to-end TrainWeave flywheel packaging pipeline.

    Steps:
    1. Validate the JSONL preference dataset via ``validate_jsonl_dataset``.
    2. Build and validate the artifact manifest (streaming SHA-256 + pinned
       ``gguf:``/``lora:`` ``model_version_id``), optionally verifying a
       GatewayWeave passport (HMAC-SHA256 or Ed25519).
    3. Optionally verify a ``chaos-cert-v1`` resilience certification JSON.
    4. Write ``artifact_manifest.json`` and ``flywheel_bundle.json`` into
       ``output_dir`` and return the bundle summary dictionary.
    """
    # 1. Dataset pre-flight validation
    preflight = validate_jsonl_dataset(dataset_path, config=preflight_config)
    if not preflight.passed:
        issue_summary = "; ".join(
            f"line {i.line_number} [{i.code}]: {i.message}" for i in preflight.issues[:5]
        )
        raise FlywheelError(
            f"Dataset pre-flight failed ({len(preflight.issues)} issue(s)): {issue_summary}",
            stage="preflight",
            report=preflight.to_dict(),
        )

    # Read optional Ed25519 public key PEM
    public_key_pem: str | None = None
    if public_key_path is not None:
        pk_path = Path(public_key_path)
        if not pk_path.is_file():
            raise FileNotFoundError(f"Public key file does not exist: {pk_path}")
        public_key_pem = pk_path.read_text(encoding="utf-8")
        if not public_key_pem.strip():
            raise FlywheelError(
                f"Public key file is empty: {pk_path}",
                stage="passport",
                report={"public_key_path": str(pk_path)},
            )

    # 2. Build artifact manifest (+ optional GatewayWeave passport verification)
    manifest = build_artifact_manifest(
        adapter_path=adapter_path,
        base_model=base_model,
        adapter_name=adapter_name,
        gguf_path=gguf_path,
        passport_path=passport_path,
        integrity_key=integrity_key,
        public_key_pem=public_key_pem,
        metadata={
            "dataset_path": str(Path(dataset_path)),
            "dataset_valid_rows": preflight.valid_rows,
        },
    )

    passport_status: dict[str, Any]
    if passport_path is not None:
        passport_info = manifest.get("passport", {})
        if not passport_info.get("verified", False):
            err_msg = "; ".join(passport_info.get("errors", [])) or "unknown passport error"
            raise FlywheelError(
                f"GatewayWeave passport verification failed: {err_msg}",
                stage="passport",
                report=passport_info,
            )
        if public_key_pem is not None and not passport_info.get("signature_checked", False):
            raise FlywheelError(
                "Ed25519 public key was provided, but cryptographic signature verification "
                "could not execute (install gatewayweave[asymmetric])",
                stage="passport",
                report=passport_info,
            )
        passport_status = {
            "status": "verified",
            **passport_info,
        }
    else:
        passport_status = {
            "status": "skipped",
            "verified": None,
        }

    # 3. Optional chaos-cert-v1 verification
    chaos_status: dict[str, Any]
    if chaos_cert_path is not None:
        chaos_status = verify_chaos_cert(chaos_cert_path)
    else:
        chaos_status = {
            "status": "skipped",
            "verified": None,
        }

    # 4. Write artifact_manifest.json and flywheel_bundle.json
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = write_artifact_manifest(manifest, out_dir / "artifact_manifest.json")
    manifest_sha256, manifest_size = sha256_file(manifest_path)

    bundle: dict[str, Any] = {
        "schema": FLYWHEEL_BUNDLE_SCHEMA,
        "status": "ready",
        "created_at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "adapter_name": manifest["adapter_name"],
        "base_model": manifest["base_model"],
        "artifact_format": manifest["artifact_format"],
        "model_version_id": manifest["model_version_id"],
        "manifest": {
            "path": str(manifest_path),
            "sha256": manifest_sha256,
            "size_bytes": manifest_size,
        },
        "dataset": {
            "path": preflight.dataset_path,
            "passed": preflight.passed,
            "total_lines": preflight.total_lines,
            "valid_rows": preflight.valid_rows,
            "invalid_rows": preflight.invalid_rows,
            "duplicate_rows": preflight.duplicate_rows,
            "pii_scanner": preflight.pii_scanner,
            "pii_findings": dict(preflight.pii_findings),
        },
        "passport": passport_status,
        "chaos_cert": chaos_status,
    }

    bundle_path = out_dir / "flywheel_bundle.json"
    bundle_path.write_text(json.dumps(bundle, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    bundle["bundle_path"] = str(bundle_path)
    return bundle


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="trainweave-flywheel",
        description="End-to-end TrainWeave local-to-edge personalization packaging CLI.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    pkg = subparsers.add_parser(
        "package",
        help="Validate dataset, emit artifact_manifest.json, verify optional gates, and write flywheel_bundle.json.",
    )
    pkg.add_argument("--dataset", required=True, help="Path to JSONL preference dataset.")
    pkg.add_argument("--adapter", required=True, help="Path to LoRA adapter file or directory.")
    pkg.add_argument("--base-model", required=True, help="Canonical base model identifier.")
    pkg.add_argument("--adapter-name", required=True, help="Logical adapter name (no spaces/@).")
    pkg.add_argument("--out-dir", required=True, help="Output directory for bundle artifacts.")
    pkg.add_argument("--gguf", default=None, help="Optional GGUF artifact path for edge export.")
    pkg.add_argument(
        "--passport",
        default=None,
        help="Optional GatewayWeave passport JSON to verify against model_version_id.",
    )
    pkg.add_argument(
        "--public-key",
        default=None,
        help="Optional Ed25519 public key PEM file for asymmetric passport verification.",
    )
    pkg.add_argument(
        "--integrity-key",
        default=None,
        help="Optional HMAC-SHA256 secret key for symmetric passport verification.",
    )
    pkg.add_argument(
        "--chaos-cert",
        default=None,
        help="Optional chaos-cert-v1 JSON artifact to verify before packaging.",
    )
    pkg.add_argument(
        "--allow-duplicates",
        action="store_true",
        help="Allow duplicate rows during dataset pre-flight validation.",
    )
    pkg.add_argument(
        "--no-pii-scan",
        action="store_true",
        help="Disable optional veilweave PII scanning during dataset pre-flight.",
    )
    pkg.add_argument(
        "--json",
        action="store_true",
        dest="json_output",
        help="Emit machine-readable JSON output.",
    )

    promote_cmd = subparsers.add_parser(
        "promote",
        help="Atomically promote a certified adapter in AdapterLineageRegistry.",
    )
    promote_cmd.add_argument(
        "--db",
        required=True,
        help="Path to SQLite AdapterLineageRegistry / promotion database.",
    )
    promote_cmd.add_argument(
        "--adapter-id",
        required=True,
        help="Registered adapter_id to promote to active.",
    )
    promote_cmd.add_argument(
        "--allow-uncertified",
        action="store_true",
        help="Allow promoting an uncertified adapter (sets require_certified=False).",
    )
    promote_cmd.add_argument(
        "--reason",
        default="",
        help="Optional operator reason recorded in promotion audit history.",
    )
    promote_cmd.add_argument(
        "--json",
        action="store_true",
        dest="json_output",
        help="Emit machine-readable JSON output.",
    )

    rollback_cmd = subparsers.add_parser(
        "rollback",
        help="Atomically roll back the active adapter for a base_model to its previous adapter.",
    )
    rollback_cmd.add_argument(
        "--db",
        required=True,
        help="Path to SQLite AdapterLineageRegistry / promotion database.",
    )
    rollback_cmd.add_argument(
        "--base-model",
        required=True,
        help="Canonical base_model identifier to roll back.",
    )
    rollback_cmd.add_argument(
        "--reason",
        default="",
        help="Optional operator reason recorded in rollback audit history.",
    )
    rollback_cmd.add_argument(
        "--json",
        action="store_true",
        dest="json_output",
        help="Emit machine-readable JSON output.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint for ``trainweave-flywheel``."""
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command == "package":
        cfg = PreflightConfig(
            allow_duplicates=args.allow_duplicates,
            scan_pii=not args.no_pii_scan,
        )
        try:
            bundle = run_flywheel_package(
                dataset_path=args.dataset,
                adapter_path=args.adapter,
                base_model=args.base_model,
                adapter_name=args.adapter_name,
                output_dir=args.out_dir,
                gguf_path=args.gguf,
                passport_path=args.passport,
                public_key_path=args.public_key,
                chaos_cert_path=args.chaos_cert,
                integrity_key=args.integrity_key,
                preflight_config=cfg,
            )
        except FlywheelError as exc:
            err_payload = {
                "status": "failed",
                "stage": exc.stage,
                "error": str(exc),
                "report": exc.report,
            }
            if args.json_output:
                print(json.dumps(err_payload, indent=2, sort_keys=True))
            else:
                print(f"ERROR [{exc.stage}]: {exc}", file=sys.stderr)
            return 2
        except (FileNotFoundError, ValueError, TypeError) as exc:
            err_payload = {
                "status": "failed",
                "stage": "input",
                "error": str(exc),
            }
            if args.json_output:
                print(json.dumps(err_payload, indent=2, sort_keys=True))
            else:
                print(f"ERROR: {exc}", file=sys.stderr)
            return 1

        if args.json_output:
            print(json.dumps(bundle, indent=2, sort_keys=True))
        else:
            print(
                f"Packaged {bundle['model_version_id']} "
                f"(valid_rows={bundle['dataset']['valid_rows']}, "
                f"passport={bundle['passport']['status']}, "
                f"chaos_cert={bundle['chaos_cert']['status']}) -> {bundle['bundle_path']}"
            )
        return 0

    if args.command == "promote":
        try:
            with AdapterLineageRegistry(args.db) as registry:
                manager = AdapterPromotionManager(registry)
                result = manager.promote(
                    args.adapter_id,
                    require_certified=not args.allow_uncertified,
                    reason=args.reason,
                )
        except PromotionError as exc:
            err_payload = {
                "promoted": False,
                "status": "rejected",
                "error": str(exc),
            }
            if args.json_output:
                print(json.dumps(err_payload, indent=2, sort_keys=True))
            else:
                print(f"ERROR [promote]: {exc}", file=sys.stderr)
            return 2
        except (FileNotFoundError, ValueError, TypeError) as exc:
            err_payload = {
                "promoted": False,
                "status": "failed",
                "error": str(exc),
            }
            if args.json_output:
                print(json.dumps(err_payload, indent=2, sort_keys=True))
            else:
                print(f"ERROR: {exc}", file=sys.stderr)
            return 1

        if args.json_output:
            print(json.dumps(result, indent=2, sort_keys=True))
        else:
            print(
                f"Promoted {result['active_adapter_id']} for {result['base_model']} "
                f"(previous={result['previous_adapter_id']})"
            )
        return 0

    if args.command == "rollback":
        try:
            with AdapterLineageRegistry(args.db) as registry:
                manager = AdapterPromotionManager(registry)
                result = manager.rollback(
                    args.base_model,
                    reason=args.reason,
                )
        except PromotionError as exc:
            err_payload = {
                "rolled_back": False,
                "status": "rejected",
                "error": str(exc),
            }
            if args.json_output:
                print(json.dumps(err_payload, indent=2, sort_keys=True))
            else:
                print(f"ERROR [rollback]: {exc}", file=sys.stderr)
            return 2
        except (FileNotFoundError, ValueError, TypeError) as exc:
            err_payload = {
                "rolled_back": False,
                "status": "failed",
                "error": str(exc),
            }
            if args.json_output:
                print(json.dumps(err_payload, indent=2, sort_keys=True))
            else:
                print(f"ERROR: {exc}", file=sys.stderr)
            return 1

        if args.json_output:
            print(json.dumps(result, indent=2, sort_keys=True))
        else:
            print(
                f"Rolled back {result['base_model']} to {result['active_adapter_id']} "
                f"(previous={result['previous_adapter_id']})"
            )
        return 0

    return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
