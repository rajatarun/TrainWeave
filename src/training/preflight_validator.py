"""Night-Shift JSONL preference dataset pre-flight validator.

Validates incoming DPO / preference JSONL datasets before launching training:
- Required ``prompt``, ``chosen``, and ``rejected`` string fields
- Empty / whitespace-only detection and identical chosen/rejected detection
- Exact duplicate row detection
- Maximum character and estimated token sanity bounds
- Optional ``veilweave`` PII/PHI scan when ``veilweave`` is installed
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

REQUIRED_FIELDS = ("prompt", "chosen", "rejected")


@dataclass(frozen=True)
class PreflightConfig:
    """Thresholds and policy knobs for Night-Shift dataset validation."""

    min_rows: int = 1
    max_prompt_chars: int = 16_000
    max_response_chars: int = 16_000
    max_estimated_tokens: int = 8_192
    allow_duplicates: bool = False
    scan_pii: bool = True
    fail_on_pii: bool = True


@dataclass(frozen=True)
class RowIssue:
    """A single validation error or warning tied to a JSONL line."""

    line_number: int
    code: str
    message: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "line_number": self.line_number,
            "code": self.code,
            "message": self.message,
        }


@dataclass
class PreflightReport:
    """Structured pass/fail report produced by ``validate_jsonl_dataset``."""

    passed: bool
    dataset_path: str
    total_lines: int
    valid_rows: int
    invalid_rows: int
    duplicate_rows: int
    pii_findings: dict[str, int] = field(default_factory=dict)
    pii_scanner: str = "unavailable"
    issues: list[RowIssue] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "dataset_path": self.dataset_path,
            "total_lines": self.total_lines,
            "valid_rows": self.valid_rows,
            "invalid_rows": self.invalid_rows,
            "duplicate_rows": self.duplicate_rows,
            "pii_findings": dict(self.pii_findings),
            "pii_scanner": self.pii_scanner,
            "issues": [issue.to_dict() for issue in self.issues],
        }


def estimate_tokens(text: str) -> int:
    """Conservative heuristic token estimator (~4 chars per token)."""
    if not text:
        return 0
    return max(1, (len(text) + 3) // 4)


def _load_pii_redactor() -> Any | None:
    """Lazily instantiate veilweave.EgressRedactor if veilweave is installed."""
    try:
        import veilweave

        return veilweave.EgressRedactor()
    except ImportError:
        return None


def _row_fingerprint(prompt: str, chosen: str, rejected: str) -> str:
    payload = f"{prompt.strip()}\0{chosen.strip()}\0{rejected.strip()}".encode()
    return hashlib.sha256(payload).hexdigest()


def validate_jsonl_dataset(
    dataset_path: str | Path,
    config: PreflightConfig | None = None,
) -> PreflightReport:
    """Validate a Night-Shift JSONL preference dataset and return a PreflightReport."""
    cfg = config or PreflightConfig()
    path = Path(dataset_path)
    if not path.is_file():
        raise FileNotFoundError(f"Dataset file does not exist: {path}")

    redactor = _load_pii_redactor() if cfg.scan_pii else None
    if not cfg.scan_pii:
        pii_scanner = "disabled"
    elif redactor is not None:
        pii_scanner = "veilweave"
    else:
        pii_scanner = "unavailable"

    issues: list[RowIssue] = []
    pii_findings: dict[str, int] = {}
    seen_fingerprints: set[str] = set()

    total_lines = 0
    valid_rows = 0
    invalid_rows = 0
    duplicate_rows = 0

    with path.open("r", encoding="utf-8") as fh:
        for line_number, raw_line in enumerate(fh, start=1):
            total_lines += 1
            stripped = raw_line.strip()
            if not stripped:
                invalid_rows += 1
                issues.append(
                    RowIssue(line_number, "EMPTY_LINE", "Line is empty or whitespace-only")
                )
                continue

            try:
                row = json.loads(stripped)
            except json.JSONDecodeError as exc:
                invalid_rows += 1
                issues.append(
                    RowIssue(line_number, "MALFORMED_JSON", f"Invalid JSON: {exc.msg}")
                )
                continue

            if not isinstance(row, dict):
                invalid_rows += 1
                issues.append(
                    RowIssue(line_number, "NOT_AN_OBJECT", "JSONL row must be a JSON object")
                )
                continue

            row_errors: list[RowIssue] = []
            values: dict[str, str] = {}

            for field_name in REQUIRED_FIELDS:
                if field_name not in row:
                    row_errors.append(
                        RowIssue(
                            line_number,
                            "MISSING_FIELD",
                            f"Missing required field '{field_name}'",
                        )
                    )
                    continue
                val = row[field_name]
                if not isinstance(val, str):
                    row_errors.append(
                        RowIssue(
                            line_number,
                            "INVALID_FIELD_TYPE",
                            f"Field '{field_name}' must be a string",
                        )
                    )
                    continue
                if not val.strip():
                    row_errors.append(
                        RowIssue(
                            line_number,
                            "EMPTY_FIELD",
                            f"Field '{field_name}' must not be empty",
                        )
                    )
                    continue
                values[field_name] = val

            if not row_errors:
                prompt = values["prompt"]
                chosen = values["chosen"]
                rejected = values["rejected"]

                if chosen.strip() == rejected.strip():
                    row_errors.append(
                        RowIssue(
                            line_number,
                            "IDENTICAL_PREFERENCES",
                            "'chosen' and 'rejected' responses are identical",
                        )
                    )

                if len(prompt) > cfg.max_prompt_chars:
                    row_errors.append(
                        RowIssue(
                            line_number,
                            "PROMPT_TOO_LONG",
                            f"prompt length {len(prompt)} exceeds max {cfg.max_prompt_chars}",
                        )
                    )
                if len(chosen) > cfg.max_response_chars:
                    row_errors.append(
                        RowIssue(
                            line_number,
                            "CHOSEN_TOO_LONG",
                            f"chosen length {len(chosen)} exceeds max {cfg.max_response_chars}",
                        )
                    )
                if len(rejected) > cfg.max_response_chars:
                    row_errors.append(
                        RowIssue(
                            line_number,
                            "REJECTED_TOO_LONG",
                            f"rejected length {len(rejected)} exceeds max {cfg.max_response_chars}",
                        )
                    )

                est_tokens = (
                    estimate_tokens(prompt)
                    + estimate_tokens(chosen)
                    + estimate_tokens(rejected)
                )
                if est_tokens > cfg.max_estimated_tokens:
                    row_errors.append(
                        RowIssue(
                            line_number,
                            "TOKENS_EXCEEDED",
                            f"estimated tokens {est_tokens} exceeds max {cfg.max_estimated_tokens}",
                        )
                    )

                fp = _row_fingerprint(prompt, chosen, rejected)
                if fp in seen_fingerprints:
                    duplicate_rows += 1
                    if not cfg.allow_duplicates:
                        row_errors.append(
                            RowIssue(
                                line_number,
                                "DUPLICATE_ROW",
                                "Duplicate (prompt, chosen, rejected) preference pair",
                            )
                        )
                else:
                    seen_fingerprints.add(fp)

                if redactor is not None:
                    row_pii: dict[str, int] = {}
                    for field_name in REQUIRED_FIELDS:
                        res = redactor.redact_text(values[field_name])
                        for entity, count in res.redacted_counts.items():
                            row_pii[entity] = row_pii.get(entity, 0) + count
                            pii_findings[entity] = pii_findings.get(entity, 0) + count
                    if row_pii and cfg.fail_on_pii:
                        summary = ", ".join(f"{k}={v}" for k, v in sorted(row_pii.items()))
                        row_errors.append(
                            RowIssue(
                                line_number,
                                "PII_DETECTED",
                                f"PII/PHI entities detected ({summary})",
                            )
                        )

            if row_errors:
                invalid_rows += 1
                issues.extend(row_errors)
            else:
                valid_rows += 1

    if valid_rows < cfg.min_rows:
        issues.append(
            RowIssue(
                0,
                "INSUFFICIENT_ROWS",
                f"Valid row count {valid_rows} is below minimum required {cfg.min_rows}",
            )
        )

    return PreflightReport(
        passed=len(issues) == 0,
        dataset_path=str(path),
        total_lines=total_lines,
        valid_rows=valid_rows,
        invalid_rows=invalid_rows,
        duplicate_rows=duplicate_rows,
        pii_findings=pii_findings,
        pii_scanner=pii_scanner,
        issues=issues,
    )
