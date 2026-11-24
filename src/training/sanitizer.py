"""Local JSONL dataset PII and secret sanitizer for TrainWeave.

Scrubs personal identifiable information (emails, phone numbers, SSNs) and
API credentials (``sk-...``, ``AKIA...``, ``Bearer ...``) from JSONL training
records before fine-tuning. Optionally delegates PII detection to ``veilweave``
when installed, with a deterministic stdlib regex fallback so standalone
installations require zero third-party dependencies.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

try:
    import veilweave as _veilweave
except ImportError:  # pragma: no cover - exercised when veilweave is not installed
    _veilweave = None  # type: ignore[assignment]

_CHUNK_SIZE = 65_536

# Secret / credential patterns are applied first so numeric subsequences inside
# tokens are never partially matched by PII detectors.
_SECRET_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    (
        re.compile(r"\bBearer\s+[A-Za-z0-9._~+/=-]{8,}", re.IGNORECASE),
        "Bearer [REDACTED_TOKEN]",
    ),
    (
        re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
        "[REDACTED_AWS_KEY]",
    ),
    (
        re.compile(r"\bsk-[A-Za-z0-9_-]{8,}\b"),
        "[REDACTED_API_KEY]",
    ),
)

# Deterministic stdlib PII fallback patterns.
_PII_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    (
        re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"),
        "[REDACTED_EMAIL]",
    ),
    (
        re.compile(r"\b(?!000|666|9\d{2})\d{3}-(?!00)\d{2}-(?!0000)\d{4}\b"),
        "[REDACTED_SSN]",
    ),
    (
        re.compile(
            r"(?<!\w)(?:\+?1[-.\s]?)?(?:\(\d{3}\)|\d{3})[-.\s]\d{3}[-.\s]\d{4}(?!\w)"
        ),
        "[REDACTED_PHONE]",
    ),
)


def _build_veilweave_redactor() -> Any | None:
    if _veilweave is None:
        return None
    detectors = (
        _veilweave.detectors.email,
        _veilweave.detectors.phone,
        _veilweave.detectors.ssn,
    )
    return _veilweave.EgressRedactor(detectors=detectors)


def _apply_patterns(
    text: str,
    patterns: tuple[tuple[re.Pattern[str], str], ...],
) -> tuple[str, int]:
    total = 0
    current = text
    for pattern, replacement in patterns:
        current, count = pattern.subn(replacement, current)
        total += count
    return current, total


def sanitize_text(
    text: str,
    *,
    use_veilweave: bool = True,
    _redactor: Any | None = None,
) -> tuple[str, int]:
    """Scrub secrets and PII from a single string, returning ``(cleaned, count)``."""
    cleaned, redactions = _apply_patterns(text, _SECRET_PATTERNS)

    redactor = _redactor
    if redactor is None and use_veilweave:
        redactor = _build_veilweave_redactor()

    if redactor is not None:
        result = redactor.redact_text(cleaned)
        cleaned = result.text
        redactions += sum(int(v) for v in result.redacted_counts.values())
    else:
        cleaned, pii_count = _apply_patterns(cleaned, _PII_PATTERNS)
        redactions += pii_count

    return cleaned, redactions


def sanitize_record(
    record: Any,
    *,
    use_veilweave: bool = True,
    _redactor: Any | None = None,
) -> tuple[Any, int]:
    """Recursively sanitize all string values inside a JSON-compatible object."""
    if isinstance(record, str):
        return sanitize_text(record, use_veilweave=use_veilweave, _redactor=_redactor)
    if isinstance(record, list):
        total = 0
        out_list: list[Any] = []
        for item in record:
            cleaned_item, count = sanitize_record(
                item, use_veilweave=use_veilweave, _redactor=_redactor
            )
            out_list.append(cleaned_item)
            total += count
        return out_list, total
    if isinstance(record, dict):
        total = 0
        out_dict: dict[Any, Any] = {}
        for key, value in record.items():
            cleaned_val, count = sanitize_record(
                value, use_veilweave=use_veilweave, _redactor=_redactor
            )
            out_dict[key] = cleaned_val
            total += count
        return out_dict, total
    return record, 0


def _sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as fh:
        while chunk := fh.read(_CHUNK_SIZE):
            hasher.update(chunk)
    return hasher.hexdigest()


def sanitize_jsonl_dataset(
    input_path: str | Path,
    output_path: str | Path,
    *,
    use_veilweave: bool = True,
) -> dict[str, Any]:
    """Scrub PII and API keys across a JSONL dataset and write sanitized output.

    Preserves valid JSON structure, key order, and non-sensitive training text.
    Blank lines in the input file are skipped.

    Returns:
        A summary dictionary with keys:
        ``records_processed``, ``records_modified``, ``redactions_count``,
        and ``output_sha256``.
    """
    src = Path(input_path)
    if not src.is_file():
        raise FileNotFoundError(f"Input JSONL dataset does not exist: {src}")

    dst = Path(output_path)
    dst.parent.mkdir(parents=True, exist_ok=True)

    redactor = _build_veilweave_redactor() if use_veilweave else None

    records_processed = 0
    records_modified = 0
    redactions_count = 0

    with (
        src.open("r", encoding="utf-8") as infile,
        dst.open("w", encoding="utf-8", newline="\n") as outfile,
    ):
        for line_no, raw_line in enumerate(infile, start=1):
            stripped = raw_line.strip()
            if not stripped:
                continue
            try:
                record = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Malformed JSON on line {line_no} of {src}: {exc}"
                ) from exc

            cleaned_record, count = sanitize_record(
                record, use_veilweave=use_veilweave, _redactor=redactor
            )
            records_processed += 1
            if count > 0:
                records_modified += 1
                redactions_count += count

            outfile.write(json.dumps(cleaned_record, ensure_ascii=False) + "\n")

    output_sha256 = _sha256_file(dst)
    return {
        "records_processed": records_processed,
        "records_modified": records_modified,
        "redactions_count": redactions_count,
        "output_sha256": output_sha256,
    }
