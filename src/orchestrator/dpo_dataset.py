"""
Conversion of TeamWeave DPO preference records into TrainWeave datasets.

TeamWeave's ``dpo_collector`` invokes every pipeline step twice, keeps the
answer with the lower ``composite_risk_score``, and uploads a chosen/rejected
record to::

    s3://{DPO_TRAINING_BUCKET}/{project}/{team}/{step_id}/{run_id}/dpo_{ts}.json

Each object is a single JSON document (``schema_version: "dpo-v1"``) with the
fields consumed here:

    prompt    str   — the step input that was sent to the agent
    context   dict  — structured context supplied alongside the prompt
    chosen    str   — the lower-risk response
    rejected  str   — the higher-risk response

Two files are produced from one set of records:

``train.jsonl`` (SFT)
    Alpaca rows ``{instruction, input, output}`` where ``output`` is the
    *chosen* response only.  **This is the file train.py consumes** — it
    fine-tunes with TRL's ``SFTTrainer``, which has no notion of a rejected
    response.  Rejected responses are discarded here.

``train.dpo.jsonl`` (preference pairs)
    Rows ``{prompt, chosen, rejected}``.  **Nothing in this repository trains
    on this file yet.**  It is written alongside the SFT dataset so the
    preference signal is preserved for a future ``DPOTrainer`` run; adopting
    it is a deliberate change of training objective, not a side effect of
    selecting a DPO dataset source.
"""

from __future__ import annotations

import json
from typing import Any, Iterable, NamedTuple

# Suffixes appended to the derived-dataset prefix in the artifacts bucket.
SFT_DATASET_FILENAME = "train.jsonl"
DPO_DATASET_FILENAME = "train.dpo.jsonl"


class Conversion(NamedTuple):
    """Result of converting a batch of DPO records."""

    sft_rows: list[dict]
    dpo_rows: list[dict]
    skipped: int


def _text(value: Any) -> str:
    """Return a stripped string for str values, empty string for anything else."""
    return value.strip() if isinstance(value, str) else ""


def render_context(context: Any) -> str:
    """Render a DPO record's ``context`` into the Alpaca ``input`` field.

    ``context`` is a free-form dict in the dpo-v1 schema.  It is serialised
    with sorted keys so the same record always produces the same dataset row
    (re-running a conversion must be a no-op).  Non-dict or empty contexts
    become an empty string, which ``train.py::_format_alpaca`` treats as
    "no input section".
    """
    if not isinstance(context, dict) or not context:
        return ""
    return json.dumps(context, ensure_ascii=False, sort_keys=True, default=str)


def sft_example(record: dict) -> dict | None:
    """Convert one dpo-v1 record to an Alpaca SFT row, or None if unusable.

    Only the ``chosen`` response becomes a training target — the SFT objective
    cannot express a rejection.
    """
    instruction = _text(record.get("prompt"))
    output = _text(record.get("chosen"))
    if not instruction or not output:
        return None
    return {
        "instruction": instruction,
        "input": render_context(record.get("context")),
        "output": output,
    }


def dpo_pair(record: dict) -> dict | None:
    """Convert one dpo-v1 record to a ``{prompt, chosen, rejected}`` row.

    Returns None when the pair is degenerate (missing side, or the two
    responses are identical — such a pair carries no preference signal).
    """
    prompt = _text(record.get("prompt"))
    chosen = _text(record.get("chosen"))
    rejected = _text(record.get("rejected"))
    if not prompt or not chosen or not rejected or chosen == rejected:
        return None
    return {"prompt": prompt, "chosen": chosen, "rejected": rejected}


def convert_records(records: Iterable[dict]) -> Conversion:
    """Convert dpo-v1 records into SFT rows and preference rows.

    A record that yields no SFT row is counted as skipped.  A record that
    yields an SFT row but no usable pair (e.g. identical responses) still
    contributes to training — it is not counted as skipped.
    """
    sft_rows: list[dict] = []
    dpo_rows: list[dict] = []
    skipped = 0

    for record in records:
        if not isinstance(record, dict):
            skipped += 1
            continue
        sft = sft_example(record)
        if sft is None:
            skipped += 1
            continue
        sft_rows.append(sft)
        pair = dpo_pair(record)
        if pair is not None:
            dpo_rows.append(pair)

    return Conversion(sft_rows=sft_rows, dpo_rows=dpo_rows, skipped=skipped)


def to_jsonl(rows: Iterable[dict]) -> str:
    """Serialise rows as JSONL (one compact JSON object per line)."""
    return "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)


def list_record_keys(s3, bucket: str, prefix: str) -> list[str]:
    """List every ``.json`` DPO record key under ``prefix``, sorted.

    The prefix is the caller's slice of the DPO bucket — typically
    ``{project}/{team}/{step_id}`` — and is matched as written; sorting keeps
    the produced dataset byte-identical across runs over the same objects.
    """
    keys: list[str] = []
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if key.endswith(".json"):
                keys.append(key)
    return sorted(keys)


def load_records(s3, bucket: str, keys: Iterable[str]) -> list[dict]:
    """Fetch and parse DPO record objects; unreadable objects are skipped."""
    records: list[dict] = []
    for key in keys:
        body = s3.get_object(Bucket=bucket, Key=key)["Body"].read()
        try:
            records.append(json.loads(body))
        except (ValueError, TypeError):
            continue
    return records
