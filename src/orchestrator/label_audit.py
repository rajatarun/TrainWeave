"""
Audit TeamWeave's preference labels against a judge model before training on them.

TeamWeave's ``dpo_collector`` decides chosen vs rejected by which of two answers
has the lower ``composite_risk_score`` -- a weighted mean of *lexical* risk
indicators (hedging words, Jaccard disagreement, numeric spread). That is a
fine gate; whether it is a fine *preference label* is a separate claim that
nothing has checked. Training DPO on it teaches the model whatever the
heuristic rewards -- fewer hedges, say -- and calls it alignment.

This module checks the labels before any GPU time is spent:

  judge_pair   a judge model compares the two answers *twice*, in both orders.
               A verdict counts only when both orders agree; a judge that flips
               with position is measuring position, not quality, and those
               pairs are reported as ``inconsistent`` rather than guessed.
  audit        over a sample: agreement = judged-for-chosen / decisive verdicts,
               with a 95% Wilson interval; agreement by heuristic-delta bucket
               (if bigger deltas do not agree more, the delta is not a strength
               signal); and a decision.

The decision is ``labels_hold`` only when the *lower* bound of the agreement
interval clears ``min_agreement`` (default 0.7) on at least ``min_decisive``
decisive pairs (default 30). Anything else is ``labels_do_not_hold`` or
``insufficient``, and the orchestrator refuses a DPO job on it. Pairs the judge
agreed with are returned separately so a DPO run can train on the verified
subset rather than on every heuristic label.

Pure: the judge is a callable ``invoke(prompt) -> text`` so tests need no model
and the orchestrator Lambda needs no Bedrock client. ``bedrock_invoke`` builds
one for the offline script.
"""
from __future__ import annotations

import hashlib
import json
import math
import random
import re
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Iterable, Sequence

JUDGE_PROMPT = """You are comparing two responses to the same task. Judge which one better accomplishes the task: correct, complete, specific, and faithful to the context. Ignore length and tone unless they affect usefulness.

TASK:
{prompt}

CONTEXT:
{context}

RESPONSE 1:
{first}

RESPONSE 2:
{second}

Reply with JSON only: {{"better": "1" | "2" | "tie", "reason": "<one sentence>"}}"""

MAX_FIELD_CHARS = 6000

#: The judge used when none is named. Claude Sonnet 4.6 is the strongest model
#: this account can call (TeamWeave's config/model_map.yaml lists Sonnet 5 and
#: Opus 5 as unavailable). It is stronger than Haiku 4.5, which writes the
#: Visibility team's drafts and edits. It is also the model TeamWeave's
#: planning agents (director, strategist, daily_operator, the health and
#: finance teams) run on -- for pairs from those steps it would be grading its
#: own answers, so audit them with ``--judge-model deepseek.v3.2`` (a different
#: family on Bedrock) or treat their audit as self-assessment.
DEFAULT_JUDGE_MODEL = "us.anthropic.claude-sonnet-4-6"


def _clip(text: Any) -> str:
    s = text if isinstance(text, str) else json.dumps(text, ensure_ascii=False, sort_keys=True, default=str)
    return s if len(s) <= MAX_FIELD_CHARS else s[:MAX_FIELD_CHARS] + " […truncated]"


def parse_verdict(raw: str) -> str | None:
    """'1', '2', 'tie', or None when the reply cannot be read (never a guess)."""
    if not raw:
        return None
    m = re.search(r"\{.*\}", raw, re.DOTALL)
    try:
        better = json.loads(m.group(0) if m else raw).get("better")
    except (ValueError, AttributeError):
        return None
    better = str(better).strip().lower() if better is not None else None
    return better if better in ("1", "2", "tie") else None


def judge_pair(record: dict, invoke: Callable[[str], str]) -> str:
    """'chosen' | 'rejected' | 'tie' | 'inconsistent' | 'error' for one dpo-v1 record."""
    prompt, context = _clip(record.get("prompt", "")), _clip(record.get("context") or {})
    chosen, rejected = _clip(record.get("chosen", "")), _clip(record.get("rejected", ""))
    try:
        v1 = parse_verdict(invoke(JUDGE_PROMPT.format(prompt=prompt, context=context, first=chosen, second=rejected)))
        v2 = parse_verdict(invoke(JUDGE_PROMPT.format(prompt=prompt, context=context, first=rejected, second=chosen)))
    except Exception:  # noqa: BLE001 - a failed call is not a verdict
        return "error"
    if v1 is None or v2 is None:
        return "error"
    # Map each order's answer to which response it preferred.
    a = {"1": "chosen", "2": "rejected", "tie": "tie"}[v1]
    b = {"1": "rejected", "2": "chosen", "tie": "tie"}[v2]
    if a == b:
        return a
    if "tie" in (a, b):
        return "tie"
    return "inconsistent"


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def record_id(record: dict) -> str:
    """Stable id for a record, so an audit can be matched to the pairs it covered."""
    basis = json.dumps([record.get("prompt"), record.get("chosen"), record.get("rejected")],
                       ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha256(basis.encode()).hexdigest()[:16]


@dataclass
class AuditResult:
    decision: str                       # labels_hold | labels_do_not_hold | insufficient
    reason: str
    sampled: int
    counts: dict[str, int]
    decisive: int
    agreement: float | None
    agreement_ci: tuple[float, float]
    position_consistency: float | None
    by_delta: list[dict[str, Any]]
    min_agreement: float
    min_decisive: int
    judge_model: str = ""
    label_source: str = ""
    verdicts: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def audit(records: Sequence[dict], invoke: Callable[[str], str], *, sample: int | None = None,
          seed: int = 0, min_agreement: float = 0.7, min_decisive: int = 30,
          judge_model: str = "") -> AuditResult:
    pool = [r for r in records if isinstance(r, dict) and r.get("chosen") and r.get("rejected")
            and r.get("chosen") != r.get("rejected")]
    if sample is not None and sample < len(pool):
        pool = random.Random(seed).sample(pool, sample)
    verdicts = {record_id(r): judge_pair(r, invoke) for r in pool}
    counts = {k: 0 for k in ("chosen", "rejected", "tie", "inconsistent", "error")}
    for v in verdicts.values():
        counts[v] += 1
    decisive = counts["chosen"] + counts["rejected"]
    agreement = counts["chosen"] / decisive if decisive else None
    ci = wilson(counts["chosen"], decisive)
    judged = decisive + counts["tie"] + counts["inconsistent"]
    consistency = (judged - counts["inconsistent"]) / judged if judged else None

    buckets: list[dict[str, Any]] = []
    edges = [0.0, 0.5, 0.7, 1.0, math.inf]
    for lo, hi in zip(edges, edges[1:]):
        rows = [r for r in pool if lo <= float(r.get("delta") or 0.0) < hi]
        k = sum(1 for r in rows if verdicts[record_id(r)] == "chosen")
        n = sum(1 for r in rows if verdicts[record_id(r)] in ("chosen", "rejected"))
        buckets.append({"delta": [lo, None if hi == math.inf else hi], "decisive": n,
                        "agreement": round(k / n, 4) if n else None})

    if decisive < min_decisive:
        decision, reason = "insufficient", f"{decisive} decisive verdicts < {min_decisive} required"
    elif ci[0] >= min_agreement:
        decision, reason = "labels_hold", (f"judge agrees with {agreement:.1%} of decisive pairs; "
                                           f"95% lower bound {ci[0]:.1%} >= {min_agreement:.0%}")
    else:
        decision, reason = "labels_do_not_hold", (f"judge agrees with {agreement:.1%} of decisive pairs; "
                                                  f"95% lower bound {ci[0]:.1%} < {min_agreement:.0%}")
    label_sources = {r.get("label_source") for r in pool if r.get("label_source")}
    return AuditResult(
        decision=decision, reason=reason, sampled=len(pool), counts=counts, decisive=decisive,
        agreement=None if agreement is None else round(agreement, 4),
        agreement_ci=(round(ci[0], 4), round(ci[1], 4)),
        position_consistency=None if consistency is None else round(consistency, 4),
        by_delta=buckets, min_agreement=min_agreement, min_decisive=min_decisive,
        judge_model=judge_model,
        label_source=",".join(sorted(label_sources)) or "unrecorded (pre-label_source records)",
        verdicts=verdicts,
    )


def verified_pairs(records: Iterable[dict], result: AuditResult) -> list[dict]:
    """``{prompt, chosen, rejected}`` rows for the pairs the judge agreed with."""
    out = []
    for r in records:
        if result.verdicts.get(record_id(r)) == "chosen":
            out.append({"prompt": str(r.get("prompt", "")).strip(), "context": r.get("context") or {},
                        "chosen": str(r["chosen"]).strip(), "rejected": str(r["rejected"]).strip()})
    return out


def bedrock_invoke(model_id: str, client: Any = None, region: str = "us-east-1") -> Callable[[str], str]:
    """``invoke(prompt) -> text`` over Bedrock Converse at temperature 0."""
    if client is None:
        import boto3
        client = boto3.client("bedrock-runtime", region_name=region)

    def invoke(prompt: str) -> str:
        resp = client.converse(
            modelId=model_id,
            messages=[{"role": "user", "content": [{"text": prompt}]}],
            inferenceConfig={"maxTokens": 200, "temperature": 0},
        )
        for block in resp.get("output", {}).get("message", {}).get("content", []):
            if "text" in block:
                return block["text"]
        return ""

    return invoke
