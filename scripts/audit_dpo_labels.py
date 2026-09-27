#!/usr/bin/env python3
"""
Check TeamWeave's heuristic preference labels against a judge model.

Reads the dpo-v1 records under a prefix, has a judge compare each sampled pair
in both orders (label_audit.judge_pair), and writes two objects next to the
records, under ``audits/{audit_id}/`` in the same bucket:

  audit.json            the decision, agreement and its 95% interval, position
                        consistency, agreement by heuristic delta, and the
                        source bucket/prefix it covers
  pairs.verified.jsonl  the pairs the judge agreed with

The orchestrator's ``objective: "dpo"`` refuses to run without an audit.json
whose decision is ``labels_hold`` for the same bucket and prefix, and trains on
the verified pairs only.

  python scripts/audit_dpo_labels.py --bucket teamweave-dpo-training \\
      --prefix teamweave/visibility/draft --sample 200     # judge: Sonnet 4.6 by default

  # offline, against a local JSONL of records, no upload:
  python scripts/audit_dpo_labels.py --file records.jsonl --out audit.json

The judge should be a stronger model than the one that produced the answers,
and not the same one: a model grading its own outputs is self-assessment. The
default, Claude Sonnet 4.6 (label_audit.DEFAULT_JUDGE_MODEL), fits the
Visibility team's writer and editor steps, which run on Haiku 4.5. The planning
steps run on Sonnet 4.6 itself: judge those with --judge-model deepseek.v3.2.
Cost is two judge calls per sampled pair.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "orchestrator"))

import dpo_dataset  # noqa: E402
import label_audit  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--prefix", help="S3 prefix of dpo-v1 records (with --bucket)")
    src.add_argument("--file", help="local JSONL of dpo-v1 records")
    ap.add_argument("--bucket")
    ap.add_argument("--judge-model", default=label_audit.DEFAULT_JUDGE_MODEL)
    ap.add_argument("--region", default="us-east-1")
    ap.add_argument("--sample", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--min-agreement", type=float, default=0.7)
    ap.add_argument("--min-decisive", type=int, default=30)
    ap.add_argument("--out", help="also write audit.json locally")
    args = ap.parse_args(argv)

    if args.file:
        records = [json.loads(l) for l in Path(args.file).read_text().splitlines() if l.strip()]
        s3 = None
    else:
        if not args.bucket:
            ap.error("--prefix needs --bucket")
        import boto3
        s3 = boto3.client("s3", region_name=args.region)
        keys = dpo_dataset.list_record_keys(s3, args.bucket, args.prefix)
        records = list(dpo_dataset.load_records(s3, args.bucket, keys))
    print(f"{len(records)} records; judging a sample of up to {args.sample} (2 calls each) with {args.judge_model}")

    result = label_audit.audit(records, label_audit.bedrock_invoke(args.judge_model, region=args.region),
                               sample=args.sample, seed=args.seed, min_agreement=args.min_agreement,
                               min_decisive=args.min_decisive, judge_model=args.judge_model)
    pairs = label_audit.verified_pairs(records, result)
    audit_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    report = result.to_dict()
    report.update({"audit_id": audit_id, "source_bucket": args.bucket or "", "source_prefix": args.prefix or "",
                   "records_total": len(records), "verified_pairs": len(pairs)})
    base = f"audits/{audit_id}"
    if s3 is not None:
        report["verified_pairs_key"] = f"{base}/pairs.verified.jsonl"
        s3.put_object(Bucket=args.bucket, Key=report["verified_pairs_key"],
                      Body="".join(json.dumps(p, ensure_ascii=False) + "\n" for p in pairs).encode(),
                      ContentType="application/x-ndjson")
        s3.put_object(Bucket=args.bucket, Key=f"{base}/audit.json",
                      Body=json.dumps(report, indent=2).encode(), ContentType="application/json")
        print(f"wrote s3://{args.bucket}/{base}/audit.json")
    summary = {k: report[k] for k in ("decision", "reason", "agreement", "agreement_ci", "counts",
                                       "position_consistency", "by_delta", "verified_pairs")}
    print(json.dumps(summary, indent=2))
    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=2))
    return 0 if result.decision == "labels_hold" else 3


if __name__ == "__main__":
    raise SystemExit(main())
