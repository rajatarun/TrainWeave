# TeamWeave → TrainWeave → DeployWeave (optional loop)

This document describes an **optional** multi-repository workflow. TrainWeave
works standalone with JSONL in S3 and Lambda invoke.

## Flow

1. **Preference data** — An upstream system records chosen/rejected pairs
   (schema `dpo-v1`) under an S3 prefix.
2. **SFT training** — Invoke TrainWeave with `dataset_source: { type: dpo, ... }`.
   The orchestrator materialises Alpaca rows from **chosen** responses only.
3. **DPO training (strict)** — Requires `scripts/audit_dpo_labels.py` to produce
   `audit.json` with `decision: labels_hold` and matching bucket/prefix.
4. **Adapter upload** — EC2 writes weights to `s3://ARTIFACTS_BUCKET/adapters/{job_id}/`.
5. **Catalogue (optional)** — If `AdapterCatalogTableName` is set, `register_adapter.py`
   upserts a DynamoDB item for a downstream resolver.
6. **Deployment (manual)** — Publishing model aliases and team config in your
   agent stack is intentionally manual.

## What stays manual

| Step | Why |
|------|-----|
| Deciding when to train | No watcher on DPO prefixes in this repo |
| Running label audit | Judge calls cost money; not invoked by Lambda |
| Publishing agent aliases | Deliberate promotion |
| Pointing agents at new adapters | Team config lives outside TrainWeave |

## DPO objective safety

Heuristic labels (e.g. composite risk scores) are **not** trusted for DPO until
a judge audit passes. The orchestrator enforces audit metadata before copying
verified pairs for `objective: dpo`.
