# Architecture

## Overview

TrainWeave is a **single Lambda** that starts one **EC2 Spot GPU instance** per
training job. The instance downloads `bootstrap.sh` and `train.py` from S3, runs
LoRA fine-tuning, uploads adapters, and terminates.

```
GitHub Actions (optional) ──► sam deploy ──► CloudFormation
Operator / CI ──► lambda invoke ──► Orchestrator ──► EC2 Spot
EC2 ──► S3 (datasets, checkpoints, adapters)
```

## Components

| Component | Path | Role |
|-----------|------|------|
| SAM template | `template.yaml` | Lambda, IAM, buckets, security group |
| Orchestrator | `src/orchestrator/app.py` | Validate event, optional DPO materialisation, `RunInstances` |
| DPO conversion | `src/orchestrator/dpo_dataset.py` | dpo-v1 → Alpaca SFT + preference JSONL |
| Label audit gate | `src/orchestrator/label_audit.py` | Parse audit results for DPO objective |
| Bootstrap | `src/training/bootstrap.sh` | pip install, download model, invoke `train.py` |
| Training | `src/training/train.py` | SFT or DPO via TRL (runs on GPU) |
| Catalogue hook | `src/training/register_adapter.py` | Optional DynamoDB registration |

## Lambda and VPC

The orchestrator runs **outside** the VPC. It only calls EC2 APIs. Training
instances run in a **private subnet** with NAT for Hugging Face / pip and (ideally)
an **S3 gateway endpoint** for free dataset and artifact transfer.

## DPO dataset source

When `dataset_source.type` is `dpo`, the Lambda lists `*.json` under a prefix,
converts records, and writes `datasets/{job_id}/train.jsonl` and
`train.dpo.jsonl` to the artifacts bucket. The EC2 job reads the SFT file unless
`objective` is `dpo` and a passing `audit` reference is supplied.

## Integration with TeamWeave / DeployWeave

Optional parameters wire cross-repo workflows:

- `DPOTrainingBucketName` — read preference records produced upstream
- `AdapterCatalogTableName` — register adapters for downstream resolver

See `docs/design/flywheel.md` for the end-to-end loop narrative (manual steps
called out explicitly).

## Cost model (qualitative)

Compared to managed SageMaker training on the same instance family, TrainWeave
avoids platform markup and **idle** clusters by terminating after each job.
Spot pricing and NAT egress for large model downloads remain variable costs.

Exact dollars depend on region, spot availability, and model size; do not treat
README cost tables as guarantees.
