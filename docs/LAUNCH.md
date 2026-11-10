# Launch draft (not published)

## Problem

Fine-tuning small LoRA adapters on AWS often means **SageMaker jobs or always-on
GPUs**. Teams pay platform markup and operational overhead for **ephemeral**
workloads that should start, train, and disappear.

## Why it matters

- **Cost** — Spot GPU + self-terminating instances align spend with job duration.
- **Simplicity** — One Lambda and one instance per job; no training cluster to babysit.
- **Preference data** — Converting DPO-style logs to SFT (and optional audited DPO)
  is easy to get wrong; TrainWeave encodes that pipeline in tested Python.

## What TrainWeave does

- Deploys a SAM stack (Lambda, IAM, S3 buckets, security group).
- Invokes Lambda with a dataset location or DPO prefix.
- Launches EC2 Spot, runs `bootstrap.sh` + `train.py`, uploads adapters to S3.
- Optionally registers adapters in a DynamoDB catalogue for downstream deploy tooling.

## Honest limits

- **Requires AWS** — Not local-first for training (local dev is pytest + dataset conversion only).
- **GPU tests** — `train.py` is not covered by unit tests in CI.
- **Manual promotion** — Agent aliases and team configs are outside this repo.
- **Heuristic DPO labels** — DPO objective requires a separate judge audit script.
- **DeployWeave / TeamWeave** — Integrations are optional; names refer to companion projects.

## Draft announcement (short)

> **TrainWeave** — open-source Lambda orchestration for **ephemeral EC2 Spot LoRA
> training**. One invoke, one GPU, zero idle cluster. Apache-2.0. Start with
> `pytest` locally, then `sam deploy` when you're ready to spend GPU dollars.

## Communities where it fits

- **AWS ML / serverless** — Spot, SAM, S3 VPC endpoints
- **Hugging Face / PEFT** — LoRA fine-tuning practitioners
- **MLOps** — Teams comparing SageMaker vs DIY GPU
- **RLHF/DPO curious** — Audited preference training path
- **r/LocalLLaMA** (ops angle) — Self-hosted fine-tune orchestration on AWS
- **CNCF/platform engineering** — Ephemeral job pattern (not Kubernetes-native)

## Pre-launch checklist

- [ ] README quickstart verified on clean clone
- [ ] SECURITY.md and CONTRIBUTING.md linked from README
- [ ] Demo GIF or architecture diagram (placeholder in README today)
- [ ] Remove or parameterise account-specific CI defaults for public forks
- [ ] Maintainer security contact confirmed
