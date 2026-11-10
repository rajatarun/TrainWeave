# TrainWeave

**Ephemeral EC2 Spot LoRA fine-tuning on AWS — one Lambda invoke, one GPU, no idle cluster.**

TrainWeave is for **ML platform engineers and teams** who want Hugging Face–style LoRA
jobs without standing up SageMaker or a persistent GPU fleet. It orchestrates training
via a single Lambda that launches a self-terminating Spot instance.

> **Cloud required for training.** Local development is offline-friendly (tests +
> dataset conversion). GPU training always runs in **your** AWS account.

<!-- Demo media placeholder: add a GIF or diagram here (e.g. docs/assets/trainweave-demo.gif) -->

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)

## Features

- SAM/CloudFormation stack: Lambda + IAM + S3 buckets + training security group
- EC2 Spot with on-demand fallback; instance self-terminates after training
- Optional **DPO bucket → Alpaca SFT** materialisation (`dataset_source.type: dpo`)
- Optional **audited DPO** training objective (orchestrator enforces label audit)
- Optional **adapter catalogue** registration (DynamoDB) for downstream deploy tooling
- Sample Alpaca datasets under `data/`

Deep dives: [architecture](docs/design/architecture.md),
[optional flywheel](docs/design/flywheel.md),
[integrations](docs/INTEGRATIONS.md),
[extending](docs/EXTENDING.md).

## 5-minute quickstart (local, no AWS)

```bash
git clone <this-repo>
cd TrainWeave
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt pytest
python3 -m pytest -q
```

Convert sample preference records to JSONL offline:

```bash
python3 scripts/convert_dpo_local.py \
  --input examples/dpo-records \
  --output /tmp/trainweave-out
ls /tmp/trainweave-out/
```

Optional synthetic benchmark (results labeled synthetic in JSON output):

```bash
python3 scripts/bench_dpo_conversion.py
```

More examples: [examples/README.md](examples/README.md).

## Core workflow (three copy-paste examples)

### 1. Launch SFT training from JSONL in S3

After [deploy](#aws-deploy), invoke the orchestrator:

```bash
FUNCTION=$(aws cloudformation describe-stacks \
  --stack-name trainweave \
  --query "Stacks[0].Outputs[?OutputKey=='OrchestratorFunctionName'].OutputValue" \
  --output text)

aws lambda invoke \
  --function-name "$FUNCTION" \
  --invocation-type Event \
  --cli-binary-format raw-in-base64-out \
  --payload '{"dataset_name":"customer-support","dataset_bucket":"YOUR_BUCKET","dataset_key":"datasets/customer-support/train.jsonl"}' \
  /dev/null
```

Adapters appear under `s3://ARTIFACTS_BUCKET/adapters/{job_id}/`.

### 2. Materialise DPO records, then train SFT on chosen responses

```bash
aws lambda invoke \
  --function-name "$FUNCTION" \
  --invocation-type Event \
  --cli-binary-format raw-in-base64-out \
  --payload '{
    "dataset_name": "my-draft",
    "dataset_source": {
      "type": "dpo",
      "bucket": "YOUR_DPO_BUCKET",
      "prefix": "project/team/step"
    }
  }' \
  /dev/null
```

Requires stack parameter `DPOTrainingBucketName` and IAM read access to that bucket.

### 3. Train DPO objective only after a passing label audit

```bash
# Run offline or on a workstation with AWS + judge access:
python scripts/audit_dpo_labels.py --bucket YOUR_DPO_BUCKET \
  --prefix project/team/step --sample 200

aws lambda invoke \
  --function-name "$FUNCTION" \
  --invocation-type Event \
  --cli-binary-format raw-in-base64-out \
  --payload '{
    "dataset_name": "my-draft-dpo",
    "objective": "dpo",
    "dataset_source": {
      "type": "dpo",
      "bucket": "YOUR_DPO_BUCKET",
      "prefix": "project/team/step",
      "audit": {"key": "audits/YOUR_AUDIT_ID/audit.json"}
    }
  }' \
  /dev/null
```

See [examples/lambda-payloads.json](examples/lambda-payloads.json) for the same shapes as JSON.

## AWS deploy

**Prerequisites:** AWS CLI v2, SAM CLI, a VPC with outbound path (NAT instance or
equivalent), Deep Learning GPU AMI, and permissions to deploy IAM + Lambda + EC2.

```bash
sam build
sam deploy --guided
aws s3 sync src/training/ s3://YOUR_CODE_BUCKET/training/
```

GitHub Actions workflow `.github/workflows/deploy.yaml` can deploy and launch sample
runs (training costs money — use workflow input `skip_training: true` for infra-only).

| Parameter | Purpose |
|-----------|---------|
| `GlobalBaseModel` | Default Hugging Face model id |
| `TrainingAmiId` | GPU Deep Learning AMI |
| `VpcId` / `SubnetId` | Where training instances run |
| `DPOTrainingBucketName` | Enable DPO dataset source |
| `AdapterCatalogTableName` | Enable post-train catalogue registration |

Full parameter list: see `template.yaml` and the legacy detailed sections in git
history or expand in your fork as needed.

### S3 layout (outputs)

```
s3://ARTIFACTS_BUCKET/
├── adapters/{job_id}/          # LoRA weights + tokenizer
├── checkpoints/{job_id}/       # optional checkpoints
└── datasets/{job_id}/          # when using dataset_source dpo
```

### Monitoring

```bash
aws logs tail /aws/lambda/trainweave-orchestrator --follow
aws ec2 describe-instances \
  --filters "Name=tag:trainweave:project,Values=trainweave" \
            "Name=instance-state-name,Values=running,pending"
```

## Tests

```bash
python3 -m pytest -q
```

CI-friendly unit tests cover DPO conversion, orchestrator dataset materialisation,
adapter catalogue items, and label audit parsing. `train.py` requires a GPU and is not
run in CI.

## Contributing / security

- [CONTRIBUTING.md](CONTRIBUTING.md)
- [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md)
- [SECURITY.md](SECURITY.md)
- [Good first issues (draft)](docs/GOOD_FIRST_ISSUES.md)

## License

Apache License 2.0 — see [LICENSE](LICENSE).
