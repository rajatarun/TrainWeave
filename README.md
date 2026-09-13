# TrainWeave

Ephemeral EC2 Spot LoRA fine-tuning orchestration. One Lambda, no SageMaker,
no idle compute, no NAT Gateway.

```
GitHub Actions
    │
    ├─ sam deploy ──────────────────────────────► CloudFormation stack
    │                                              (Lambda + IAM — persists)
    │
    ├─ aws lambda invoke (dataset-a) ──────────► Lambda
    ├─ aws lambda invoke (dataset-b) ──────────► Lambda   ──► EC2 Spot g4dn.xlarge
    └─ aws lambda invoke (dataset-c) ──────────► Lambda       │
                                                               ├─ pull bootstrap.sh (S3 VPC endpoint, free)
                                                               ├─ pull dataset.jsonl (S3 VPC endpoint, free)
                                                               ├─ download base model (NAT instance → HF Hub)
                                                               ├─ run LoRA fine-tuning
                                                               ├─ sync checkpoints → S3 (free)
                                                               ├─ upload adapter → S3 (free)
                                                               └─ shutdown -h now → instance terminates
```

---

---

## The closed loop (TeamWeave → TrainWeave → DeployWeave)

TrainWeave is the middle of a training flywheel that spans three repositories.
Arrows marked **[code]** run by themselves; arrows marked **[manual]** need a
person.

```
 TeamWeave                     TrainWeave                    DeployWeave
 ─────────                     ──────────                    ───────────
 worker_handler invokes each
 step twice, keeps the answer
 with the lower composite risk
        │
        │ [code] dpo_collector writes a chosen/rejected record per step
        ▼
 s3://$DPO_TRAINING_BUCKET/{project}/{team}/{step_id}/{run_id}/dpo_*.json
        │
        │ [manual] someone decides there are enough pairs and invokes the
        │          orchestrator with dataset_source={"type":"dpo",…}
        ▼
                   app.py::_materialise_dpo_dataset          [code]
                     train.jsonl      SFT rows (trained on)
                     train.dpo.jsonl  preference pairs (NOT trained on yet)
                               │
                               │ [code] EC2 spot → bootstrap.sh → train.py
                               ▼
                   s3://$ARTIFACTS_BUCKET/adapters/{job_id}/
                               │
                               │ [code] bootstrap.sh → register_adapter.py
                               │        (skipped when ADAPTER_CATALOG_TABLE unset)
                               ▼
                                          DynamoDB deployweave-adapter-catalog
                                                     │
                                                     │ [code] adapter_resolver
                                                     │        team_provisioner
                                                     ▼
                                          Bedrock agent with the adapter attached
                                                     │
        ┌────────────────────────────────────────────┘
        │ [manual] create/publish the agent alias, then edit team.json in S3:
        │          agents[].bedrock.model_aliases maps model_id → aliasId
        ▼
 TeamWeave's next run uses the new model and produces new pairs
```

### Training on DPO pairs

```bash
aws lambda invoke \
  --function-name "$FUNCTION" \
  --invocation-type Event \
  --payload '{
    "dataset_name": "visibility-draft",
    "dataset_source": {
      "type": "dpo",
      "bucket": "teamweave-dpo-training",
      "prefix": "teamweave/visibility/draft"
    }
  }' \
  /dev/null
```

Before launching the instance the Lambda reads every `dpo-v1` record under the
prefix and writes two files to `s3://$ARTIFACTS_BUCKET/datasets/{job_id}/`:

| File | Contents | Consumed by |
|---|---|---|
| `train.jsonl` | Alpaca `{instruction, input, output}` rows where `output` is the **chosen** response; `input` is the record's `context` as JSON | **`train.py` trains on this file.** It is plain SFT — the rejected response is discarded |
| `train.dpo.jsonl` | `{prompt, chosen, rejected}` rows | **Nothing.** Written so the preference signal survives; training on it needs a `DPOTrainer`, which is a deliberate change of objective, not a config flag |

Requires the `DPOTrainingBucketName` stack parameter to be set — that is what
grants the Lambda read access to TeamWeave's bucket.

### Registering the adapter with DeployWeave

When `AdapterCatalogTableName` is set, `bootstrap.sh` runs
`register_adapter.py` immediately after the adapter upload succeeds (never
before — the catalogue must not advertise an adapter that is not in S3). It
upserts one item in the shape `adapter_resolver` reads: `adapter_id` (derived
from the job id, so a re-run replaces rather than duplicates), `base_model`,
`task_type` (the dataset name), `s3_path`, `tags`/`primary_tag`, `created_at`,
and a `metadata` map carrying the DPO bucket, prefix and record count.

With the parameter empty, registration is skipped with a log line and the job
still succeeds.

> DeployWeave's `lora_validator.py` only accepts base models prefixed
> `amazon.titan-` or `meta.llama`. TrainWeave's default base model is a
> HuggingFace id, so an adapter trained on it is catalogued but
> `team_provisioner` will decline to attach it. `register_adapter.py` logs a
> warning when it sees this.

### What is still manual

| Step | Why |
|---|---|
| Deciding to train, and invoking this Lambda | Nothing watches the DPO prefix for a quorum of pairs. |
| Publishing the Bedrock agent alias for the new adapter | Alias publication is a deliberate promotion, not a side effect of training. |
| Editing `model_aliases` in TeamWeave's `team.json` | Team configs live in S3 and are edited by a human. Nothing writes them back. |
| Training on `train.dpo.jsonl` | `train.py` is SFT-only (TRL `SFTTrainer`). |

---

## Why EC2 Spot over SageMaker

| | EC2 Spot `g4dn.xlarge` | SageMaker `ml.g4dn.xlarge` |
|---|---|---|
| Compute/hr | ~$0.18 (spot) | ~$0.53 (on-demand rate baked in) |
| Platform fee | $0 | ~20–40% overhead |
| Idle cost | $0 — self-terminates | $0 — jobs are ephemeral |
| NAT Gateway | $0 — reuse existing NAT instance | $0 |
| S3 data transfer | **$0** — S3 VPC gateway endpoint | $0 |
| NAT data transfer (pip + HF Hub) | ~$0.045/GB | ~$0.045/GB |
| Lambda orchestration | ~$0.000002/invoke | n/a |
| **3 runs × 2 hr** | **~$1.31** | **~$2.75+** |

Savings come from two places:
1. **No SageMaker markup** — raw spot price vs managed training job price.
2. **S3 VPC gateway endpoint** (`vpce-0c0989507284dc612` in `tarun-teamweave-shared`) —
   dataset downloads and adapter uploads go through the endpoint at $0, never
   touching the NAT instance. Only pip installs and HuggingFace Hub model
   downloads transit the NAT.

### When SageMaker Managed Spot is the better choice

- **Distributed training** — multi-node jobs (Horovod/NCCL) need managed networking
  that SageMaker provides out of the box.
- **Automatic spot interruption recovery** — SageMaker checkpointing integrates
  directly with spot interruption and restarts from the last checkpoint. With this
  setup you get the same result via S3 sync + manual re-run.
- **Compliance requirements** — HIPAA/SOC2 workloads that require a fully managed ML
  environment with SageMaker's audit trail.
- **Experiment tracking** — if your team is already using SageMaker Experiments or
  Model Registry, the managed job integration reduces friction.
- **Long training on large models** — SageMaker's Pipe mode and managed data channels
  outperform manual `aws s3 cp` for very large datasets.

---

## Prerequisites

- AWS CLI v2
- AWS SAM CLI (`pip install aws-sam-cli`)
- An existing VPC with a NAT instance (not NAT Gateway) and S3 VPC gateway endpoint
- Two S3 buckets: one for training artifacts, one for training code
- A Deep Learning AMI ID (Amazon Linux 2 + PyTorch)
- A security group that allows outbound HTTPS (443) from training instances
- GitHub repository with Actions enabled

---

## First-Time Setup

### 1. Create S3 buckets

```bash
aws s3 mb s3://my-trainweave-artifacts --region us-east-1
aws s3 mb s3://my-trainweave-code      --region us-east-1
```

### 2. Find the latest Deep Learning AMI

```bash
aws ec2 describe-images \
  --owners amazon \
  --filters \
    "Name=name,Values=Deep Learning AMI GPU PyTorch*Amazon Linux 2*" \
    "Name=state,Values=available" \
  --query "sort_by(Images, &CreationDate)[-1].[ImageId,Name]" \
  --output text \
  --region us-east-1
```

### 3. Create a GitHub OIDC trust policy (recommended)

This lets GitHub Actions assume an IAM role without static credentials.

```bash
# Create the OIDC provider (one-time per account)
aws iam create-open-id-connect-provider \
  --url https://token.actions.githubusercontent.com \
  --client-id-list sts.amazonaws.com \
  --thumbprint-list 6938fd4d98bab03faadb97b34396831e3780aea1

# Create a role with a trust policy that allows your repo
# (see AWS docs: "Configuring OpenID Connect in AWS")
```

Alternatively, create `AWS_ACCESS_KEY_ID` + `AWS_SECRET_ACCESS_KEY` secrets and
swap the OIDC step in `deploy.yaml` for the static-key block (already commented in
the workflow file).

### 4. Set GitHub Actions secrets

| Secret | Value | Where to find |
|---|---|---|
| `AWS_ROLE_ARN` | `arn:aws:iam::ACCOUNT:role/...` | IAM console |
| `TRAINING_ARTIFACTS_BUCKET` | your artifacts bucket name | S3 console |
| `TRAINING_CODE_BUCKET` | your code bucket name | S3 console |
| `VPC_ID` | `vpc-...` | VPC console |
| `SUBNET_ID` | `subnet-01714c0c1b4590e4e` | `tarun-teamweave-shared` → LambdaSubnetAz1Id |
| `SECURITY_GROUP_ID` | `sg-...` | EC2 → Security Groups |
| `TRAINING_AMI_ID` | `ami-...` | Output from step 2 above |
| `HF_TOKEN` | `hf_...` | HuggingFace → Settings → Access Tokens (only for gated models) |

---

## Deployment

### Deploy infrastructure only

```bash
# Build the SAM application
sam build --use-container

# Deploy (first run — guided prompts for parameter values)
sam deploy --guided

# Subsequent runs (non-interactive)
sam deploy \
  --stack-name trainweave \
  --no-confirm-changeset \
  --no-fail-on-empty-changeset \
  --capabilities CAPABILITY_IAM CAPABILITY_NAMED_IAM \
  --parameter-overrides \
    GlobalBaseModel="meta-llama/Llama-3.2-3B-Instruct" \
    LocalBaseModel="" \
    TrainingArtifactsBucket="my-trainweave-artifacts" \
    TrainingCodeBucket="my-trainweave-code" \
    VpcId="vpc-..." \
    SubnetId="subnet-01714c0c1b4590e4e" \
    SecurityGroupId="sg-..." \
    TrainingAmiId="ami-..."
```

### Upload training code to S3

Run this after any change to `bootstrap.sh` or `train.py`:

```bash
aws s3 sync src/training/ s3://my-trainweave-code/training/ --delete
```

The deploy workflow does this automatically on every run.

### Trigger training via GitHub Actions

1. Go to **Actions → Deploy and Train → Run workflow**
2. Optionally override `global_base_model` for this run
3. Click **Run workflow**

Three EC2 Spot instances will launch (one per dataset), train, and self-terminate.

### Trigger training manually (CLI)

```bash
# Get the function name from the stack
FUNCTION=$(aws cloudformation describe-stacks \
  --stack-name trainweave \
  --query "Stacks[0].Outputs[?OutputKey=='OrchestratorFunctionName'].OutputValue" \
  --output text)

# Launch a single training run
aws lambda invoke \
  --function-name "$FUNCTION" \
  --invocation-type Event \
  --payload '{"dataset_name":"dataset-a","dataset_bucket":"my-ml-datasets","dataset_key":"lora/customer-support/train.jsonl"}' \
  /dev/null
```

---

## How to Add a New Dataset

1. Add a new step to `.github/workflows/deploy.yaml` — copy the `Train — dataset-c`
   block and update `dataset_name`, `dataset_bucket`, and `dataset_key`:

```yaml
- name: Train — dataset-d (my new dataset)
  if: ${{ inputs.skip_training != 'true' }}
  env:
    FUNCTION_NAME: ${{ steps.stack-outputs.outputs.function_name }}
  run: |
    JOB_ID="trainweave-dataset-d-$(date -u +%Y%m%d-%H%M%S)"
    PAYLOAD=$(jq -n \
      --arg job_id "$JOB_ID" \
      --arg dataset_name "dataset-d" \
      --arg dataset_bucket "my-ml-datasets" \
      --arg dataset_key "lora/my-new-task/train.jsonl" \
      '{job_id: $job_id, dataset_name: $dataset_name, dataset_bucket: $dataset_bucket, dataset_key: $dataset_key}')

    aws lambda invoke \
      --function-name "$FUNCTION_NAME" \
      --invocation-type Event \
      --payload "$PAYLOAD" \
      --region "$AWS_REGION" \
      /dev/null
```

2. Upload your dataset to S3:

```bash
aws s3 cp my-new-task.jsonl s3://my-ml-datasets/lora/my-new-task/train.jsonl
```

No infrastructure changes, no `sam deploy` needed.

---

## S3 Output Structure

```
s3://ARTIFACTS_BUCKET/
├── datasets/                              ← only for dataset_source={"type":"dpo"}
│   └── trainweave-dataset-a-20260407-120000/
│       ├── train.jsonl                    ← SFT dataset train.py consumes
│       └── train.dpo.jsonl                ← preference pairs; nothing trains on these yet
├── adapters/
│   └── trainweave-dataset-a-20260407-120000/
│       ├── adapter_config.json
│       ├── adapter_model.safetensors
│       ├── tokenizer.json
│       ├── tokenizer_config.json
│       └── trainweave_run_config.json    ← hyperparameters used for this run
└── checkpoints/
    └── trainweave-dataset-a-20260407-120000/
        ├── checkpoint-100/
        ├── checkpoint-200/
        └── ...
```

---

## Monitoring

**Running instances:**
```bash
aws ec2 describe-instances \
  --filters "Name=tag:trainweave:project,Values=trainweave" "Name=instance-state-name,Values=running,pending" \
  --query "Reservations[*].Instances[*].[InstanceId,InstanceType,Tags[?Key=='Name'].Value|[0],State.Name]" \
  --output table
```

**Lambda logs:**
```bash
aws logs tail /aws/lambda/trainweave-orchestrator --follow
```

**Bootstrap logs (on-instance, while running):**
- Connect via SSM Session Manager (no key pair required): `aws ssm start-session --target INSTANCE_ID`
- `tail -f /var/log/trainweave-bootstrap.log`

---

## Parameters Reference

| Parameter | Default | Description |
|---|---|---|
| `GlobalBaseModel` | — | HuggingFace model ID for all runs |
| `LocalBaseModel` | `""` | If set, overrides `GlobalBaseModel` |
| `TrainingArtifactsBucket` | — | S3 bucket for adapter outputs |
| `TrainingCodeBucket` | — | S3 bucket for bootstrap.sh + train.py |
| `VpcId` | — | Existing VPC |
| `SubnetId` | — | Private subnet (routes via NAT instance) |
| `SecurityGroupId` | — | SG for training instances (allow egress 443) |
| `DefaultTrainingInstanceType` | `g4dn.xlarge` | GPU instance type |
| `SpotMaxPrice` | `""` | Spot max price; empty = AWS automatic bid |
| `TrainingAmiId` | — | Deep Learning AMI ID (Amazon Linux 2) |
| `KeyPairName` | `""` | SSH key pair for debug access |
| `DPOTrainingBucketName` | `""` | TeamWeave's DPO bucket; enables `dataset_source={"type":"dpo"}` |
| `AdapterCatalogTableName` | `""` | DeployWeave adapter catalogue; enables post-upload registration |

---

## Tests

```bash
python3 -m pytest -q
```

Unit tests cover the DPO→dataset conversion (`tests/test_dpo_dataset.py`) and
the DeployWeave catalogue item builder (`tests/test_register_adapter.py`). Both
are pure functions with no AWS calls — no credentials required. The training
stack itself (`train.py`) is not unit tested; it needs a GPU.

---

## Architecture Notes

**Lambda runs outside VPC.** It only calls EC2 and IAM APIs — no VPC config needed.
This avoids ENI provisioning latency and simplifies the stack.

**S3 VPC gateway endpoint.** The `tarun-teamweave-shared` stack already provisions
`vpce-0c0989507284dc612`. EC2 instances in the designated private subnets automatically
route S3 traffic through this endpoint — no code changes, zero NAT data-transfer cost
for dataset downloads and adapter uploads.

**HuggingFace model downloads** go through the NAT instance (`i-0484a84c97a514e60`).
A 7B model is ~14GB; at $0.045/GB NAT data transfer that's ~$0.63 per run. For frequent
re-training of the same model, pre-cache weights in S3 and modify `bootstrap.sh` to
pull from there instead.

**Self-termination** uses `shutdown -h now` + `InstanceInitiatedShutdownBehavior=terminate`
(set by the Lambda in `RunInstances`). The OS-initiated shutdown triggers EC2 termination
without requiring `ec2:TerminateInstances` in the IAM policy.

**No static per-run CloudFormation resources.** The Lambda calls `ec2.run_instances`
directly. Every training run is a single API call — no stack per job, no drift to clean up.
