# GitHub Actions deploy configuration

The workflow [`.github/workflows/deploy.yaml`](../.github/workflows/deploy.yaml) deploys the SAM stack and optionally launches sample training jobs. Before the next deploy, configure the repository secrets and variables below in **Settings → Secrets and variables → Actions**.

> **Git history:** Older commits may still contain real account IDs or resource identifiers. This document uses placeholders only. Rotate or audit anything that was ever committed if the repository was public.

## Secrets

| Name | Purpose | Example placeholder |
|------|---------|---------------------|
| `AWS_DEPLOY_ROLE_ARN` | IAM role ARN for OIDC (`configure-aws-credentials`). Trust policy must allow this repo's GitHub OIDC subject. | `arn:aws:iam::123456789012:role/example-github-actions-sam-deployer` |

## Variables

| Name | Required | Purpose | Example placeholder |
|------|----------|---------|---------------------|
| `AWS_REGION` | No | AWS Region for deploy and training orchestration. Defaults to `us-east-1` when unset. | `us-east-1` |
| `SAM_STACK_NAME` | No | CloudFormation stack name for this application. Defaults to `trainweave` when unset. | `trainweave` |
| `SHARED_STACK_NAME` | **Yes** | Name of the shared networking stack whose output `LambdaSubnetAz1Id` supplies the training subnet. | `my-org-teamweave-shared` |
| `GLOBAL_BASE_MODEL` | No | Default Hugging Face model id when the workflow dispatch input is empty. | `Qwen/Qwen2.5-0.5B-Instruct` |
| `LOCAL_BASE_MODEL` | No | Optional override model id (non-empty overrides global for the run). | *(empty)* |

## SAM parameters (workflow-supplied)

These are **not** GitHub secrets; the workflow passes them via `sam deploy --parameter-overrides` after resolving VPC/subnet and AMI:

| Parameter | Source |
|-----------|--------|
| `GlobalBaseModel` | Workflow env (`GLOBAL_BASE_MODEL`) |
| `LocalBaseModel` | Workflow env (`LOCAL_BASE_MODEL`) |
| `SubnetId` | `SHARED_STACK_NAME` → output `LambdaSubnetAz1Id` |
| `VpcId` | Derived from subnet via `ec2:DescribeSubnets` |
| `TrainingAmiId` | Latest matching Deep Learning PyTorch AMI in `AWS_REGION` |

Optional integration parameters (`DPOTrainingBucketName`, `AdapterCatalogTableName`, `KeyPairName`, etc.) are not set by the default workflow; add `--parameter-overrides` in the workflow or deploy manually with `sam deploy --guided`.

## Local / manual deploy

For `sam deploy` outside GitHub Actions, supply `VpcId`, `SubnetId`, and `TrainingAmiId` yourself (or use `--guided`). Bucket names in the stack use `AWS::AccountId` and `AWS::Region` — no account id needs to be hardcoded in `template.yaml`.

## Failure modes

- Missing `AWS_DEPLOY_ROLE_ARN`: the **Configure AWS (OIDC)** step fails.
- Missing or wrong `SHARED_STACK_NAME`: VPC resolution fails or returns an empty subnet.
- Missing trust/OIDC on the deploy role: assume-role denied.

Deploy steps are not validated in CI without real AWS credentials; use `skip_training: true` on workflow dispatch to limit cost while testing infra.
