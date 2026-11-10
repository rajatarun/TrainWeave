# Security notes (threat model sketch)

TrainWeave launches **ephemeral GPU instances** in **your AWS account**. It is
not a multi-tenant SaaS. The main risks are misconfiguration and supply-chain
exposure during bootstrap.

## Assets

| Asset | Location | Sensitivity |
|-------|----------|-------------|
| Training datasets | S3 (your buckets) | Business / PII depending on content |
| LoRA adapters | S3 artifacts bucket | Model weights; may encode proprietary data |
| DPO preference records | Optional DPO bucket | Prompts and model outputs |
| HuggingFace token | Lambda event → EC2 env | Secret if provided |
| Adapter catalogue | Optional DynamoDB | Metadata + S3 paths |

## Trust boundaries

```
Operator / CI ──► Lambda orchestrator ──► EC2 (UserData + bootstrap.sh)
                      │                        │
                      ▼                        ▼
                   AWS EC2 API              S3 / HF Hub / pip
```

- **Lambda** must not be granted more S3/DynamoDB access than needed.
  The template scopes DPO and catalogue permissions behind parameters.
- **EC2** runs arbitrary shell from `bootstrap.sh` pulled from your code bucket.
  Protect the code bucket write path (IAM, bucket policy).
- **UserData** embeds job parameters; anyone with `ec2:DescribeInstanceAttribute`
  or SSM on the instance can read them.

## Primary threats

1. **Bucket takeover** — If an attacker can upload to the training code bucket,
   they can replace `bootstrap.sh`. Mitigation: least-privilege CI role, bucket
   versioning, and deploy only from trusted branches.

2. **Dataset poisoning** — Malicious JSONL or DPO records affect model behavior,
   not host RCE by themselves. Mitigation: validate datasets; audit DPO labels
   before `objective: dpo`.

3. **Spot instance exposure** — Instances have outbound HTTPS; no inbound by
   default. SSM access is optional via `AmazonSSMManagedInstanceCore`.

4. **Catalogue spam** — With `AdapterCatalogTableName` set, training instances
   can `PutItem`. Mitigation: table IAM limited to that table; resolver validates
   adapters before attach (in downstream deploy tooling).

## Out of scope

- Hardening Hugging Face, PyPI, or NVIDIA supply chains (standard ML ops risk).
- Bedrock / agent runtime security (owned by DeployWeave or your agent stack).

## Local / offline use

`scripts/convert_dpo_local.py` and unit tests do not call AWS. They are safe
for untrusted **record JSON** only in the sense that they parse JSON — do not
point them at untrusted multi-gigabyte inputs without resource limits.
