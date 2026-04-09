"""
TrainWeave orchestrator Lambda.

Receives a training job request, resolves the effective base model,
builds EC2 UserData, and launches a single EC2 Spot GPU instance.
Returns immediately — the instance bootstraps, trains, and self-terminates.

Environment variables (set by SAM template):
  GLOBAL_BASE_MODEL      HuggingFace model ID or local path
  LOCAL_BASE_MODEL       If non-empty, overrides GLOBAL_BASE_MODEL
  ARTIFACTS_BUCKET       S3 bucket for adapter outputs and checkpoints
  CODE_BUCKET            S3 bucket holding bootstrap.sh and train.py
  SUBNET_ID              Private subnet for EC2 (routes via NAT instance)
  SECURITY_GROUP_ID      SG for training instances
  INSTANCE_PROFILE_ARN   EC2 instance profile ARN
  DEFAULT_INSTANCE_TYPE  Fallback instance type (e.g. g4dn.xlarge)
  SPOT_MAX_PRICE         Max spot bid; empty = AWS automatic (on-demand cap)
  TRAINING_AMI_ID        Deep Learning AMI ID
  KEY_PAIR_NAME          SSH key pair; empty = no key (production default)
"""

import base64
import json
import logging
import os
import time
from datetime import datetime, timezone

import boto3
from botocore.exceptions import ClientError

logger = logging.getLogger()
logger.setLevel(logging.INFO)

ec2 = boto3.client("ec2")

# Spot errors that are worth retrying (transient capacity/limit issues)
_SPOT_RETRYABLE = {
    "MaxSpotInstanceCountExceeded",
    "InsufficientInstanceCapacity",
    "SpotMaxPriceTooLow",
}


def _run_instances_with_retry(run_kwargs: dict, max_retries: int = 4) -> dict:
    """
    Call ec2.run_instances with exponential backoff on transient spot errors.

    Retries up to max_retries times (delays: 2s, 4s, 8s, 16s).
    Raises the original exception if retries are exhausted.
    """
    for attempt in range(max_retries + 1):
        try:
            return ec2.run_instances(**run_kwargs)
        except ClientError as exc:
            code = exc.response["Error"]["Code"]
            if code in _SPOT_RETRYABLE and attempt < max_retries:
                delay = 2 ** (attempt + 1)  # 2, 4, 8, 16 s
                logger.warning(
                    "Spot error %s (attempt %d/%d) — retrying in %ds",
                    code, attempt + 1, max_retries, delay,
                )
                time.sleep(delay)
            else:
                raise


def _launch_instance(run_kwargs: dict) -> tuple[dict, str]:
    """
    Try spot first; fall back to on-demand if spot quota is exhausted.

    Returns (response, market_type) where market_type is 'spot' or 'on-demand'.
    """
    try:
        response = _run_instances_with_retry(run_kwargs)
        return response, "spot"
    except ClientError as exc:
        if exc.response["Error"]["Code"] in _SPOT_RETRYABLE:
            logger.warning(
                "Spot exhausted after retries (%s) — falling back to on-demand",
                exc.response["Error"]["Code"],
            )
            on_demand_kwargs = {k: v for k, v in run_kwargs.items() if k != "InstanceMarketOptions"}
            return ec2.run_instances(**on_demand_kwargs), "on-demand"
        raise



def _resolve_effective_model(local: str, global_: str) -> str:
    """Return local model if set, otherwise fall back to global model."""
    stripped = local.strip()
    if stripped:
        logger.info("Using LocalBaseModel: %s", stripped)
        return stripped
    logger.info("Using GlobalBaseModel: %s", global_)
    return global_


def _build_userdata(env_vars: dict[str, str]) -> str:
    """
    Build a minimal shell script that exports env vars and pulls + runs
    bootstrap.sh from S3. The instance role grants S3 access; the S3 VPC
    gateway endpoint (already present in tarun-teamweave-shared) ensures
    this call never transits the NAT instance.

    Returns base64-encoded string ready for EC2 RunInstances UserData field.
    """
    # Export every env var, quoting values to handle spaces/special chars
    exports = "\n".join(
        f'export {k}={json.dumps(str(v))}' for k, v in sorted(env_vars.items())
    )

    script = f"""#!/bin/bash
set -euo pipefail

# ── Environment ──────────────────────────────────────────────────────────────
{exports}

# ── Logging ──────────────────────────────────────────────────────────────────
# Send all output to /var/log/trainweave-userdata.log so it survives until
# bootstrap.sh takes over with its own logging setup.
exec > >(tee /var/log/trainweave-userdata.log) 2>&1
echo "[trainweave-userdata] Starting job $JOB_ID on $(date -u +%Y-%m-%dT%H:%M:%SZ)"

# ── Pull and run bootstrap ────────────────────────────────────────────────────
# S3 traffic routes through the VPC gateway endpoint (no NAT cost).
aws s3 cp "s3://$CODE_BUCKET/training/bootstrap.sh" /tmp/trainweave-bootstrap.sh
chmod +x /tmp/trainweave-bootstrap.sh
bash /tmp/trainweave-bootstrap.sh
"""
    return base64.b64encode(script.encode()).decode()


def handler(event: dict, context) -> dict:
    """
    Event schema:
    {
        "dataset_name":   "dataset-a",           # required
        "dataset_bucket": "my-ml-datasets",       # required
        "dataset_key":    "lora/cs/train.jsonl",  # required
        "instance_type":  "g4dn.xlarge",          # optional override
        "job_id":         "my-custom-job-id"      # optional; auto-generated if absent
        "hf_token":       "hf_..."                # optional; for gated models
    }
    """
    # ── Validate required fields ───────────────────────────────────────────────
    for field in ("dataset_name", "dataset_bucket", "dataset_key"):
        if not event.get(field):
            raise ValueError(f"Missing required event field: {field}")

    dataset_name: str = event["dataset_name"]
    dataset_bucket: str = event["dataset_bucket"]
    dataset_key: str = event["dataset_key"]
    instance_type: str = event.get("instance_type") or os.environ["DEFAULT_INSTANCE_TYPE"]
    hf_token: str = event.get("hf_token", "")

    # ── Resolve effective base model ───────────────────────────────────────────
    effective_model = _resolve_effective_model(
        local=os.environ.get("LOCAL_BASE_MODEL", ""),
        global_=os.environ["GLOBAL_BASE_MODEL"],
    )

    # ── Build deterministic job ID ─────────────────────────────────────────────
    job_id: str = event.get("job_id") or (
        f"trainweave-{dataset_name}-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}"
    )

    logger.info(
        "Launching training job | job_id=%s dataset=%s model=%s instance=%s",
        job_id, dataset_name, effective_model, instance_type,
    )

    # ── Build UserData env vars ────────────────────────────────────────────────
    env_vars = {
        "JOB_ID": job_id,
        "EFFECTIVE_MODEL": effective_model,
        "DATASET_BUCKET": dataset_bucket,
        "DATASET_KEY": dataset_key,
        "ARTIFACTS_BUCKET": os.environ["ARTIFACTS_BUCKET"],
        "CODE_BUCKET": os.environ["CODE_BUCKET"],
        "AWS_DEFAULT_REGION": os.environ.get("AWS_REGION", "us-east-1"),
    }
    # Only inject HF_TOKEN if provided — avoids empty env var breaking HF CLI
    if hf_token:
        env_vars["HF_TOKEN"] = hf_token

    userdata = _build_userdata(env_vars)

    # ── Build RunInstances kwargs ──────────────────────────────────────────────
    spot_options: dict = {
        "SpotInstanceType": "one-time",  # cheapest; no persistent request
    }
    max_price = os.environ.get("SPOT_MAX_PRICE", "").strip()
    if max_price:
        # Only set when explicitly configured; empty = AWS caps at on-demand rate
        spot_options["MaxPrice"] = max_price

    run_kwargs: dict = {
        "ImageId": os.environ["TRAINING_AMI_ID"],
        "InstanceType": instance_type,
        "MinCount": 1,
        "MaxCount": 1,
        "SubnetId": os.environ["SUBNET_ID"],
        "SecurityGroupIds": [os.environ["SECURITY_GROUP_ID"]],
        "IamInstanceProfile": {"Arn": os.environ["INSTANCE_PROFILE_ARN"]},
        "UserData": userdata,
        # Instance shuts down → terminate. Paired with `shutdown -h now` in
        # bootstrap.sh. No ec2:TerminateInstances permission needed.
        "InstanceInitiatedShutdownBehavior": "terminate",
        "InstanceMarketOptions": {
            "MarketType": "spot",
            "SpotOptions": spot_options,
        },
        # Tag the instance at launch so it appears named in the EC2 console
        # immediately, even before the instance is running.
        "TagSpecifications": [
            {
                "ResourceType": "instance",
                "Tags": [
                    {"Key": "Name", "Value": f"trainweave-{job_id}"},
                    {"Key": "trainweave:job-id", "Value": job_id},
                    {"Key": "trainweave:dataset", "Value": dataset_name},
                    {"Key": "trainweave:model", "Value": effective_model},
                    {"Key": "trainweave:project", "Value": "trainweave"},
                ],
            }
        ],
    }

    key_pair = os.environ.get("KEY_PAIR_NAME", "").strip()
    if key_pair:
        run_kwargs["KeyName"] = key_pair

    # ── Launch instance (spot with on-demand fallback) ─────────────────────────
    response, market_type = _launch_instance(run_kwargs)
    instance_id: str = response["Instances"][0]["InstanceId"]

    logger.info(
        "EC2 instance launched | market=%s instance_id=%s job_id=%s",
        market_type, instance_id, job_id,
    )

    result = {
        "instance_id": instance_id,
        "job_id": job_id,
        "effective_model": effective_model,
        "dataset_name": dataset_name,
        "instance_type": instance_type,
        "market_type": market_type,
        "artifacts_prefix": f"s3://{os.environ['ARTIFACTS_BUCKET']}/adapters/{job_id}/",
    }
    logger.info("Response: %s", json.dumps(result))
    return result
