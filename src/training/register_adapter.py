"""
Register a trained LoRA adapter in DeployWeave's adapter catalogue.

Runs on the training instance, from bootstrap.sh, immediately after the
adapter has been uploaded to S3 — registering earlier would advertise an
adapter that may never exist.

The item written here is exactly the shape DeployWeave's ``adapter_resolver``
reads (deployweave_mcp.py):

    adapter_id   S   partition key; ``get_adapter`` reads it directly
    base_model   S   filtered on by ``list_adapters`` and ``search_by_tags``,
                     and by ``_search_adapters_by_task`` for team_provisioner
    task_type    S   hash key of the ``task_type-index`` GSI
    s3_path      S   where the adapter weights live
    tags         SS  ``search_by_tags`` membership-tests the extra tags
    primary_tag  S   hash key of the ``primary_tag-index`` GSI (= tags[0])
    created_at   N   range key of both GSIs (unix epoch seconds)
    metadata     M   free-form; carries the dataset provenance

Registration is opt-in: when ``ADAPTER_CATALOG_TABLE`` is unset the script
logs why it is skipping and exits 0, so a TrainWeave deployment that does not
know about DeployWeave keeps working unchanged.

The adapter_id is derived from the job id rather than a fresh UUID, so
re-running a job upserts its catalogue entry instead of accumulating
duplicates that point at the same S3 prefix.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%SZ",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("trainweave.register_adapter")

# Base-model prefixes DeployWeave's lora_validator.py accepts. An adapter
# trained on any other base model is catalogued, but team_provisioner will
# refuse to attach it (ValueError from validate_lora_compatibility).
DEPLOYWEAVE_LORA_PREFIXES = ("amazon.titan-", "meta.llama")


def adapter_id_for_job(job_id: str) -> str:
    """Derive the catalogue key from the job id (stable across re-runs)."""
    return f"trainweave-{job_id}"


def build_catalog_item(
    *,
    job_id: str,
    base_model: str,
    adapter_s3_uri: str,
    dataset_name: str,
    dataset_source_type: str = "s3",
    dpo_source_bucket: str = "",
    dpo_source_prefix: str = "",
    dpo_record_count: int = 0,
    dpo_pairs_key: str = "",
    created_at: int | None = None,
) -> dict:
    """Build the DynamoDB item for one trained adapter.

    ``task_type`` is the TrainWeave dataset name: DeployWeave looks adapters
    up by the capability the agent needs, and the dataset is the closest thing
    TrainWeave knows about what the adapter was taught.

    ``primary_tag`` is the dataset name too, so ``search_by_tags`` finds the
    adapter by capability; ``trainweave`` and the dataset source type are
    carried as extra tags for provenance filtering.
    """
    if not job_id:
        raise ValueError("job_id is required")
    if not base_model:
        raise ValueError("base_model is required")
    if not adapter_s3_uri:
        raise ValueError("adapter_s3_uri is required")

    task_type = dataset_name or "general"
    now = int(created_at if created_at is not None else time.time())

    metadata = {
        "source": "trainweave",
        "job_id": job_id,
        "base_model": base_model,
        "s3_path": adapter_s3_uri,
        "dataset_name": task_type,
        "dataset_source_type": dataset_source_type,
        "created_at_iso": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(now)),
    }
    # Dataset provenance — only meaningful for DPO-derived datasets.
    if dataset_source_type == "dpo":
        metadata.update({
            "dpo_source_bucket": dpo_source_bucket,
            "dpo_source_prefix": dpo_source_prefix,
            "dpo_record_count": int(dpo_record_count or 0),
            "dpo_pairs_key": dpo_pairs_key,
        })

    tags = [task_type, "trainweave", f"source:{dataset_source_type}"]

    return {
        "adapter_id": adapter_id_for_job(job_id),
        "base_model": base_model,
        "task_type": task_type,
        "s3_path": adapter_s3_uri,
        "tags": set(tags),
        "primary_tag": tags[0],
        "created_at": now,
        "metadata": metadata,
    }


def register(table, item: dict) -> None:
    """Upsert the item. ``put_item`` without a condition, so a re-run replaces."""
    table.put_item(Item=item)


def _env_int(name: str) -> int:
    try:
        return int(os.environ.get(name, "0") or 0)
    except ValueError:
        return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Register a trained adapter in DeployWeave")
    parser.add_argument("--adapter-s3-uri", required=True, help="s3://bucket/adapters/<job_id>/")
    args = parser.parse_args(argv)

    table_name = os.environ.get("ADAPTER_CATALOG_TABLE", "").strip()
    if not table_name:
        logger.info(
            "ADAPTER_CATALOG_TABLE is unset — skipping DeployWeave adapter registration"
        )
        return 0

    base_model = os.environ.get("EFFECTIVE_MODEL", "")
    item = build_catalog_item(
        job_id=os.environ.get("JOB_ID", ""),
        base_model=base_model,
        adapter_s3_uri=args.adapter_s3_uri,
        dataset_name=os.environ.get("DATASET_NAME", ""),
        dataset_source_type=os.environ.get("DATASET_SOURCE_TYPE", "s3"),
        dpo_source_bucket=os.environ.get("DPO_SOURCE_BUCKET", ""),
        dpo_source_prefix=os.environ.get("DPO_SOURCE_PREFIX", ""),
        dpo_record_count=_env_int("DPO_RECORD_COUNT"),
        dpo_pairs_key=os.environ.get("DPO_PAIRS_KEY", ""),
    )

    if not base_model.startswith(DEPLOYWEAVE_LORA_PREFIXES):
        logger.warning(
            "Base model %s is outside DeployWeave's LoRA-compatible prefixes %s — the "
            "adapter is catalogued, but team_provisioner will decline to attach it",
            base_model, list(DEPLOYWEAVE_LORA_PREFIXES),
        )

    import boto3  # imported late so the pure builder is testable without boto3

    table = boto3.resource(
        "dynamodb", region_name=os.environ.get("AWS_DEFAULT_REGION", "us-east-1")
    ).Table(table_name)
    register(table, item)

    logger.info(
        "Adapter registered in %s | %s",
        table_name,
        json.dumps({k: v for k, v in item.items() if k != "tags"}, default=str),
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
