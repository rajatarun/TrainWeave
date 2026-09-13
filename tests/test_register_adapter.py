"""Tests for the DeployWeave adapter-catalogue item builder.

The expected item shape is taken from DeployWeave's ``adapter_resolver``
(deployweave_mcp.py): ``adapter_id`` is the table's partition key,
``task_type`` and ``primary_tag`` are GSI hash keys with ``created_at`` as
their range key, and ``tags`` is membership-tested as a set.
"""

import pytest

import register_adapter


def build(**overrides) -> dict:
    kwargs = {
        "job_id": "trainweave-visibility-20260913-101112",
        "base_model": "meta.llama3-8b-instruct-v1:0",
        "adapter_s3_uri": "s3://trainweave-artifacts/adapters/trainweave-visibility-20260913-101112/",
        "dataset_name": "visibility-draft",
        "dataset_source_type": "dpo",
        "dpo_source_bucket": "teamweave-dpo-training",
        "dpo_source_prefix": "teamweave/visibility/draft",
        "dpo_record_count": 42,
        "dpo_pairs_key": "datasets/trainweave-visibility-20260913-101112/train.dpo.jsonl",
        "created_at": 1789000000,
    }
    kwargs.update(overrides)
    return register_adapter.build_catalog_item(**kwargs)


def test_item_has_exactly_the_attributes_adapter_resolver_reads():
    item = build()
    assert set(item) == {
        "adapter_id", "base_model", "task_type", "s3_path",
        "tags", "primary_tag", "created_at", "metadata",
    }


def test_key_types_match_the_table_schema():
    item = build()
    # adapter_id (HASH, S); created_at is the GSI range key and must be numeric.
    assert isinstance(item["adapter_id"], str)
    assert isinstance(item["created_at"], int)
    assert isinstance(item["task_type"], str)
    assert isinstance(item["primary_tag"], str)
    assert isinstance(item["s3_path"], str)


def test_tags_is_a_set_with_primary_tag_first_element():
    item = build()
    # search_by_tags does `extra_tag in (i.get("tags") or set())`.
    assert isinstance(item["tags"], set)
    assert item["primary_tag"] in item["tags"]
    assert item["primary_tag"] == item["task_type"]
    assert "trainweave" in item["tags"]
    # An empty string set is not a valid DynamoDB value.
    assert item["tags"]


def test_task_type_is_the_dataset_name_so_search_by_task_finds_it():
    assert build()["task_type"] == "visibility-draft"
    assert build(dataset_name="")["task_type"] == "general"


def test_adapter_id_is_derived_from_job_id_so_reruns_upsert():
    first = build()
    second = build(created_at=1789009999)
    assert first["adapter_id"] == second["adapter_id"]
    assert first["adapter_id"] == "trainweave-trainweave-visibility-20260913-101112"


def test_dataset_provenance_is_carried_in_metadata():
    meta = build()["metadata"]
    assert meta["dpo_source_bucket"] == "teamweave-dpo-training"
    assert meta["dpo_source_prefix"] == "teamweave/visibility/draft"
    assert meta["dpo_record_count"] == 42
    assert meta["dpo_pairs_key"].endswith("train.dpo.jsonl")
    assert meta["source"] == "trainweave"
    assert meta["job_id"] == "trainweave-visibility-20260913-101112"


def test_created_timestamp_is_recorded_twice_epoch_and_iso():
    item = build()
    assert item["created_at"] == 1789000000
    assert item["metadata"]["created_at_iso"] == "2026-09-10T00:26:40Z"


def test_non_dpo_datasets_carry_no_dpo_provenance():
    meta = build(dataset_source_type="s3")["metadata"]
    assert "dpo_source_bucket" not in meta
    assert meta["dataset_source_type"] == "s3"
    assert "source:s3" in build(dataset_source_type="s3")["tags"]


def test_base_model_and_s3_uri_are_surfaced_at_top_level():
    item = build()
    assert item["base_model"] == "meta.llama3-8b-instruct-v1:0"
    assert item["s3_path"].startswith("s3://")


@pytest.mark.parametrize("missing", ["job_id", "base_model", "adapter_s3_uri"])
def test_required_fields_are_enforced(missing):
    with pytest.raises(ValueError, match=missing):
        build(**{missing: ""})


def test_register_upserts_without_a_condition_expression():
    calls = []

    class FakeTable:
        def put_item(self, **kwargs):
            calls.append(kwargs)

    item = build()
    register_adapter.register(FakeTable(), item)
    assert calls == [{"Item": item}]
    # A ConditionExpression would make a re-run of the same job fail.
    assert "ConditionExpression" not in calls[0]


def test_lora_compatible_prefixes_match_deployweave_validator():
    # DeployWeave's lora_validator.LORA_COMPATIBLE_MODELS.
    assert register_adapter.DEPLOYWEAVE_LORA_PREFIXES == ("amazon.titan-", "meta.llama")
