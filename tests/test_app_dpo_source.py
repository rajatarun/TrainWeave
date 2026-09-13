"""Tests for the dpo dataset_source branch of the orchestrator Lambda.

Covers the glue between dpo_dataset and the EC2 launch: which S3 objects are
written, and which of them the instance is pointed at.
"""

import json
import os

import pytest

os.environ.setdefault("AWS_DEFAULT_REGION", "us-east-1")

import app  # noqa: E402  - imported after the region default is in place

from test_dpo_dataset import make_record  # noqa: E402


class RecordingS3:
    """Serves DPO records and records every put_object call."""

    def __init__(self, records: dict[str, dict]):
        self.records = records
        self.puts: dict[str, bytes] = {}

    def get_paginator(self, name):
        outer = self

        class _Paginator:
            def paginate(self, Bucket, Prefix):  # noqa: N803
                yield {"Contents": [
                    {"Key": k} for k in sorted(outer.records) if k.startswith(Prefix)
                ]}

        return _Paginator()

    def get_object(self, Bucket, Key):  # noqa: N803
        payload = json.dumps(self.records[Key]).encode()

        class _Body:
            def read(self_inner):
                return payload

        return {"Body": _Body()}

    def put_object(self, Bucket, Key, Body, **kwargs):  # noqa: N803
        self.puts[Key] = Body


@pytest.fixture
def s3(monkeypatch):
    fake = RecordingS3({
        "tw/visibility/draft/run-1/dpo_1.json": make_record(run_id="run-1"),
        "tw/visibility/draft/run-2/dpo_2.json": make_record(run_id="run-2", rejected=""),
        "tw/other/draft/run-3/dpo_3.json": make_record(run_id="run-3"),
    })
    monkeypatch.setattr(app, "_s3", lambda: fake)
    monkeypatch.setenv("ARTIFACTS_BUCKET", "trainweave-artifacts")
    return fake


SOURCE = {"type": "dpo", "bucket": "teamweave-dpo", "prefix": "tw/visibility/draft"}


def test_writes_both_datasets_under_the_job_prefix(s3):
    bucket, key, provenance = app._materialise_dpo_dataset("job-1", SOURCE)

    assert bucket == "trainweave-artifacts"
    assert key == "datasets/job-1/train.jsonl"
    assert set(s3.puts) == {"datasets/job-1/train.jsonl", "datasets/job-1/train.dpo.jsonl"}
    assert provenance["dpo_pairs_key"] == "datasets/job-1/train.dpo.jsonl"


def test_the_instance_is_pointed_at_the_sft_file_not_the_pairs(s3):
    _bucket, key, _prov = app._materialise_dpo_dataset("job-1", SOURCE)
    # train.py is SFT-only; pointing it at the pairs file would silently
    # change what is being learned.
    assert key.endswith("train.jsonl")
    assert not key.endswith("train.dpo.jsonl")


def test_sft_and_pair_counts_differ_when_a_record_has_no_rejected_side(s3):
    app._materialise_dpo_dataset("job-1", SOURCE)
    sft = s3.puts["datasets/job-1/train.jsonl"].decode().splitlines()
    pairs = s3.puts["datasets/job-1/train.dpo.jsonl"].decode().splitlines()
    assert len(sft) == 2      # both records are trainable
    assert len(pairs) == 1    # only one carries a preference


def test_provenance_reports_the_source_and_the_trainable_record_count(s3):
    _b, _k, provenance = app._materialise_dpo_dataset("job-1", SOURCE)
    assert provenance["dpo_source_bucket"] == "teamweave-dpo"
    assert provenance["dpo_source_prefix"] == "tw/visibility/draft"
    assert provenance["dpo_record_count"] == 2


def test_records_outside_the_prefix_are_not_pulled_in(s3):
    app._materialise_dpo_dataset("job-1", SOURCE)
    rows = s3.puts["datasets/job-1/train.jsonl"].decode()
    assert "tw/other" not in rows


def test_an_empty_prefix_fails_loudly_instead_of_training_on_nothing(s3):
    with pytest.raises(ValueError, match="No DPO records found"):
        app._materialise_dpo_dataset("job-1", {**SOURCE, "prefix": "tw/nothing/here"})


@pytest.mark.parametrize("source", [
    {"type": "dpo", "bucket": "", "prefix": "p"},
    {"type": "dpo", "bucket": "b", "prefix": ""},
])
def test_bucket_and_prefix_are_both_required(s3, source):
    with pytest.raises(ValueError, match="requires both"):
        app._materialise_dpo_dataset("job-1", source)


def test_unknown_dataset_source_type_is_rejected():
    with pytest.raises(ValueError, match="Unsupported dataset_source type"):
        app.handler({"dataset_name": "x", "dataset_source": {"type": "rlhf"}}, None)


def test_dataset_bucket_and_key_stay_required_without_a_dataset_source():
    with pytest.raises(ValueError, match="dataset_bucket"):
        app.handler({"dataset_name": "x"}, None)
