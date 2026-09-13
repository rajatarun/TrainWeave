"""Tests for converting TeamWeave DPO records into TrainWeave datasets.

The sample records mirror, field for field, what TeamWeave's
``src/orchestrator/dpo_collector.py::_upload_dpo_record`` writes to S3
(``schema_version: "dpo-v1"``).
"""

import json

import pytest

import dpo_dataset


def make_record(**overrides) -> dict:
    """A dpo-v1 record exactly as TeamWeave's dpo_collector uploads it."""
    record = {
        "schema_version": "dpo-v1",
        "timestamp": "2026-09-13T10:11:12.131415",
        "project": "teamweave",
        "team": "visibility",
        "step_id": "draft",
        "run_id": "run-abc123",
        "prompt": "Draft a LinkedIn post about serverless cost control.",
        "context": {"tone": "practical", "audience": "platform engineers"},
        "chosen": "Serverless bills drop when idle compute disappears...",
        "rejected": "Serverless is revolutionary and will change everything!!!",
        "chosen_composite_score": 0.12,
        "rejected_composite_score": 0.71,
        "delta": 0.59,
        "metrics_a": {"composite_risk_score": 0.12, "prompt_tokens": 410},
        "metrics_b": {"composite_risk_score": 0.71, "prompt_tokens": 402},
    }
    record.update(overrides)
    return record


# ── SFT conversion ────────────────────────────────────────────────────────────

def test_sft_example_uses_chosen_response_as_output():
    row = dpo_dataset.sft_example(make_record())
    assert row["instruction"] == "Draft a LinkedIn post about serverless cost control."
    assert row["output"].startswith("Serverless bills drop")
    # The rejected response must never leak into an SFT target.
    assert "revolutionary" not in row["output"]


def test_sft_row_matches_the_alpaca_schema_train_py_accepts():
    row = dpo_dataset.sft_example(make_record())
    # train.py::load_jsonl_dataset dispatches on 'instruction' + 'output'.
    assert set(row) == {"instruction", "input", "output"}
    assert all(isinstance(v, str) for v in row.values())


def test_context_is_rendered_deterministically_into_input():
    row_a = dpo_dataset.sft_example(make_record())
    row_b = dpo_dataset.sft_example(
        make_record(context={"audience": "platform engineers", "tone": "practical"})
    )
    # Key order in the source record must not change the dataset bytes.
    assert row_a["input"] == row_b["input"]
    assert json.loads(row_a["input"])["tone"] == "practical"


def test_empty_context_becomes_empty_string():
    # train.py calls example.get("input", "").strip() — it must be a string.
    assert dpo_dataset.sft_example(make_record(context={}))["input"] == ""
    assert dpo_dataset.sft_example(make_record(context=None))["input"] == ""


@pytest.mark.parametrize("bad", [{"prompt": ""}, {"chosen": ""}, {"chosen": "   "}])
def test_records_without_prompt_or_chosen_are_unusable(bad):
    assert dpo_dataset.sft_example(make_record(**bad)) is None


# ── Preference pairs ──────────────────────────────────────────────────────────

def test_dpo_pair_keeps_both_sides():
    pair = dpo_dataset.dpo_pair(make_record())
    assert set(pair) == {"prompt", "chosen", "rejected"}
    assert pair["chosen"] != pair["rejected"]


def test_identical_responses_carry_no_preference_signal():
    assert dpo_dataset.dpo_pair(make_record(rejected=make_record()["chosen"])) is None


def test_missing_rejected_yields_no_pair_but_still_trains():
    record = make_record(rejected="")
    assert dpo_dataset.dpo_pair(record) is None
    assert dpo_dataset.sft_example(record) is not None


# ── Batch conversion ──────────────────────────────────────────────────────────

def test_convert_records_splits_sft_rows_from_pairs():
    records = [
        make_record(run_id="r1"),
        make_record(run_id="r2", rejected=""),        # SFT only
        make_record(run_id="r3", chosen=""),          # unusable
        "not-a-dict",                                  # unusable
    ]
    result = dpo_dataset.convert_records(records)
    assert len(result.sft_rows) == 2
    assert len(result.dpo_rows) == 1
    assert result.skipped == 2


def test_to_jsonl_emits_one_object_per_line():
    result = dpo_dataset.convert_records([make_record(), make_record(run_id="r2")])
    text = dpo_dataset.to_jsonl(result.sft_rows)
    lines = text.splitlines()
    assert len(lines) == 2
    assert text.endswith("\n")
    assert all(set(json.loads(line)) == {"instruction", "input", "output"} for line in lines)


def test_empty_input_produces_empty_jsonl():
    assert dpo_dataset.to_jsonl([]) == ""


# ── S3 access helpers ─────────────────────────────────────────────────────────

class FakeS3:
    """Minimal stand-in for the S3 client surface dpo_dataset uses."""

    def __init__(self, objects: dict[str, bytes]):
        self.objects = objects

    def get_paginator(self, name):
        assert name == "list_objects_v2"
        outer = self

        class _Paginator:
            def paginate(self, Bucket, Prefix):  # noqa: N803 - boto3 kwarg casing
                keys = [k for k in outer.objects if k.startswith(Prefix)]
                # Two pages, to exercise pagination.
                yield {"Contents": [{"Key": k} for k in keys[:1]]}
                yield {"Contents": [{"Key": k} for k in keys[1:]]}

        return _Paginator()

    def get_object(self, Bucket, Key):  # noqa: N803 - boto3 kwarg casing
        class _Body:
            def __init__(self, data):
                self._data = data

            def read(self):
                return self._data

        return {"Body": _Body(self.objects[Key])}


def test_list_record_keys_filters_and_sorts():
    s3 = FakeS3({
        "teamweave/visibility/draft/run-b/dpo_2.json": b"{}",
        "teamweave/visibility/draft/run-a/dpo_1.json": b"{}",
        "teamweave/visibility/draft/run-a/notes.txt": b"x",
        "teamweave/visibility/edit/run-c/dpo_3.json": b"{}",
    })
    keys = dpo_dataset.list_record_keys(s3, "bucket", "teamweave/visibility/draft")
    assert keys == [
        "teamweave/visibility/draft/run-a/dpo_1.json",
        "teamweave/visibility/draft/run-b/dpo_2.json",
    ]


def test_load_records_skips_unparseable_objects():
    s3 = FakeS3({
        "p/a.json": json.dumps(make_record()).encode(),
        "p/b.json": b"{not json",
    })
    records = dpo_dataset.load_records(s3, "bucket", ["p/a.json", "p/b.json"])
    assert len(records) == 1
    assert records[0]["schema_version"] == "dpo-v1"
