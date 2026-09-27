"""Heuristic preference labels are checked against a judge before DPO trains on them.

The judge is a fake with a known opinion, so what is under test is the audit's
arithmetic and its refusals: position bias is caught rather than counted, the
decision uses the interval's lower bound, and the orchestrator will not start a
DPO job without a passing audit of the same data.
"""
from __future__ import annotations

import importlib.util
import io
import json
import os
from pathlib import Path

import pytest

os.environ.setdefault("AWS_DEFAULT_REGION", "us-east-1")

import label_audit as L  # noqa: E402
from preference_format import ALPACA_TEMPLATE, alpaca_prompt, alpaca_text, dpo_example  # noqa: E402
from test_dpo_dataset import make_record  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def judge_prefers(pred):
    """A judge that prefers whichever response satisfies pred, in either position."""
    def invoke(prompt):
        r1 = prompt.split("RESPONSE 1:\n", 1)[1].split("\n\nRESPONSE 2:", 1)[0]
        r2 = prompt.split("RESPONSE 2:\n", 1)[1].split("\n\nReply with JSON", 1)[0]
        a, b = pred(r1), pred(r2)
        return json.dumps({"better": "1" if a and not b else "2" if b and not a else "tie"})
    return invoke


def always_first(prompt):
    return '{"better": "1"}'


def records(n, good_chosen, delta=0.5):
    out = []
    for i in range(n):
        good, bad = f"GOOD answer {i}", f"bad answer {i}"
        chosen, rejected = (good, bad) if i < good_chosen else (bad, good)
        out.append(make_record(prompt=f"task {i}", chosen=chosen, rejected=rejected, delta=delta))
    return out


def test_judge_pair_maps_both_orders_back_to_chosen_or_rejected():
    judge = judge_prefers(lambda r: r.startswith("GOOD"))
    assert L.judge_pair(make_record(chosen="GOOD x", rejected="bad y"), judge) == "chosen"
    assert L.judge_pair(make_record(chosen="bad y", rejected="GOOD x"), judge) == "rejected"


def test_a_position_biased_judge_is_inconsistent_not_agreement():
    assert L.judge_pair(make_record(), always_first) == "inconsistent"
    result = L.audit(records(40, 40), always_first)
    assert result.decisive == 0 and result.decision == "insufficient"
    assert result.position_consistency == 0.0


def test_labels_hold_only_when_the_lower_bound_clears_the_bar():
    judge = judge_prefers(lambda r: r.startswith("GOOD"))
    assert L.audit(records(100, 95), judge).decision == "labels_hold"
    weak = L.audit(records(100, 72), judge)       # 72% agreement: lower bound under 0.7
    assert weak.decision == "labels_do_not_hold" and weak.agreement == 0.72
    assert L.audit(records(10, 10), judge).decision == "insufficient"


def test_verified_pairs_are_only_the_agreed_ones():
    judge = judge_prefers(lambda r: r.startswith("GOOD"))
    recs = records(40, 30)
    result = L.audit(recs, judge)
    pairs = L.verified_pairs(recs, result)
    assert len(pairs) == 30 and all(p["chosen"].startswith("GOOD") for p in pairs)


def test_failed_or_unreadable_judge_calls_are_errors_not_verdicts():
    def boom(prompt):
        raise RuntimeError("throttled")
    assert L.judge_pair(make_record(), boom) == "error"
    assert L.judge_pair(make_record(), lambda p: "I think the first") == "error"
    assert L.parse_verdict('```json\n{"better": "2"}\n```') == "2"


def test_dpo_prompt_is_the_sft_text_without_the_response():
    assert alpaca_text("do x", "ctx", "ANSWER") == alpaca_prompt("do x", "ctx") + "ANSWER"
    assert "### Response:\n{output}" in ALPACA_TEMPLATE
    ex = dpo_example({"prompt": "do x", "context": {"b": 1, "a": 2}, "chosen": "c", "rejected": "r"})
    assert ex["prompt"].endswith("### Response:\n") and '{"a": 2, "b": 1}' in ex["prompt"]
    assert dpo_example({"prompt": "p", "chosen": "same", "rejected": "same"}) is None


def test_train_py_uses_the_shared_template_and_bootstrap_ships_it():
    src = (ROOT / "src/training/train.py").read_text()
    assert "from preference_format import ALPACA_TEMPLATE" in src and "ALPACA_TEMPLATE = (" not in src
    boot = (ROOT / "src/training/bootstrap.sh").read_text()
    assert "training/preference_format.py" in boot, "train.py would die at import on the instance"
    assert '--objective       "${TRAIN_OBJECTIVE:-sft}"' in boot


# ── orchestrator gate ────────────────────────────────────────────────────────

class FakeS3:
    def __init__(self, objects):
        self.objects = dict(objects)
        self.puts = {}

    def get_object(self, Bucket, Key):  # noqa: N803
        return {"Body": io.BytesIO(self.objects[(Bucket, Key)])}

    def put_object(self, Bucket, Key, Body, ContentType=None):  # noqa: N803
        self.puts[(Bucket, Key)] = Body


@pytest.fixture
def app_env(monkeypatch):
    import app
    monkeypatch.setenv("ARTIFACTS_BUCKET", "artifacts")
    return app


def _audit(decision="labels_hold", prefix="teamweave/visibility/draft"):
    return json.dumps({"decision": decision, "reason": "r", "source_bucket": "dpo", "source_prefix": prefix,
                       "verified_pairs_key": "audits/1/pairs.verified.jsonl", "agreement": 0.9,
                       "agreement_ci": [0.85, 0.94]}).encode()


SOURCE = {"type": "dpo", "bucket": "dpo", "prefix": "teamweave/visibility/draft",
          "audit": {"key": "audits/1/audit.json"}}


def test_dpo_job_trains_on_the_audited_pairs(app_env, monkeypatch):
    s3 = FakeS3({("dpo", "audits/1/audit.json"): _audit(),
                 ("dpo", "audits/1/pairs.verified.jsonl"): b'{"prompt":"p","chosen":"c","rejected":"r"}\n'})
    monkeypatch.setattr(app_env, "_s3", lambda: s3)
    bucket, key, prov = app_env._resolve_audited_pairs("job1", SOURCE)
    assert (bucket, key) == ("artifacts", "datasets/job1/train.dpo.jsonl")
    assert s3.puts[(bucket, key)].startswith(b'{"prompt"') and prov["dpo_record_count"] == 1


@pytest.mark.parametrize("objects,match", [
    ({("dpo", "audits/1/audit.json"): _audit("labels_do_not_hold")}, "did not pass"),
    ({("dpo", "audits/1/audit.json"): _audit(prefix="teamweave/other/step")}, "covers"),
])
def test_dpo_job_is_refused_on_a_failed_or_foreign_audit(app_env, monkeypatch, objects, match):
    monkeypatch.setattr(app_env, "_s3", lambda: FakeS3(objects))
    with pytest.raises(ValueError, match=match):
        app_env._resolve_audited_pairs("job1", SOURCE)


def test_dpo_objective_without_an_audit_is_refused_before_anything_launches(app_env):
    with pytest.raises(ValueError, match="audit"):
        app_env._resolve_audited_pairs("job1", {k: v for k, v in SOURCE.items() if k != "audit"})
    with pytest.raises(ValueError, match="dataset_source"):
        app_env.handler({"dataset_name": "x", "dataset_bucket": "b", "dataset_key": "k", "objective": "dpo"}, None)
    with pytest.raises(ValueError, match="objective"):
        app_env.handler({"dataset_name": "x", "dataset_bucket": "b", "dataset_key": "k", "objective": "rlhf"}, None)


# ── reliability scoring ──────────────────────────────────────────────────────

def _load_eval():
    spec = importlib.util.spec_from_file_location("eval_reliability", ROOT / "scripts" / "eval_reliability.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_reliability_verdict_needs_the_interval_above_half():
    E = _load_eval()
    judge = judge_prefers(lambda r: r.startswith("{"))
    rows = [{"id": i, "prompt": f"p{i}", "base": "not json", "adapter": '{"ok": 1}'} for i in range(40)]
    out = E.score(rows, judge, expect_json=True)
    assert out["verdict"] == "improved" and out["json_valid"] == {"base": 0.0, "adapter": 1.0}
    mixed = [{"id": i, "prompt": f"p{i}", "base": '{"a":1}' if i % 2 else "x", "adapter": '{"a":1}' if i % 2 == 0 else "x"}
             for i in range(40)]
    assert E.score(mixed, judge)["verdict"] == "no_evidence"
    with pytest.raises(SystemExit):
        E.score(rows, judge, train_prompts={"p3"})
