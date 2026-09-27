#!/usr/bin/env python3
"""
Did training on the preferences make the model more reliable?

A lower training loss says the adapter fits its data; it says nothing about
whether its answers got better. This compares the adapter with its base model on
held-out prompts, the same way the labels were audited -- a judge in both orders
-- so "better" means the same thing before and after training.

  generate   (on a GPU box) answer each held-out prompt with the base model and
             with base + adapter; writes {id, prompt, context, base, adapter}
  score      judge adapter vs base per prompt (label_audit.judge_pair, adapter in
             the "chosen" slot): win rate over decisive verdicts with a 95%
             Wilson interval, plus, with --expect-json, the fraction of answers
             that parse as JSON -- the failure TeamWeave's structured steps
             repair after the fact

  python scripts/eval_reliability.py generate --base Qwen/Qwen2.5-0.5B-Instruct \\
      --adapter ./adapter --prompts heldout.jsonl --out gens.jsonl
  python scripts/eval_reliability.py score gens.jsonl --judge-model <id> --expect-json

The verdict is ``improved`` only when the interval's lower bound clears 0.5.
Held-out prompts must not overlap the training pairs; ``score`` refuses a
generations file whose prompts appear in --train-pairs when it is given.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Callable

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "orchestrator"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "training"))

import label_audit  # noqa: E402


def _parses_as_json(text: str) -> bool:
    t = (text or "").strip()
    if t.startswith("```"):
        t = t.strip("`")
        t = t[t.find("\n") + 1:] if "\n" in t else t
    try:
        json.loads(t)
        return True
    except ValueError:
        return False


def score(rows: list[dict], invoke: Callable[[str], str], expect_json: bool = False,
          train_prompts: set[str] | None = None) -> dict:
    if train_prompts:
        leaked = [r["id"] for r in rows if r.get("prompt") in train_prompts]
        if leaked:
            raise SystemExit(f"{len(leaked)} held-out prompts appear in the training pairs, e.g. {leaked[:3]}")
    counts = {k: 0 for k in ("adapter", "base", "tie", "inconsistent", "error")}
    for r in rows:
        v = label_audit.judge_pair({"prompt": r["prompt"], "context": r.get("context") or {},
                                    "chosen": r["adapter"], "rejected": r["base"]}, invoke)
        counts[{"chosen": "adapter", "rejected": "base"}.get(v, v)] += 1
    decisive = counts["adapter"] + counts["base"]
    lo, hi = label_audit.wilson(counts["adapter"], decisive)
    out = {
        "n": len(rows), "counts": counts, "decisive": decisive,
        "win_rate": round(counts["adapter"] / decisive, 4) if decisive else None,
        "win_rate_ci": [round(lo, 4), round(hi, 4)],
        "verdict": "improved" if decisive and lo > 0.5 else ("regressed" if decisive and hi < 0.5 else "no_evidence"),
    }
    if expect_json:
        out["json_valid"] = {
            "base": round(sum(_parses_as_json(r["base"]) for r in rows) / len(rows), 4) if rows else None,
            "adapter": round(sum(_parses_as_json(r["adapter"]) for r in rows) / len(rows), 4) if rows else None,
        }
    return out


def generate(base: str, adapter: str, prompts_path: str, out_path: str, max_new_tokens: int) -> None:
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from preference_format import alpaca_prompt, render_input

    tok = AutoTokenizer.from_pretrained(base)
    model = AutoModelForCausalLM.from_pretrained(base, torch_dtype=torch.bfloat16, device_map="auto")
    tuned = PeftModel.from_pretrained(model, adapter)

    def answer(m, text):
        ids = tok(text, return_tensors="pt").to(m.device)
        out = m.generate(**ids, max_new_tokens=max_new_tokens, do_sample=False)
        return tok.decode(out[0][ids["input_ids"].shape[1]:], skip_special_tokens=True).strip()

    with open(prompts_path) as fh, open(out_path, "w") as w:
        for i, line in enumerate(fh):
            if not line.strip():
                continue
            row = json.loads(line)
            text = alpaca_prompt(row["prompt"], render_input(row.get("context")))
            with tuned.disable_adapter():
                b = answer(tuned, text)
            a = answer(tuned, text)
            w.write(json.dumps({"id": row.get("id", i), "prompt": row["prompt"], "context": row.get("context"),
                                "base": b, "adapter": a}) + "\n")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("generate")
    g.add_argument("--base", required=True)
    g.add_argument("--adapter", required=True)
    g.add_argument("--prompts", required=True)
    g.add_argument("--out", required=True)
    g.add_argument("--max-new-tokens", type=int, default=512)
    s = sub.add_parser("score")
    s.add_argument("generations")
    s.add_argument("--judge-model", required=True)
    s.add_argument("--region", default="us-east-1")
    s.add_argument("--expect-json", action="store_true")
    s.add_argument("--train-pairs", default=None)
    s.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    if args.cmd == "generate":
        generate(args.base, args.adapter, args.prompts, args.out, args.max_new_tokens)
        return 0
    rows = [json.loads(l) for l in Path(args.generations).read_text().splitlines() if l.strip()]
    train = None
    if args.train_pairs:
        train = {json.loads(l).get("prompt") for l in Path(args.train_pairs).read_text().splitlines() if l.strip()}
    result = score(rows, label_audit.bedrock_invoke(args.judge_model, region=args.region), args.expect_json, train)
    print(json.dumps(result, indent=2))
    if args.out:
        Path(args.out).write_text(json.dumps(result, indent=2))
    return 0 if result["verdict"] == "improved" else 3


if __name__ == "__main__":
    raise SystemExit(main())
