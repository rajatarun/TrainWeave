"""
Prompt formatting shared by the SFT and DPO objectives, free of training deps.

``train.py`` imports torch, transformers and TRL at module scope, so nothing in
it can be unit-tested off the GPU instance. What *must* be tested is that a DPO
prompt is the SFT prompt with the response removed -- a model trained with SFT
on one template and preferences on another learns two formats and follows
neither -- so the template lives here, and both objectives build from it.

bootstrap.sh pulls this file next to train.py.
"""
from __future__ import annotations

import json
from typing import Any

ALPACA_TEMPLATE = (
    "Below is an instruction that describes a task"
    "{input_section}. "
    "Write a response that appropriately completes the request.\n\n"
    "### Instruction:\n{instruction}\n\n"
    "{input_part}"
    "### Response:\n{output}"
)


def render_input(context: Any) -> str:
    """The Alpaca ``input`` for a record's context -- same rule as dpo_dataset.render_context."""
    if isinstance(context, str):
        return context.strip()
    if not isinstance(context, dict) or not context:
        return ""
    return json.dumps(context, ensure_ascii=False, sort_keys=True, default=str)


def alpaca_text(instruction: str, input_text: str, output: str) -> str:
    has_input = bool((input_text or "").strip())
    return ALPACA_TEMPLATE.format(
        input_section=", using the input below as a context" if has_input else "",
        instruction=instruction,
        input_part=f"### Input:\n{input_text}\n\n" if has_input else "",
        output=output,
    )


def alpaca_prompt(instruction: str, input_text: str) -> str:
    """The SFT text up to and including ``### Response:\\n`` -- what DPO conditions on."""
    return alpaca_text(instruction, input_text, "")


def dpo_example(row: dict) -> dict | None:
    """``{prompt, chosen, rejected}`` in the training format, or None when unusable."""
    instruction = row.get("prompt")
    chosen, rejected = row.get("chosen"), row.get("rejected")
    if not all(isinstance(x, str) and x.strip() for x in (instruction, chosen, rejected)):
        return None
    if chosen.strip() == rejected.strip():
        return None
    return {
        "prompt": alpaca_prompt(instruction.strip(), render_input(row.get("context"))),
        "chosen": chosen.strip(),
        "rejected": rejected.strip(),
    }


def load_preference_rows(path: str) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                ex = dpo_example(json.loads(line))
                if ex is not None:
                    rows.append(ex)
    return rows
