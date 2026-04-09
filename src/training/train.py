"""
TrainWeave — LoRA fine-tuning script (QLoRA, 4-bit).

Runs on an EC2 Spot GPU instance. Called by bootstrap.sh with CLI args.

Key design choices:
- 4-bit QLoRA (BitsAndBytesConfig) to fit large models on a T4/A10G GPU
- PEFT LoRA targeting attention projection layers
- SFTTrainer (TRL) for instruction-tuning with automatic tokenization
- S3CheckpointCallback periodically syncs checkpoints so they survive
  spot interruption between explicit save_steps
- Final adapter upload is handled by bootstrap.sh for clarity, but the
  callback ensures partial progress is never lost
"""

import argparse
import json
import logging
import os
import subprocess
import sys

import torch
from datasets import Dataset, load_dataset
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainerCallback,
    TrainerControl,
    TrainerState,
    TrainingArguments,
)
from trl import SFTTrainer, SFTConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%SZ",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("trainweave.train")


# ── S3 checkpoint callback ─────────────────────────────────────────────────────

class S3CheckpointCallback(TrainerCallback):
    """
    Syncs the output directory to S3 every `sync_steps` training steps.

    Uses `aws s3 sync` (already authenticated via EC2 instance profile) so
    checkpoints survive spot interruption. The S3 VPC gateway endpoint ensures
    this traffic never touches the NAT instance — zero data-transfer cost.
    """

    def __init__(self, artifacts_bucket: str, job_id: str, output_dir: str, sync_steps: int = 100):
        self.s3_prefix = f"s3://{artifacts_bucket}/checkpoints/{job_id}/"
        self.output_dir = output_dir
        self.sync_steps = sync_steps
        self._last_synced_step = 0

    def on_step_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ) -> None:
        step = state.global_step
        if step > 0 and step % self.sync_steps == 0 and step != self._last_synced_step:
            self._sync(step)
            self._last_synced_step = step

    def on_train_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ) -> None:
        # Final sync at end of training
        self._sync(state.global_step, label="final")

    def _sync(self, step: int, label: str = "") -> None:
        tag = f"step-{step}" + (f"-{label}" if label else "")
        logger.info("Syncing checkpoints to %s (%s)", self.s3_prefix, tag)
        result = subprocess.run(
            ["aws", "s3", "sync", self.output_dir, self.s3_prefix, "--quiet"],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            # Log but don't raise — a sync failure shouldn't abort training
            logger.warning("S3 sync warning: %s", result.stderr.strip())
        else:
            logger.info("Checkpoint sync complete (%s)", tag)


# ── Dataset helpers ────────────────────────────────────────────────────────────

ALPACA_TEMPLATE = (
    "Below is an instruction that describes a task"
    "{input_section}. "
    "Write a response that appropriately completes the request.\n\n"
    "### Instruction:\n{instruction}\n\n"
    "{input_part}"
    "### Response:\n{output}"
)


def _format_alpaca(example: dict) -> dict:
    """Convert Alpaca-style {instruction, input, output} record to a text string."""
    has_input = bool(example.get("input", "").strip())
    text = ALPACA_TEMPLATE.format(
        input_section=", using the input below as a context" if has_input else "",
        instruction=example.get("instruction", ""),
        input_part=f"### Input:\n{example['input']}\n\n" if has_input else "",
        output=example.get("output", ""),
    )
    return {"text": text}


def load_jsonl_dataset(path: str) -> Dataset:
    """
    Load a JSONL file and normalise to {"text": "..."} format.

    Supported input schemas:
      1. {"text": "..."}                            — used as-is
      2. {"instruction": ..., "input": ..., "output": ...}  — Alpaca format
      3. {"messages": [...]}                        — chat format, joined to text
    """
    raw = load_dataset("json", data_files=path, split="train")

    if "text" in raw.column_names:
        logger.info("Dataset schema: text field (pass-through)")
        return raw.select_columns(["text"])

    if "instruction" in raw.column_names and "output" in raw.column_names:
        logger.info("Dataset schema: Alpaca (instruction/input/output → text)")
        return raw.map(_format_alpaca, remove_columns=raw.column_names)

    if "messages" in raw.column_names:
        logger.info("Dataset schema: chat messages → joined text")

        def _join_messages(example: dict) -> dict:
            parts = []
            for msg in example.get("messages", []):
                role = msg.get("role", "user").capitalize()
                content = msg.get("content", "")
                parts.append(f"{role}: {content}")
            return {"text": "\n".join(parts)}

        return raw.map(_join_messages, remove_columns=raw.column_names)

    raise ValueError(
        f"Unrecognised dataset schema. Columns found: {raw.column_names}. "
        "Expected one of: 'text', 'instruction'+'output', or 'messages'."
    )


# ── Model loading ──────────────────────────────────────────────────────────────

def load_model_and_tokenizer(model_id: str):
    """
    Load model with 4-bit quantization (QLoRA).

    BitsAndBytesConfig with nf4 is the standard QLoRA recipe:
    - Loads weights in 4-bit NormalFloat (nf4) format
    - Computes in bfloat16
    - Lets a 7B+ model fit on a T4 (16GB) with room for activations/optimizer

    For smaller models (<3B) you may skip quantization, but keeping it on
    is harmless and makes the code portable across GPU sizes.
    """
    logger.info("Loading model: %s", model_id)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,  # extra quantization of quant constants
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto",          # distributes across available GPUs/CPU
        trust_remote_code=True,     # needed for some custom architectures
        dtype=torch.bfloat16,
    )

    # Required before adding LoRA layers: re-casts layer norms, enables
    # gradient checkpointing, and freezes quantized weights.
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

    # Many causal LM tokenizers lack a pad token; use eos as a safe fallback.
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    logger.info(
        "Model loaded | params=%dM | vocab=%d",
        sum(p.numel() for p in model.parameters()) // 1_000_000,
        len(tokenizer),
    )
    return model, tokenizer


def apply_lora(model, r: int, lora_alpha: int, lora_dropout: float):
    """
    Wrap the model with PEFT LoRA adapters targeting attention projections.

    Only q/k/v/o projection matrices are trained — the standard QLoRA recipe.
    Gate/up/down projections in MLP layers can be added to r for higher
    expressiveness at the cost of more trainable parameters.
    """
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        bias="none",
        inference_mode=False,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model


# ── Training ───────────────────────────────────────────────────────────────────

def train(args: argparse.Namespace) -> None:
    os.makedirs(args.output_dir, exist_ok=True)

    # Save run config alongside the adapter for reproducibility
    config_path = os.path.join(args.output_dir, "trainweave_run_config.json")
    with open(config_path, "w") as f:
        json.dump(vars(args), f, indent=2)

    # ── Load data ────────────────────────────────────────────────────────────
    dataset = load_jsonl_dataset(args.dataset_path)
    logger.info("Loaded %d training examples", len(dataset))

    # ── Load model + tokenizer ───────────────────────────────────────────────
    model, tokenizer = load_model_and_tokenizer(args.model_id)

    # Cap sequence length on the tokenizer — works across all TRL versions.
    # max_seq_length was removed from SFTConfig in TRL ≥ 0.10; setting it
    # here ensures SFTTrainer truncates correctly regardless of installed version.
    tokenizer.model_max_length = args.max_seq_len

    # ── Apply LoRA ───────────────────────────────────────────────────────────
    model = apply_lora(model, r=args.r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout)

    # ── Training arguments ───────────────────────────────────────────────────
    # warmup_ratio was removed in transformers 5.2; compute warmup_steps (int)
    # instead — compatible with all transformers versions.
    steps_per_epoch = max(1, len(dataset) // (args.batch_size * args.grad_accum))
    total_steps = steps_per_epoch * args.epochs
    warmup_steps = max(1, int(0.03 * total_steps))

    training_args = SFTConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_steps=warmup_steps,
        fp16=False,
        bf16=True,              # bfloat16 is more numerically stable than fp16
        optim="paged_adamw_8bit",  # 8-bit AdamW from bitsandbytes — lower VRAM
        logging_steps=10,
        save_strategy="steps",
        save_steps=args.checkpoint_steps,
        save_total_limit=3,     # keep last 3 checkpoints to cap disk usage
        dataset_text_field="text",
        packing=False,          # disable packing for simplicity; enable for speed
        report_to="none",       # no WandB/MLflow — keeps dependencies minimal
        run_name=args.job_id,
    )

    # ── Callbacks ────────────────────────────────────────────────────────────
    callbacks = [
        S3CheckpointCallback(
            artifacts_bucket=args.artifacts_bucket,
            job_id=args.job_id,
            output_dir=args.output_dir,
            sync_steps=args.checkpoint_steps,
        )
    ]

    # ── Trainer ──────────────────────────────────────────────────────────────
    # TRL ≥ 0.10 renamed `tokenizer` → `processing_class`; use the new name.
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
        callbacks=callbacks,
    )

    logger.info(
        "Starting training | epochs=%d batch=%d grad_accum=%d effective_batch=%d",
        args.epochs,
        args.batch_size,
        args.grad_accum,
        args.batch_size * args.grad_accum,
    )

    trainer.train()

    # ── Save final adapter ────────────────────────────────────────────────────
    # Saves only the LoRA adapter weights (not the full base model).
    # bootstrap.sh then syncs this to S3.
    logger.info("Saving adapter to %s", args.output_dir)
    trainer.model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    logger.info("Adapter saved.")


# ── Entry point ────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="TrainWeave LoRA fine-tuning",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Required
    p.add_argument("--model-id", required=True, help="HuggingFace model ID or local path")
    p.add_argument("--dataset-path", required=True, help="Local path to JSONL dataset")
    p.add_argument("--output-dir", required=True, help="Local directory for adapter output")
    p.add_argument("--job-id", required=True, help="Unique job identifier (used in S3 paths)")
    p.add_argument("--artifacts-bucket", required=True, help="S3 bucket for checkpoint sync")

    # LoRA hyperparameters
    p.add_argument("--r", type=int, default=16, help="LoRA rank")
    p.add_argument("--lora-alpha", type=int, default=32, help="LoRA alpha (scaling factor)")
    p.add_argument("--lora-dropout", type=float, default=0.05, help="LoRA dropout rate")

    # Training hyperparameters
    p.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    p.add_argument("--batch-size", type=int, default=4, help="Per-device train batch size")
    p.add_argument("--grad-accum", type=int, default=4, help="Gradient accumulation steps")
    p.add_argument("--max-seq-len", type=int, default=512, help="Max token sequence length")
    p.add_argument("--lr", type=float, default=2e-4, help="Peak learning rate")
    p.add_argument(
        "--checkpoint-steps",
        type=int,
        default=100,
        help="Save checkpoint + S3 sync every N steps",
    )

    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    logger.info("TrainWeave training started | job_id=%s | model=%s", args.job_id, args.model_id)
    train(args)
    logger.info("TrainWeave training finished | job_id=%s", args.job_id)
