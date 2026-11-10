#!/usr/bin/env python3
"""Convert dpo-v1 JSON records on disk into TrainWeave dataset files (offline).

Reads every ``*.json`` file under ``--input`` (file or directory) and writes
``train.jsonl`` (Alpaca SFT) and ``train.dpo.jsonl`` (preference pairs) to
``--output``. No AWS credentials required.

This mirrors what the orchestrator Lambda does in ``_materialise_dpo_dataset``
before uploading to S3.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "src" / "orchestrator"))

import dpo_dataset  # noqa: E402


def _iter_json_paths(root: Path) -> list[Path]:
    if root.is_file():
        return [root] if root.suffix == ".json" else []
    return sorted(p for p in root.rglob("*.json") if p.is_file())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Path to a dpo-v1 .json file or a directory tree of *.json records",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Directory to write train.jsonl and train.dpo.jsonl",
    )
    args = parser.parse_args()

    paths = _iter_json_paths(args.input)
    if not paths:
        print(f"No JSON records found under {args.input}", file=sys.stderr)
        return 1

    conversion = dpo_dataset.convert_records(dpo_dataset.load_records_from_paths(paths))
    if not conversion.sft_rows:
        print(
            f"All {len(paths)} record(s) were unusable (missing prompt or chosen response)",
            file=sys.stderr,
        )
        return 1

    args.output.mkdir(parents=True, exist_ok=True)
    sft_path = args.output / dpo_dataset.SFT_DATASET_FILENAME
    dpo_path = args.output / dpo_dataset.DPO_DATASET_FILENAME
    sft_path.write_text(dpo_dataset.to_jsonl(conversion.sft_rows), encoding="utf-8")
    dpo_path.write_text(dpo_dataset.to_jsonl(conversion.dpo_rows), encoding="utf-8")

    print(
        f"Wrote {len(conversion.sft_rows)} SFT rows -> {sft_path}\n"
        f"Wrote {len(conversion.dpo_rows)} DPO pairs -> {dpo_path}\n"
        f"Skipped {conversion.skipped} record(s)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
