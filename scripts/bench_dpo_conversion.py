#!/usr/bin/env python3
"""Synthetic micro-benchmark for DPO record conversion (offline, CI-safe).

Reports rows/sec over in-memory dpo-v1 records. Results are illustrative only;
not a production SLA.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "src" / "orchestrator"))

import dpo_dataset  # noqa: E402


def _synthetic_record(i: int) -> dict:
    return {
        "schema_version": "dpo-v1",
        "prompt": f"Question {i}?",
        "context": {"idx": i},
        "chosen": f"Good answer {i}",
        "rejected": f"Bad answer {i}",
    }


def main() -> int:
    n = 10_000
    records = [_synthetic_record(i) for i in range(n)]
    t0 = time.perf_counter()
    conversion = dpo_dataset.convert_records(records)
    elapsed = time.perf_counter() - t0
    rows_per_sec = len(conversion.sft_rows) / elapsed if elapsed > 0 else 0.0
    print(
        json.dumps(
            {
                "synthetic": True,
                "input_records": n,
                "sft_rows": len(conversion.sft_rows),
                "dpo_pairs": len(conversion.dpo_rows),
                "skipped": conversion.skipped,
                "elapsed_sec": round(elapsed, 4),
                "sft_rows_per_sec": round(rows_per_sec, 1),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
