"""Tests for offline DPO conversion."""

import json
import subprocess
import sys
from pathlib import Path

import dpo_dataset


def test_load_records_from_paths_reads_json(tmp_path: Path):
    p = tmp_path / "r.json"
    p.write_text(json.dumps({"prompt": "Hi", "chosen": "There", "context": {}}))
    records = dpo_dataset.load_records_from_paths([p])
    assert len(records) == 1
    assert records[0]["prompt"] == "Hi"


def test_convert_dpo_local_cli(tmp_path: Path):
    root = Path(__file__).resolve().parents[1]
    script = root / "scripts" / "convert_dpo_local.py"
    input_dir = root / "examples" / "dpo-records"
    out = tmp_path / "out"
    proc = subprocess.run(
        [sys.executable, str(script), "--input", str(input_dir), "--output", str(out)],
        capture_output=True,
        text=True,
        cwd=root,
    )
    assert proc.returncode == 0, proc.stderr
    sft = (out / dpo_dataset.SFT_DATASET_FILENAME).read_text(encoding="utf-8")
    assert "Spot instances" in sft
