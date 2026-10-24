#!/usr/bin/env python3
"""Print actual recorded run status once, for human review and real terminal screenshots."""

import argparse
import json
from pathlib import Path

from minimind_lab.training.evidence import utc_now, validate_run_id

ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser()
parser.add_argument("--run-id", required=True)
args = parser.parse_args()
run = ROOT / "artifacts/runs" / validate_run_id(args.run_id)
manifest = json.loads((run / "manifest.json").read_text())
print("MiniMind formal experiment — recorded evidence review")
print("Reviewed at UTC:", utc_now())
print("Run:", manifest["run_id"])
print("Source commit:", manifest["code_commit"])
print("GPU:", manifest["runtime"]["gpu"])
print("Pinned data files:", len(manifest["data_files"]))
events = [json.loads(line) for line in (run / "events.jsonl").read_text().splitlines()]
print("Latest event:", json.dumps(events[-1], ensure_ascii=False))
for path in sorted((run / "metrics").glob("*.jsonl")):
    rows = path.read_text().splitlines()
    record = json.loads(rows[-1])
    fields = (
        "step",
        "loss",
        "grad_norm",
        "learning_rate",
        "training_seconds",
        "padded_tokens_per_second",
        "cuda_peak_allocated_bytes",
        "resumed_from_step",
    )
    print(path.stem, json.dumps({key: record.get(key) for key in fields}, ensure_ascii=False))
