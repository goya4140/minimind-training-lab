#!/usr/bin/env python3
"""Readable one-shot view of actual run records, for Terminal evidence capture."""
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
print("MINIMIND | FORMAL EXPERIMENT RECORD REVIEW")
print("Review UTC:", utc_now())
print("Run:", manifest["run_id"])
print("Commit:", manifest["code_commit"])
print("GPU:", manifest["runtime"]["gpu"])
print("Pinned input fingerprints:", len(manifest["data_files"]))
for path in sorted((run / "metrics").glob("*.jsonl")):
    record = json.loads(path.read_text().splitlines()[-1])
    print("Stage:", path.stem)
    print("Recorded UTC:", record["recorded_at_utc"])
    print("Micro-step:", record["step"], "| Optimizer updates:", record["optimizer_updates"])
    print(f"Loss: {record['loss']:.6f} | Gradient norm: {record['grad_norm']}")
    print(f"Learning rate: {record['learning_rate']:.10f}")
    print(f"Active training: {record['training_seconds']:.2f} s")
    print(f"Nominal throughput (includes padding): {record['padded_tokens_per_second']:.0f} token/s")
    print(f"Peak PyTorch allocated: {record['cuda_peak_allocated_bytes'] / 2**30:.3f} GiB")
    print("Resumed from:", record["resumed_from_step"], "| Segment:", record["segment_id"])
audit_path = run / "audits/first-resume.json"
if audit_path.exists():
    audit = json.loads(audit_path.read_text())
    print("Recovery audit step:", audit["resume_step"])
    print("Model finite:", audit["all_finite"], "| Optimizer finite:", audit["optimizer_all_finite"])
print("This is an in-progress training record, not final evaluation.")
