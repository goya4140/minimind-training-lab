#!/usr/bin/env python3
"""Freeze an auditable formal run before allocating any training process."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from preflight_mps_pipeline import STAGES

from minimind_lab.training import load_config
from minimind_lab.training.evidence import RunEvidence

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    configs = [config.replace("-mps", "-cuda") for _, _, config in STAGES]
    for path in configs:
        checkpoint = ROOT / load_config(ROOT / path)["training"]["checkpoint_path"]
        if checkpoint.exists() or checkpoint.with_suffix(".resume.pt").exists():
            raise RuntimeError(f"formal output already exists; preserve or explicitly resume its run: {checkpoint}")
    data = sorted(
        {
            *ROOT.glob("data/raw/*.jsonl"),
            *ROOT.glob("data/raw/*.parquet"),
            *ROOT.glob("data/raw/qivd/*"),
            *ROOT.glob("data/raw/qivd/videos/*.mp4"),
            *ROOT.glob("assets/tokenizer/*.json"),
            *ROOT.glob("assets/models/siglip2-base-p32-256-ve/*"),
            *ROOT.glob("data/manifests/*.json"),
            *ROOT.glob("data/processed/qivd-siglip2-patch-fp16.*"),
        }
    )
    manifest = RunEvidence(ROOT, args.run_id).create(
        configs, [path for path in data if path.is_file() and not path.name.endswith((".part", ".lock"))]
    )
    print(
        json.dumps(
            {
                "run_id": manifest["run_id"],
                "code_commit": manifest["code_commit"],
                "hashed_data_files": len(manifest["data_files"]),
                "status": "prepared",
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
