#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from preflight_mps_pipeline import STAGES, active_pid

from minimind_lab.training import acquire_run_lock

EVALUATIONS = {
    "llm-sft": (
        "artifacts/eval/llm-sft-final.json",
        [
            "scripts/evaluate_bpe_llm.py",
            "--config",
            "configs/llm/pretrain-mps.yaml",
            "--checkpoint",
            "artifacts/checkpoints/llm-64m-sft-mps.pt",
            "--output",
            "artifacts/eval/llm-sft-final.json",
        ],
    ),
    "vlm-sft": (
        "artifacts/eval/vlm-final.json",
        [
            "scripts/evaluate_vlm.py",
            "--config",
            "configs/vlm/sft-mps.yaml",
            "--checkpoint",
            "artifacts/checkpoints/vlm-sft-mps.pt",
            "--output",
            "artifacts/eval/vlm-final.json",
        ],
    ),
    "omni-i2t": (
        "artifacts/eval/omni-final.json",
        [
            "scripts/evaluate_omni.py",
            "--checkpoint",
            "artifacts/checkpoints/omni-i2t-mps.pt",
            "--device",
            "mps",
            "--output",
            "artifacts/eval/omni-final.json",
        ],
    ),
}


def output_for(config_path: str) -> Path:
    with (ROOT / config_path).open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    return ROOT / config["training"]["checkpoint_path"]


def run(command: list[str]) -> None:
    printable = " ".join(command)
    print(f"running: {printable}", flush=True)
    resolved = [sys.executable, str(ROOT / command[0]), *command[1:]]
    subprocess.run(resolved, cwd=ROOT, check=True)


def evaluate_if_needed(stage: str, checkpoint: Path) -> None:
    if stage not in EVALUATIONS:
        return
    output_name, command = EVALUATIONS[stage]
    output = ROOT / output_name
    if output.exists() and output.stat().st_mtime >= checkpoint.stat().st_mtime:
        print(f"evaluation already current: {output_name}", flush=True)
        return
    run(command)


def verify_stage(config: str) -> None:
    run(["scripts/verify_stage_artifact.py", "--config", config])


def main() -> None:
    parser = argparse.ArgumentParser(description="Resume the complete MiniMind MPS training and evaluation chain.")
    parser.add_argument("--poll-seconds", type=float, default=30.0)
    args = parser.parse_args()
    pipeline_lock = acquire_run_lock(ROOT / "artifacts/checkpoints/pipeline-mps.lock")
    print(json.dumps({"pipeline": "mps", "stages": [item[0] for item in STAGES]}, ensure_ascii=False), flush=True)

    for stage, runner, config in STAGES:
        output = output_for(config)
        polls = 0
        while active_pid(output) is not None:
            if polls % 20 == 0:
                print(f"waiting for active stage: {stage} (lock owner {active_pid(output)})", flush=True)
            polls += 1
            time.sleep(args.poll_seconds)
        if output.exists():
            print(f"checkpoint already complete: {output.relative_to(ROOT)}", flush=True)
        else:
            run([runner, "--config", config, "--resume"])
        if not output.exists():
            raise RuntimeError(f"stage exited without final checkpoint: {stage}")
        verify_stage(config)
        evaluate_if_needed(stage, output)

    print("MPS training and evaluation pipeline complete", flush=True)
    pipeline_lock.close()


if __name__ == "__main__":
    main()
