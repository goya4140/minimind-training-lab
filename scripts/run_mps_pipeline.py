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
    "llm-pretrain": (
        "artifacts/eval/llm-pretrain-final.json",
        [
            "scripts/evaluate_bpe_llm.py",
            "--config",
            "configs/llm/pretrain-mps.yaml",
            "--checkpoint",
            "artifacts/checkpoints/llm-64m-pretrain-mps.pt",
            "--output",
            "artifacts/eval/llm-pretrain-final.json",
        ],
    ),
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
    "video-omni-sft": (
        "artifacts/eval/video-omni-final.json",
        [
            "scripts/evaluate_video_omni.py",
            "--config",
            "configs/video/sft-mps.yaml",
            "--checkpoint",
            "artifacts/checkpoints/video-omni-sft-mps.pt",
            "--output",
            "artifacts/eval/video-omni-final.json",
        ],
    ),
}


def output_for(config_path: str) -> Path:
    with (ROOT / config_path).open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    return ROOT / config["training"]["checkpoint_path"]


def run(command: list[str], console_log: Path | None = None) -> None:
    printable = " ".join(command)
    print(f"running: {printable}", flush=True)
    resolved = [sys.executable, str(ROOT / command[0]), *command[1:]]
    if console_log is None:
        subprocess.run(resolved, cwd=ROOT, check=True)
        return
    console_log.parent.mkdir(parents=True, exist_ok=True)
    with console_log.open("a", encoding="utf-8") as handle:
        process = subprocess.Popen(
            resolved,
            cwd=ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        if process.stdout is None:
            raise RuntimeError("failed to capture stage output")
        for line in process.stdout:
            print(line, end="", flush=True)
            handle.write(line)
            handle.flush()
        return_code = process.wait()
    if return_code:
        raise subprocess.CalledProcessError(return_code, resolved)


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
        if stage == "video-omni-alignment":
            run(
                ["scripts/fetch_qivd.py"],
                console_log=ROOT / "artifacts/logs/qivd-integrity.console.log",
            )
            run(
                ["scripts/cache_video_features.py", "--config", config],
                console_log=ROOT / "artifacts/logs/video-feature-cache.console.log",
            )
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
            with (ROOT / config).open(encoding="utf-8") as handle:
                experiment = yaml.safe_load(handle)["experiment"]["name"]
            run(
                [runner, "--config", config, "--resume"],
                console_log=ROOT / f"artifacts/logs/{experiment}.console.log",
            )
        if not output.exists():
            raise RuntimeError(f"stage exited without final checkpoint: {stage}")
        verify_stage(config)
        evaluate_if_needed(stage, output)

    run(["scripts/build_final_report.py"])
    print("MPS training, evaluation, and handbook reporting complete", flush=True)
    pipeline_lock.close()


if __name__ == "__main__":
    main()
