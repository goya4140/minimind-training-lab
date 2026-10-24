#!/usr/bin/env python3
"""Run six isolated, eight-step real-data CUDA probes, each resuming at step four."""

from __future__ import annotations

import argparse
import copy
import json
import subprocess
import sys
import time
from pathlib import Path

import torch
import yaml

from minimind_lab.training import load_config, verify_resume_checkpoint

ROOT = Path(__file__).resolve().parents[1]
STAGES = [
    ("llm/pretrain", "train_pretrain.py"),
    ("llm/sft", "train_sft.py"),
    ("vlm/alignment", "train_vlm.py"),
    ("vlm/sft", "train_vlm.py"),
    ("video/alignment", "train_video_omni.py"),
    ("video/sft", "train_video_omni.py"),
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--wait-for-data", action="store_true")
    args = parser.parse_args()
    results = []
    for stage, script in STAGES:
        config_path = f"configs/{stage}-cuda-probe.yaml"
        config = load_config(ROOT / config_path)
        if args.wait_for_data:
            required = ROOT / config["data"]["path"]
            if stage.startswith("video/"):
                required = ROOT / "data/manifests/qivd.json"
            deadline = time.monotonic() + 3600
            if not required.is_file():
                print(f"waiting for verified data: {required}", flush=True)
            while not required.is_file():
                if time.monotonic() > deadline:
                    raise TimeoutError(f"verified data not ready: {required}")
                time.sleep(5)
        checkpoint = ROOT / config["training"]["checkpoint_path"]
        resume = checkpoint.with_suffix(".resume.pt")
        if resume.exists() or checkpoint.exists():
            raise RuntimeError(f"probe output already exists; select a fresh output before rerunning: {resume}")
        log = ROOT / f"artifacts/logs/{config['experiment']['name']}.console.log"
        with log.open("w") as output:
            command = [sys.executable, str(ROOT / "scripts" / script), "--config", config_path]
            subprocess.run(
                [*command, "--stop-after", "4"], cwd=ROOT, stdout=output, stderr=subprocess.STDOUT, check=True
            )
            before = verify_resume_checkpoint(resume, expected_config=config)
            assert before["resume_step"] == 4
            rng_states = torch.load(resume, map_location="cpu", weights_only=False).get("cuda_rng_state")
            assert rng_states and all(state.dtype == torch.uint8 for state in rng_states)
            subprocess.run([*command, "--resume"], cwd=ROOT, stdout=output, stderr=subprocess.STDOUT, check=True)
        after = verify_resume_checkpoint(resume, expected_config=config, require_complete=True)
        if not after["all_finite"] or not after["optimizer_all_finite"]:
            raise RuntimeError(f"non-finite probe state: {stage}")
        summary = json.loads((ROOT / f"artifacts/logs/{config['experiment']['name']}.json").read_text())
        if stage == "llm/pretrain":
            reference = copy.deepcopy(config)
            reference["experiment"]["name"] += "-reference"
            reference["training"]["checkpoint_path"] = str(checkpoint.relative_to(ROOT)).replace(".pt", "-reference.pt")
            reference_path = ROOT / ".cache/cuda-probe-reference.yaml"
            reference_path.parent.mkdir(parents=True, exist_ok=True)
            reference_path.write_text(yaml.safe_dump(reference))
            with (ROOT / "artifacts/logs/cuda-probe-reference.console.log").open("w") as output:
                subprocess.run(
                    [sys.executable, str(ROOT / "scripts" / script), "--config", str(reference_path)],
                    cwd=ROOT,
                    stdout=output,
                    stderr=subprocess.STDOUT,
                    check=True,
                )
            expected = torch.load(
                ROOT / reference["training"]["checkpoint_path"], map_location="cpu", weights_only=False
            )
            actual = torch.load(checkpoint, map_location="cpu", weights_only=False)
            difference = max(
                float((actual["model"][key] - tensor).abs().max()) for key, tensor in expected["model"].items()
            )
            if difference > 1e-6:
                raise RuntimeError(f"resumed model differs from uninterrupted reference: {difference}")
            summary["resume_reference_max_absolute_difference"] = difference
        results.append(
            {"stage": stage, "resumed_from": before["resume_step"], "verification": after, "summary": summary}
        )
        (ROOT / "artifacts/logs/cuda-probe-results.json").write_text(json.dumps(results, indent=2))
        print(json.dumps({"stage": stage, "passed": True, "environment": summary["environment"]}), flush=True)


if __name__ == "__main__":
    main()
