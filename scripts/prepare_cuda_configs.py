#!/usr/bin/env python3
"""Generate explicit CUDA baselines and isolated short probes from the MPS baseline."""

from __future__ import annotations

import argparse
import copy
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
STAGES = [
    ("llm/pretrain", "llm-64m-pretrain"),
    ("llm/sft", "llm-64m-sft"),
    ("vlm/alignment", "vlm-vision-alignment"),
    ("vlm/sft", "vlm-sft"),
    ("video/alignment", "video-omni-alignment"),
    ("video/sft", "video-omni-sft"),
]


def replace_paths(value, probe=False, tag="cuda-probe"):
    if isinstance(value, dict):
        return {key: replace_paths(item, probe, tag) for key, item in value.items()}
    if isinstance(value, list):
        return [replace_paths(item, probe, tag) for item in value]
    if isinstance(value, str):
        suffix = f"-{tag}" if probe else "-cuda"
        return value.replace("-mps", suffix)
    return value


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--probe-tag", default="cuda-probe")
    args = parser.parse_args()
    for stage, name in STAGES:
        source = yaml.safe_load((ROOT / f"configs/{stage}-mps.yaml").read_text())
        config = replace_paths(source)
        config["experiment"]["name"] = name + "-cuda"
        config["experiment"]["device"] = "cuda"
        config["training"]["precision"] = "bfloat16"
        target = ROOT / f"configs/{stage}-cuda.yaml"
        target.write_text(yaml.safe_dump(config, sort_keys=False))
        probe = replace_paths(copy.deepcopy(source), probe=True, tag=args.probe_tag)
        probe["experiment"]["name"] = name + "-" + args.probe_tag
        probe["experiment"]["device"] = "cuda"
        probe["experiment"]["deterministic"] = True
        probe["training"].update(
            steps=8, gradient_accumulation_steps=2, save_interval=4, log_interval=1, precision="bfloat16"
        )
        if "validation_samples" in probe["data"]:
            probe["data"]["validation_samples"] = 2
        if stage.startswith("video/"):
            probe["data"]["train_samples"] = 8
            probe["data"].pop("feature_cache", None)
        (ROOT / f"configs/{stage}-cuda-probe.yaml").write_text(yaml.safe_dump(probe, sort_keys=False))
        print(target.relative_to(ROOT))


if __name__ == "__main__":
    main()
