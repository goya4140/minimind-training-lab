#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]

STAGES = [
    ("llm-pretrain", "scripts/train_pretrain.py", "configs/llm/pretrain-mps.yaml"),
    ("llm-sft", "scripts/train_sft.py", "configs/llm/sft-mps.yaml"),
    ("vlm-alignment", "scripts/train_vlm.py", "configs/vlm/alignment-mps.yaml"),
    ("vlm-sft", "scripts/train_vlm.py", "configs/vlm/sft-mps.yaml"),
    ("omni-t2a", "scripts/train_omni.py", "configs/omni/t2a-mps.yaml"),
    ("omni-audio-alignment", "scripts/train_omni.py", "configs/omni/a2a-alignment-mps.yaml"),
    ("omni-a2a-sft", "scripts/train_omni.py", "configs/omni/a2a-sft-mps.yaml"),
    ("omni-i2t", "scripts/train_omni.py", "configs/omni/i2t-mps.yaml"),
]
DEPENDENCY_KEYS = ("language_checkpoint", "alignment_checkpoint", "checkpoint")
COMPONENT_KEYS = ("vision_encoder", "audio_encoder", "codec")


def relative(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def active_pid(output: Path) -> int | None:
    lock_path = output.with_suffix(".lock")
    try:
        pid = int(lock_path.read_text(encoding="utf-8").strip())
        os.kill(pid, 0)
    except (FileNotFoundError, ProcessLookupError, PermissionError, ValueError):
        return None
    return pid


def inspect_stage(name: str, runner: str, config_path: str, prior_outputs: set[Path]) -> tuple[dict, Path]:
    absolute_config = ROOT / config_path
    with absolute_config.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    missing = []
    pending = []
    required = [ROOT / runner, ROOT / config["tokenizer"]["path"], ROOT / config["data"]["path"]]
    model = config["model"]
    required.extend(ROOT / model[key] for key in COMPONENT_KEYS if model.get(key))
    for path in required:
        if not path.exists():
            missing.append(relative(path))

    dependencies = [model[key] for key in DEPENDENCY_KEYS if model.get(key)]
    if config["training"].get("initial_checkpoint"):
        dependencies.append(config["training"]["initial_checkpoint"])
    for value in dependencies:
        path = ROOT / value
        if not path.exists():
            if path in prior_outputs:
                pending.append(relative(path))
            else:
                missing.append(relative(path))

    output = ROOT / config["training"]["checkpoint_path"]
    prior_outputs.add(output)
    training = config["training"]
    owner = active_pid(output)
    if missing:
        status = "missing"
    elif output.exists():
        status = "complete"
    elif owner is not None:
        status = "active"
    elif pending:
        status = "pending"
    else:
        status = "ready"
    result = {
        "stage": name,
        "config": config_path,
        "device": config["experiment"]["device"],
        "micro_batch_size": training["batch_size"],
        "gradient_accumulation_steps": training.get("gradient_accumulation_steps", 1),
        "effective_batch_size": training["batch_size"] * training.get("gradient_accumulation_steps", 1),
        "planned_micro_steps": training.get("steps"),
        "planned_sample_exposures": training.get("steps", 0) * training["batch_size"],
        "output": relative(output),
        "status": status,
        "lock_owner_pid": owner,
        "missing": sorted(set(missing)),
        "pending_dependencies": pending,
    }
    return result, output


def main() -> None:
    stages = []
    outputs: set[Path] = set()
    for name, runner, config in STAGES:
        result, _ = inspect_stage(name, runner, config, outputs)
        stages.append(result)
    report = {
        "mps_available": torch.backends.mps.is_available(),
        "all_resources_present": all(not stage["missing"] for stage in stages),
        "stages": stages,
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report["mps_available"] or not report["all_resources_present"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
