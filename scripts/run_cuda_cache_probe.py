#!/usr/bin/env python3
"""Exercise the formal cached-video input path using isolated probe checkpoints."""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

import yaml

from minimind_lab.training import load_config, verify_resume_checkpoint

ROOT = Path(__file__).resolve().parents[1]


def main():
    config = load_config(ROOT / "configs/video/alignment-cuda-probe.yaml")
    formal = load_config(ROOT / "configs/video/alignment-cuda.yaml")
    config["data"]["feature_cache"] = formal["data"]["feature_cache"]
    config["experiment"]["name"] += "-cached"
    config["training"]["checkpoint_path"] = config["training"]["checkpoint_path"].replace(".pt", "-cached.pt")
    config["training"]["steps"] = 4
    checkpoint = ROOT / config["training"]["checkpoint_path"]
    if checkpoint.exists() or checkpoint.with_suffix(".resume.pt").exists():
        raise RuntimeError("cached probe output already exists")
    source_model = ROOT / config["model"]["language_checkpoint"]
    manifest = (ROOT / config["data"]["feature_cache"]).with_suffix(".manifest.json")
    deadline = time.monotonic() + 1800
    while not source_model.is_file() or not manifest.is_file():
        if time.monotonic() > deadline:
            raise TimeoutError("cached probe prerequisites are not ready")
        time.sleep(5)
    path = ROOT / ".cache/cuda-cached-video-probe.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False))
    subprocess.run(
        [sys.executable, str(ROOT / "scripts/train_video_omni.py"), "--config", str(path)], cwd=ROOT, check=True
    )
    result = verify_resume_checkpoint(
        checkpoint.with_suffix(".resume.pt"), expected_config=config, require_complete=True
    )
    (ROOT / "artifacts/logs/cuda-cached-video-probe-result.json").write_text(json.dumps(result, indent=2))
    print(json.dumps({"cached_video_probe": "passed", **result}), flush=True)


if __name__ == "__main__":
    main()
