#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
from transformers import SiglipImageProcessor, SiglipVisionModel

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from minimind_lab.data import decode_uniform_video
from minimind_lab.training import acquire_run_lock, load_config, resolve_device, seed_everything
from minimind_lab.training.utils import environment_info


def main() -> None:
    parser = argparse.ArgumentParser(description="Cache frozen SigLIP2 patch features for QIVD training.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--batch-size", type=int, default=2)
    args = parser.parse_args()
    config = load_config(args.config)
    seed_everything(config["experiment"]["seed"])
    device = resolve_device(config["experiment"]["device"])
    data = config["data"]
    model = config["model"]
    root = ROOT / data["path"]
    output = ROOT / data["feature_cache"]
    done_path = output.with_suffix(".done.npy")
    manifest_path = output.with_suffix(".manifest.json")
    rows = pq.read_table(root / "metadata.parquet").to_pylist()
    num_frames = int(model["num_frames"])
    expected_shape = (len(rows), num_frames, 64, int(model["vision_hidden_size"]))
    if output.is_file() and done_path.is_file():
        existing = np.load(output, mmap_mode="r")
        done = np.load(done_path, mmap_mode="r")
        if existing.shape == expected_shape and done.shape == (len(rows),) and bool(done.all()):
            print(json.dumps({"status": "already-complete", "path": str(output), "shape": expected_shape}))
            return

    lock = acquire_run_lock(output.with_suffix(".lock"))
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        cache = np.load(output, mmap_mode="r+")
        if cache.shape != expected_shape or cache.dtype != np.float16:
            raise ValueError("existing feature cache has incompatible shape or dtype")
    else:
        cache = np.lib.format.open_memmap(output, mode="w+", dtype=np.float16, shape=expected_shape)
    if done_path.exists():
        done = np.load(done_path, mmap_mode="r+")
        if done.shape != (len(rows),):
            raise ValueError("existing completion bitmap has incompatible shape")
    else:
        done = np.lib.format.open_memmap(done_path, mode="w+", dtype=np.bool_, shape=(len(rows),))
        done[:] = False
        done.flush()

    vision_path = ROOT / model["vision_encoder"]
    processor = SiglipImageProcessor.from_pretrained(vision_path, local_files_only=True)
    vision = SiglipVisionModel.from_pretrained(vision_path, local_files_only=True).to(device).eval()
    started = time.time()
    completed_before = int(done.sum())
    pending = [index for index in range(len(rows)) if not done[index]]
    for batch_start in range(0, len(pending), args.batch_size):
        indices = pending[batch_start : batch_start + args.batch_size]
        frames = [
            decode_uniform_video(root / rows[index]["video_file_name"], num_frames)
            for index in indices
        ]
        pixels = processor(images=[image for video in frames for image in video], return_tensors="pt")[
            "pixel_values"
        ].to(device)
        with torch.inference_mode():
            features = vision(pixels).last_hidden_state
        features = features.view(len(indices), num_frames, features.size(1), features.size(2))
        if tuple(features.shape[1:]) != expected_shape[1:]:
            raise ValueError(f"unexpected SigLIP2 feature shape: {tuple(features.shape)}")
        cache[indices] = features.cpu().to(torch.float16).numpy()
        done[indices] = True
        if batch_start == 0 or (batch_start // args.batch_size + 1) % 25 == 0:
            cache.flush()
            done.flush()
            completed = int(done.sum())
            print(
                json.dumps(
                    {
                        "completed": completed,
                        "total": len(rows),
                        "seconds_per_video": (time.time() - started) / max(completed - completed_before, 1),
                    }
                ),
                flush=True,
            )
    cache.flush()
    done.flush()
    report = {
        "status": "complete",
        "path": str(output.relative_to(ROOT)),
        "shape": expected_shape,
        "dtype": "float16",
        "bytes": output.stat().st_size,
        "videos": len(rows),
        "frames_per_video": num_frames,
        "training_only": True,
        "environment": environment_info(device),
    }
    temporary = manifest_path.with_suffix(".tmp")
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, manifest_path)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    lock.close()


if __name__ == "__main__":
    main()
