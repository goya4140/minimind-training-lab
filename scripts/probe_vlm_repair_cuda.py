#!/usr/bin/env python3
"""One CUDA update on the exact formerly failing batch; no formal weights saved."""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch
from transformers import AutoTokenizer, SiglipImageProcessor, SiglipVisionModel
from minimind_lab.data import ParquetVLMDataset, collate_vlm
from minimind_lab.training import load_config, seed_everything
from minimind_lab.training.artifacts import file_sha256
from minimind_lab.vlm import MiniMindVLM, VLMConfig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-root", required=True, type=Path)
    parser.add_argument("--parent-run-id", required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Do not overwrite a previous CUDA probe")
    parent_run = args.parent_root.resolve() / "artifacts/runs" / args.parent_run_id
    manifest = json.loads((parent_run / "manifest.json").read_text())
    if os.environ.get("CUDA_VISIBLE_DEVICES") != manifest["runtime"]["cuda_visible_devices"]:
        raise ValueError("Probe requires the parent run's exact GPU binding")
    config = load_config(ROOT / "configs/vlm/alignment-cuda.yaml")
    seed_everything(config["experiment"]["seed"])
    tokenizer = AutoTokenizer.from_pretrained(ROOT / "assets/tokenizer", local_files_only=True)
    vision_path = ROOT / config["model"]["vision_encoder"]
    processor = SiglipImageProcessor.from_pretrained(vision_path, local_files_only=True)
    vision = SiglipVisionModel.from_pretrained(vision_path, local_files_only=True)
    values = dict(config["model"])
    values.pop("vision_encoder"); values.pop("language_checkpoint")
    model = MiniMindVLM(VLMConfig(**values), vision)
    checkpoint = parent_run / "stages/llm-sft/llm-64m-sft-cuda.pt"
    events = [json.loads(line) for line in (parent_run / "events.jsonl").read_text().splitlines()]
    hashes = {a["sha256"] for e in events if e.get("kind") == "artifacts_archived"
              and e.get("stage") == "llm-sft" for a in e["artifacts"]
              if Path(a["archive"]).name == checkpoint.name}
    checkpoint_hash = file_sha256(checkpoint)
    if hashes != {checkpoint_hash}:
        raise ValueError("Probe LLM checkpoint must match the archived parent artifact")
    model.language_model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=False)["model"])
    model.set_alignment_trainable()
    model.to("cuda").train()
    dataset = ParquetVLMDataset(ROOT / config["data"]["path"], tokenizer, processor,
                                config["data"]["sequence_length"])
    indices = [606220, 1246578, 584084, 1175311]
    ids, labels, pixels, counts = collate_vlm([dataset[i] for i in indices])
    slots = (ids == model.config.image_token_id).sum(dim=1).tolist()
    targets = (labels[:, 1:] != -100).sum(dim=1).tolist()
    assert slots == [64] * 4 and min(targets) > 0
    parameters = [p for p in model.parameters() if p.requires_grad]
    before = [p.detach().clone() for p in parameters]
    optimizer = torch.optim.AdamW(parameters, lr=config["training"]["learning_rate"])
    with torch.autocast("cuda", dtype=torch.bfloat16):
        loss = model(ids.cuda(), pixels.cuda(), labels.cuda(), image_counts=counts.cuda())["loss"]
    assert bool(torch.isfinite(loss))
    loss.backward()
    norm = float(torch.nn.utils.clip_grad_norm_(parameters, 1.0))
    assert math.isfinite(norm) and norm > 0
    optimizer.step()
    torch.cuda.synchronize()
    changed = any(not torch.equal(old, new) for old, new in zip(before, parameters))
    assert changed and all(bool(torch.isfinite(p).all()) for p in parameters)
    assert all(p.grad is None for p in model.language_model.parameters())
    assert all(p.grad is None for p in model.vision_encoder.parameters())
    result = {"schema": 1, "status": "passed", "kind": "cuda-regression-probe-not-formal-training",
              "batch_filtered_indices": indices, "image_slots": slots, "supervised_targets": targets,
              "loss": float(loss), "grad_norm": norm, "projector_changed": changed,
              "frozen_language_and_vision_have_no_gradients": True,
              "checkpoint_sha256": checkpoint_hash, "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
              "torch": torch.__version__, "gpu": torch.cuda.get_device_name(),
              "vlm_data_sha256": file_sha256(ROOT / "src/minimind_lab/data/vlm.py"),
              "probe_tool_sha256": file_sha256(Path(__file__)), "formal_weights_saved": False}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as handle:
        json.dump(result, handle, indent=2)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
