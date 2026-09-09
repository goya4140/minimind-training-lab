#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from contextlib import nullcontext
from pathlib import Path

import torch
from torch.utils.data import Subset
from transformers import AutoTokenizer, SiglipImageProcessor, SiglipVisionModel

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from minimind_lab.data import DeterministicBatchStream, ParquetVLMDataset, collate_vlm
from minimind_lab.training import acquire_run_lock, load_config, resolve_device, seed_everything
from minimind_lab.training.utils import environment_info, write_json
from minimind_lab.vlm import MiniMindVLM, VLMConfig


def trainable_state(model: MiniMindVLM) -> dict[str, torch.Tensor]:
    return {key: value for key, value in model.state_dict().items() if not key.startswith("vision_encoder.")}


def save_resume(path: Path, model, optimizer, config: dict, step: int, history: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(
        {
            "model": trainable_state(model),
            "optimizer": optimizer.state_dict(),
            "config": config,
            "step": step,
            "history": history,
            "torch_rng_state": torch.get_rng_state(),
        },
        temporary,
    )
    os.replace(temporary, path)


def optimizer_to(optimizer, device: torch.device) -> None:
    for state in optimizer.state.values():
        for key, value in state.items():
            if isinstance(value, torch.Tensor):
                state[key] = value.to(device)


def cosine_lr(step: int, total_steps: int, base_lr: float) -> float:
    return base_lr * (0.1 + 0.45 * (1 + math.cos(math.pi * step / total_steps)))


@torch.inference_mode()
def validation_loss(model, dataset, batch_size: int, device: torch.device) -> float:
    model.eval()
    losses = []
    for start in range(0, len(dataset), batch_size):
        samples = [dataset[index] for index in range(start, min(start + batch_size, len(dataset)))]
        input_ids, labels, pixels, counts = collate_vlm(samples)
        output = model(
            input_ids.to(device),
            pixels.to(device),
            labels.to(device),
            image_counts=counts.to(device),
        )
        if torch.isfinite(output["loss"]):
            losses.append(float(output["loss"].item()))
    model.train()
    if not losses:
        raise RuntimeError("VLM validation produced no finite losses")
    return sum(losses) / len(losses)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    training = config["training"]
    run_lock = acquire_run_lock((ROOT / training["checkpoint_path"]).with_suffix(".lock"))
    seed_everything(config["experiment"]["seed"])
    device = resolve_device(config["experiment"]["device"])

    model_values = dict(config["model"])
    language_checkpoint = model_values.pop("language_checkpoint", None)
    alignment_checkpoint = model_values.pop("alignment_checkpoint", None)
    vision_path = ROOT / model_values.pop("vision_encoder")
    tokenizer = AutoTokenizer.from_pretrained(ROOT / config["tokenizer"]["path"], local_files_only=True)
    processor = SiglipImageProcessor.from_pretrained(vision_path, local_files_only=True)
    vision_encoder = SiglipVisionModel.from_pretrained(vision_path, local_files_only=True)
    model = MiniMindVLM(VLMConfig(**model_values), vision_encoder).to(device)

    stage = training["stage"]
    if stage == "projector-only":
        initial = torch.load(ROOT / language_checkpoint, map_location="cpu", weights_only=False)
        model.language_model.load_state_dict(initial["model"])
        model.set_alignment_trainable()
    elif stage == "projector-and-llm-boundary-layers":
        initial = torch.load(ROOT / alignment_checkpoint, map_location="cpu", weights_only=False)
        model.load_state_dict(initial["model"], strict=False)
        model.set_instruction_trainable()
    else:
        raise ValueError(f"unknown VLM training stage: {stage}")

    full_dataset = ParquetVLMDataset(
        ROOT / config["data"]["path"],
        tokenizer,
        processor,
        sequence_length=config["data"]["sequence_length"],
        image_token_length=model.config.image_token_length,
    )
    validation_size = config["data"]["validation_samples"]
    split = len(full_dataset) - validation_size
    if split <= 0:
        raise ValueError("validation split is larger than the dataset")
    dataset = Subset(full_dataset, range(split))
    validation = Subset(full_dataset, range(split, len(full_dataset)))
    batch_stream = DeterministicBatchStream(
        dataset, batch_size=training["batch_size"], seed=config["experiment"]["seed"]
    )
    total_steps = batch_stream.steps_per_epoch * training["epochs"]
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=training["learning_rate"],
        weight_decay=training.get("weight_decay", 0.1),
    )
    accumulation = training.get("gradient_accumulation_steps", 1)
    resume_path = (ROOT / training["checkpoint_path"]).with_suffix(".resume.pt")
    start_step, history = 0, []
    if args.resume and resume_path.exists():
        resume = torch.load(resume_path, map_location="cpu", weights_only=False)
        model.load_state_dict(resume["model"], strict=False)
        optimizer.load_state_dict(resume["optimizer"])
        optimizer_to(optimizer, device)
        torch.set_rng_state(resume["torch_rng_state"])
        start_step, history = resume["step"], resume["history"]
        print(f"resuming from step {start_step}: {resume_path}")

    use_amp = device.type == "cuda" and training.get("precision") == "bfloat16"
    autocast = torch.autocast("cuda", dtype=torch.bfloat16) if use_amp else nullcontext()
    started = time.time()
    optimizer.zero_grad(set_to_none=True)
    model.train()
    last_grad_norm = None
    for step in range(start_step + 1, total_steps + 1):
        samples = [dataset[index] for index in batch_stream.indices_for_step(step)]
        input_ids, labels, pixels, counts = collate_vlm(samples)
        with autocast:
            loss = (
                model(
                    input_ids.to(device),
                    pixels.to(device),
                    labels.to(device),
                    image_counts=counts.to(device),
                )["loss"]
                / accumulation
            )
        if not torch.isfinite(loss):
            raise RuntimeError(f"non-finite loss at step {step}")
        loss.backward()
        if step % accumulation == 0:
            for group in optimizer.param_groups:
                group["lr"] = cosine_lr(step, total_steps, training["learning_rate"])
            last_grad_norm = float(
                torch.nn.utils.clip_grad_norm_(model.parameters(), training.get("grad_clip", 1.0)).item()
            )
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        if step == 1 or step % training.get("log_interval", 20) == 0:
            record = {
                "step": step,
                "loss": float(loss.detach().item() * accumulation),
                "grad_norm": last_grad_norm,
                "seconds_per_step": (time.time() - started) / (step - start_step),
                "learning_rate": optimizer.param_groups[0]["lr"],
            }
            history.append(record)
            print(json.dumps(record, ensure_ascii=False), flush=True)
        if step % training.get("save_interval", 1000) == 0:
            save_resume(resume_path, model, optimizer, config, step, history)
            print(f"saved resume checkpoint: {resume_path}", flush=True)

    training_seconds = time.time() - started
    val_loss = validation_loss(model, validation, training["batch_size"], device)
    checkpoint_path = ROOT / training["checkpoint_path"]
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": trainable_state(model), "config": config, "history": history}, checkpoint_path)
    save_resume(resume_path, model, optimizer, config, total_steps, history)
    report = {
        "experiment": config["experiment"]["name"],
        "status": "complete",
        "stage": stage,
        "dataset_samples": len(full_dataset),
        "total_steps": total_steps,
        "training_seconds": training_seconds,
        "validation_loss": val_loss,
        "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "environment": environment_info(device),
        "checkpoint": str(checkpoint_path.relative_to(ROOT)),
    }
    write_json(ROOT / f"artifacts/logs/{config['experiment']['name']}.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    run_lock.close()


if __name__ == "__main__":
    main()
