#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from minimind_lab.data import DeterministicBatchStream, JsonlPretrainDataset
from minimind_lab.llm import MiniMindConfig, MiniMindForCausalLM
from minimind_lab.training import (
    acquire_run_lock,
    load_config,
    optimizer_step_size,
    rescale_partial_accumulation,
    resolve_device,
    seed_everything,
    should_save_resume,
)
from minimind_lab.training.utils import environment_info, write_json


@torch.inference_mode()
def validation_loss(model, loader, device: torch.device) -> float:
    model.eval()
    losses = []
    for input_ids, labels in loader:
        output = model(input_ids.to(device), labels.to(device))
        losses.append(output["loss"].item())
    model.train()
    return sum(losses) / len(losses)


def learning_rate(step: int, total_steps: int, base_lr: float) -> float:
    return base_lr * (0.1 + 0.45 * (1 + math.cos(math.pi * step / total_steps)))


def save_resume(
    path: Path, model, optimizer, config: dict, step: int, history: list[dict], training_seconds: float
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "config": config,
            "step": step,
            "history": history,
            "training_seconds": training_seconds,
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    run_lock = acquire_run_lock((ROOT / config["training"]["checkpoint_path"]).with_suffix(".lock"))
    seed_everything(config["experiment"]["seed"])
    device = resolve_device(config["experiment"]["device"])
    tokenizer = AutoTokenizer.from_pretrained(ROOT / config["tokenizer"]["path"], local_files_only=True)
    dataset = JsonlPretrainDataset(
        ROOT / config["data"]["path"], tokenizer, sequence_length=config["data"]["sequence_length"]
    )
    validation_size = config["data"]["validation_samples"]
    if len(dataset) <= validation_size:
        raise ValueError("validation split is larger than the dataset")
    split = len(dataset) - validation_size
    train_dataset = Subset(dataset, range(split))
    validation_dataset = Subset(dataset, range(split, len(dataset)))
    training = config["training"]
    batch_stream = DeterministicBatchStream(
        train_dataset, batch_size=training["batch_size"], seed=config["experiment"]["seed"]
    )
    validation_loader = DataLoader(validation_dataset, batch_size=training["batch_size"], num_workers=0)
    model = MiniMindForCausalLM(MiniMindConfig(**config["model"])).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=training["learning_rate"], weight_decay=training["weight_decay"]
    )
    accumulation = training["gradient_accumulation_steps"]
    history = []
    start_step = 0
    prior_training_seconds = 0.0
    resume_path = (ROOT / training["checkpoint_path"]).with_suffix(".resume.pt")
    if args.resume and resume_path.exists():
        resume = torch.load(resume_path, map_location="cpu", weights_only=False)
        model.load_state_dict(resume["model"])
        optimizer.load_state_dict(resume["optimizer"])
        optimizer_to(optimizer, device)
        torch.set_rng_state(resume["torch_rng_state"])
        start_step = resume["step"]
        history = resume["history"]
        prior_training_seconds = float(resume.get("training_seconds", 0))
        print(f"resuming from step {start_step}: {resume_path}")
    started = time.time()
    model.train()
    optimizer.zero_grad(set_to_none=True)
    last_grad_norm = None
    for step in range(start_step + 1, training["steps"] + 1):
        input_ids, labels = batch_stream.batch(step)
        output = model(input_ids.to(device), labels.to(device))
        loss = output["loss"] / accumulation
        loss.backward()
        accumulated = optimizer_step_size(step, training["steps"], accumulation)
        if accumulated:
            rescale_partial_accumulation(model.parameters(), accumulated, accumulation)
            current_lr = learning_rate(step, training["steps"], training["learning_rate"])
            for group in optimizer.param_groups:
                group["lr"] = current_lr
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), training["grad_clip"])
            last_grad_norm = grad_norm.detach().item()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        if step == 1 or step % training["log_interval"] == 0:
            elapsed = time.time() - started
            completed_this_run = step - start_step
            record = {
                "step": step,
                "loss": loss.detach().item() * accumulation,
                "grad_norm": last_grad_norm,
                "seconds_per_step": elapsed / completed_this_run,
                "learning_rate": optimizer.param_groups[0]["lr"],
            }
            history.append(record)
            print(json.dumps(record, ensure_ascii=False), flush=True)
        save_interval = training.get("save_interval", 0)
        if should_save_resume(step, save_interval, accumulation):
            save_resume(
                resume_path,
                model,
                optimizer,
                config,
                step,
                history,
                prior_training_seconds + time.time() - started,
            )
            print(f"saved resume checkpoint: {resume_path}", flush=True)
    training_elapsed = prior_training_seconds + time.time() - started
    val_loss = validation_loss(model, validation_loader, device)
    checkpoint_path = ROOT / training["checkpoint_path"]
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "config": config, "history": history}, checkpoint_path)
    save_resume(resume_path, model, optimizer, config, training["steps"], history, training_elapsed)
    elapsed = time.time() - started
    report = {
        "experiment": config["experiment"]["name"],
        "status": "complete",
        "parameters": model.parameter_count(),
        "dataset_samples": len(dataset),
        "train_samples": len(train_dataset),
        "validation_samples": len(validation_dataset),
        "total_steps": training["steps"],
        "elapsed_seconds": round(elapsed, 3),
        "training_seconds": training_elapsed,
        "seconds_per_step": training_elapsed / training["steps"],
        "tokens_per_second": training["batch_size"]
        * config["data"]["sequence_length"]
        * training["steps"]
        / training_elapsed,
        "validation_loss": val_loss,
        "validation_perplexity": math.exp(min(val_loss, 20)),
        "environment": environment_info(device),
        "checkpoint": str(checkpoint_path.relative_to(ROOT)),
    }
    write_json(ROOT / f"artifacts/logs/{config['experiment']['name']}.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    run_lock.close()


if __name__ == "__main__":
    main()
