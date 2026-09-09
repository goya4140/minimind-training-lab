#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path

import torch
from transformers import AutoTokenizer, SiglipImageProcessor, SiglipVisionModel

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from minimind_lab.data import DeterministicBatchStream, ParquetOmniDataset, collate_omni
from minimind_lab.omni import MiniMindOmni, OmniConfig
from minimind_lab.omni.external import load_sensevoice
from minimind_lab.training import acquire_run_lock, load_config, resolve_device, resolve_training_steps, seed_everything
from minimind_lab.training.utils import environment_info, write_json

PATH_KEYS = {"language_checkpoint", "checkpoint", "audio_encoder", "vision_encoder", "codec"}


def model_state(model: MiniMindOmni) -> dict[str, torch.Tensor]:
    return model.state_dict()


def save_resume(path, model, optimizer, config, model_config, step, history) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(
        {
            "model": model_state(model),
            "model_config": model_config,
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


def prepare_features(batch, audio_encoder, vision_encoder, device):
    encoded_audio = encoded_audio_lengths = encoded_images = None
    if audio_encoder is not None and batch["audio_features"] is not None:
        with torch.no_grad():
            encoded_audio, encoded_audio_lengths = audio_encoder(
                batch["audio_features"].to(device), batch["audio_lengths"].to(device)
            )
    if vision_encoder is not None and batch["pixel_values"] is not None:
        with torch.no_grad():
            encoded_images = vision_encoder(pixel_values=batch["pixel_values"].to(device)).last_hidden_state
    return encoded_audio, encoded_audio_lengths, encoded_images


def forward_batch(model, batch, audio_encoder, vision_encoder, device):
    encoded_audio, audio_lengths, encoded_images = prepare_features(batch, audio_encoder, vision_encoder, device)
    return model(
        batch["text_ids"].to(device),
        batch["audio_ids"].to(device),
        text_labels=batch["text_labels"].to(device),
        audio_labels=batch["audio_labels"].to(device),
        encoded_audio=encoded_audio,
        encoded_audio_lengths=audio_lengths,
        encoded_images=encoded_images,
        speaker_embedding=batch["speaker_embedding"].to(device),
        speaker_positions=batch["speaker_positions"].to(device),
    )


@torch.inference_mode()
def validation_loss(model, dataset, batch_size, audio_encoder, vision_encoder, device) -> float:
    model.eval()
    losses = []
    for start in range(0, len(dataset), batch_size):
        samples = [dataset[index] for index in range(start, min(start + batch_size, len(dataset)))]
        output = forward_batch(model, collate_omni(samples), audio_encoder, vision_encoder, device)
        if output.loss is not None and torch.isfinite(output.loss):
            losses.append(float(output.loss.item()))
    model.train()
    if not losses:
        raise RuntimeError("Omni validation produced no finite losses")
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
    language_path = model_values.get("language_checkpoint")
    checkpoint_path = model_values.get("checkpoint")
    initial = None
    if checkpoint_path:
        initial = torch.load(ROOT / checkpoint_path, map_location="cpu", weights_only=False)
        architecture = dict(initial["model_config"])
    elif language_path:
        initial = torch.load(ROOT / language_path, map_location="cpu", weights_only=False)
        architecture = dict(initial["config"]["model"])
    else:
        raise ValueError("an LLM or Omni initial checkpoint is required")
    architecture.update({key: value for key, value in model_values.items() if key not in PATH_KEYS})
    model = MiniMindOmni(OmniConfig(**architecture)).to(device)
    if checkpoint_path:
        model.load_state_dict(initial["model"])
    else:
        model.thinker.load_state_dict(initial["model"])
        model.initialize_talker_from_thinker()
    model.configure_trainable(training["stage"])

    audio_encoder = audio_processor = None
    if model_values.get("audio_encoder"):
        audio_encoder, audio_processor = load_sensevoice(ROOT / model_values["audio_encoder"], device)
    vision_encoder = vision_processor = None
    if model_values.get("vision_encoder"):
        vision_path = ROOT / model_values["vision_encoder"]
        vision_encoder = SiglipVisionModel.from_pretrained(vision_path, local_files_only=True).eval().to(device)
        vision_processor = SiglipImageProcessor.from_pretrained(vision_path, local_files_only=True)
        for parameter in vision_encoder.parameters():
            parameter.requires_grad = False

    tokenizer = AutoTokenizer.from_pretrained(ROOT / config["tokenizer"]["path"], local_files_only=True)
    full_dataset = ParquetOmniDataset(
        ROOT / config["data"]["path"],
        tokenizer,
        sequence_length=config["data"]["sequence_length"],
        audio_processor=audio_processor,
        vision_processor=vision_processor,
        num_codebooks=model.config.num_audio_codebooks,
        audio_pad_token_id=model.config.audio_pad_token_id,
        audio_stop_token_id=model.config.audio_stop_token_id,
        audio_speaker_token_id=model.config.audio_speaker_token_id,
        speaker_embedding_size=model.config.speaker_embedding_size,
        image_token_length=model.config.image_token_length,
    )
    validation_size = config["data"]["validation_samples"]
    split = len(full_dataset) - validation_size
    if split <= 0:
        raise ValueError("validation split is larger than the dataset")
    train_dataset = torch.utils.data.Subset(full_dataset, range(split))
    validation = torch.utils.data.Subset(full_dataset, range(split, len(full_dataset)))
    batch_stream = DeterministicBatchStream(train_dataset, training["batch_size"], config["experiment"]["seed"])
    total_steps = resolve_training_steps(training, batch_stream.steps_per_epoch)
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
        model.load_state_dict(resume["model"])
        optimizer.load_state_dict(resume["optimizer"])
        optimizer_to(optimizer, device)
        torch.set_rng_state(resume["torch_rng_state"])
        start_step, history = resume["step"], resume["history"]
        print(f"resuming from step {start_step}: {resume_path}")

    started = time.time()
    model.train()
    optimizer.zero_grad(set_to_none=True)
    last_grad_norm = None
    resolved_config = asdict(model.config)
    for step in range(start_step + 1, total_steps + 1):
        samples = [train_dataset[index] for index in batch_stream.indices_for_step(step)]
        output = forward_batch(model, collate_omni(samples), audio_encoder, vision_encoder, device)
        loss = output.loss / accumulation
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
                "text_loss": float(output.text_loss.detach().item()) if output.text_loss is not None else None,
                "audio_loss": float(output.audio_loss.detach().item()) if output.audio_loss is not None else None,
                "grad_norm": last_grad_norm,
                "seconds_per_step": (time.time() - started) / (step - start_step),
                "learning_rate": optimizer.param_groups[0]["lr"],
            }
            history.append(record)
            print(json.dumps(record, ensure_ascii=False), flush=True)
        if step % training.get("save_interval", 1000) == 0:
            save_resume(resume_path, model, optimizer, config, resolved_config, step, history)
            print(f"saved resume checkpoint: {resume_path}", flush=True)

    training_seconds = time.time() - started
    val_loss = validation_loss(model, validation, training["batch_size"], audio_encoder, vision_encoder, device)
    final_path = ROOT / training["checkpoint_path"]
    final_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"model": model_state(model), "model_config": resolved_config, "config": config, "history": history},
        final_path,
    )
    save_resume(resume_path, model, optimizer, config, resolved_config, total_steps, history)
    report = {
        "experiment": config["experiment"]["name"],
        "status": "complete",
        "stage": training["stage"],
        "dataset_samples": len(full_dataset),
        "total_steps": total_steps,
        "training_seconds": training_seconds,
        "validation_loss": val_loss,
        "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "environment": environment_info(device),
        "checkpoint": str(final_path.relative_to(ROOT)),
    }
    write_json(ROOT / f"artifacts/logs/{config['experiment']['name']}.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    run_lock.close()


if __name__ == "__main__":
    main()
