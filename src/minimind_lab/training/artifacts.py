from __future__ import annotations

import hashlib
import math
from pathlib import Path
from typing import Any

import torch


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_model_state(target: Path, checkpoint: dict[str, Any]) -> dict[str, int | str | bool]:
    state = checkpoint.get("model")
    if not isinstance(state, dict) or not state:
        raise ValueError("checkpoint does not contain a non-empty model state")
    tensors = [value for value in state.values() if isinstance(value, torch.Tensor)]
    if not tensors:
        raise ValueError("checkpoint model state does not contain tensors")
    unique_storages = {}
    for tensor in tensors:
        storage = tensor.untyped_storage()
        unique_storages.setdefault((storage.data_ptr(), storage.nbytes()), tensor)
    return {
        "checkpoint_bytes": target.stat().st_size,
        "checkpoint_sha256": file_sha256(target),
        "state_tensors": len(tensors),
        "state_parameters": sum(tensor.numel() for tensor in unique_storages.values()),
        "all_finite": all(bool(torch.isfinite(tensor).all()) for tensor in tensors),
    }


def verify_checkpoint(path: str | Path) -> dict[str, int | str | bool]:
    target = Path(path)
    checkpoint = torch.load(target, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict):
        raise TypeError("checkpoint root must be a mapping")
    return _verify_model_state(target, checkpoint)


def verify_resume_checkpoint(
    path: str | Path,
    expected_config: dict[str, Any] | None = None,
    *,
    require_complete: bool = False,
) -> dict[str, int | float | str | bool]:
    target = Path(path)
    checkpoint = torch.load(target, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict):
        raise TypeError("resume checkpoint root must be a mapping")
    required = {"model", "optimizer", "config", "step", "history", "training_seconds", "torch_rng_state"}
    missing = sorted(required - checkpoint.keys())
    if missing:
        raise ValueError(f"resume checkpoint is missing keys: {', '.join(missing)}")

    config = checkpoint["config"]
    if not isinstance(config, dict):
        raise TypeError("resume config must be a mapping")
    if expected_config is not None and config != expected_config:
        raise ValueError("resume config does not match the requested experiment config")
    training = config.get("training")
    if not isinstance(training, dict):
        raise TypeError("resume config does not contain a training mapping")

    step = checkpoint["step"]
    if isinstance(step, bool) or not isinstance(step, int) or step < 0:
        raise ValueError("resume step must be a non-negative integer")
    total_steps = training.get("steps")
    accumulation = training.get("gradient_accumulation_steps")
    if not isinstance(total_steps, int) or total_steps <= 0 or step > total_steps:
        raise ValueError("resume step is outside the configured training range")
    if require_complete and step != total_steps:
        raise ValueError("resume checkpoint has not reached the configured final step")
    if not isinstance(accumulation, int) or accumulation <= 0:
        raise ValueError("gradient accumulation must be a positive integer")
    if step != total_steps and step % accumulation:
        raise ValueError("resume step is not an optimizer boundary")

    history = checkpoint["history"]
    if not isinstance(history, list):
        raise TypeError("resume history must be a list")
    history_steps = [row.get("step") for row in history if isinstance(row, dict)]
    if len(history_steps) != len(history) or any(not isinstance(item, int) for item in history_steps):
        raise ValueError("every resume history row must contain an integer step")
    if history_steps != sorted(set(history_steps)) or (history_steps and history_steps[-1] > step):
        raise ValueError("resume history steps must be unique, increasing, and no later than the checkpoint")

    training_seconds = checkpoint["training_seconds"]
    if (
        isinstance(training_seconds, bool)
        or not isinstance(training_seconds, int | float)
        or not math.isfinite(training_seconds)
        or (step > 0 and training_seconds <= 0)
    ):
        raise ValueError("resume cumulative training time must be finite and positive after step zero")

    rng_state = checkpoint["torch_rng_state"]
    if not isinstance(rng_state, torch.Tensor) or rng_state.numel() == 0:
        raise ValueError("resume checkpoint does not contain a valid PyTorch RNG state")

    optimizer = checkpoint["optimizer"]
    if not isinstance(optimizer, dict):
        raise TypeError("resume optimizer state must be a mapping")
    optimizer_state = optimizer.get("state")
    param_groups = optimizer.get("param_groups")
    if not isinstance(optimizer_state, dict) or (step > 0 and not optimizer_state):
        raise ValueError("resume optimizer state is empty after training has started")
    if not isinstance(param_groups, list) or not param_groups:
        raise ValueError("resume optimizer parameter groups are missing")

    model_verification = _verify_model_state(target, checkpoint)
    return {
        **model_verification,
        "resume_step": step,
        "optimizer_boundary": True,
        "optimizer_state_entries": len(optimizer_state),
        "history_records": len(history),
        "training_seconds": float(training_seconds),
        "rng_state_bytes": rng_state.numel() * rng_state.element_size(),
        "config_matches": expected_config is None or config == expected_config,
        "training_complete": step == total_steps,
    }
