from __future__ import annotations

import json
import os
import platform
import random
from fcntl import LOCK_EX, LOCK_NB, flock
from pathlib import Path
from typing import TextIO

import numpy as np
import torch
import yaml


def load_config(path: str | Path) -> dict:
    with Path(path).open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def resolve_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def resolve_training_steps(training: dict, steps_per_epoch: int) -> int:
    """Resolve an explicit micro-step budget or fall back to whole epochs."""
    if "steps" in training:
        total = int(training["steps"])
    else:
        total = steps_per_epoch * int(training["epochs"])
    if total <= 0:
        raise ValueError("training steps must be positive")
    return total


def write_json(path: str | Path, payload: dict) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def acquire_run_lock(path: str | Path) -> TextIO:
    """Hold an advisory process lock until the returned file handle is closed."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    handle = target.open("a+", encoding="utf-8")
    try:
        flock(handle.fileno(), LOCK_EX | LOCK_NB)
    except BlockingIOError as error:
        handle.seek(0)
        owner = handle.read().strip() or "unknown"
        handle.close()
        raise RuntimeError(f"training run is already active (pid={owner}): {target}") from error
    handle.seek(0)
    handle.truncate()
    handle.write(str(os.getpid()))
    handle.flush()
    return handle


def environment_info(device: torch.device) -> dict[str, str]:
    return {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "device": str(device),
    }
