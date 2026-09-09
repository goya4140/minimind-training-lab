from __future__ import annotations

import hashlib
from pathlib import Path

import torch


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_checkpoint(path: str | Path) -> dict[str, int | str | bool]:
    target = Path(path)
    checkpoint = torch.load(target, map_location="cpu", weights_only=False)
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
