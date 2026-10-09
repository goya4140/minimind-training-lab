#!/usr/bin/env python3
"""Extract a model-only weight snapshot from a running resume checkpoint.

The resume checkpoint is continuously overwritten by the trainer, so mid-training
diagnostics need an archived snapshot that will not be replaced. This copies just
the model state_dict into a standalone file and reports its size and SHA-256.

Reads with map_location=cpu so it never competes with MPS training for GPU memory.
The source file is written atomically by the trainer, so a concurrent read always
sees a complete checkpoint.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 22), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, help="resume checkpoint, relative to repo root")
    parser.add_argument("--target", required=True, help="output snapshot path, relative to repo root")
    args = parser.parse_args()

    source = ROOT / args.source
    target = ROOT / args.target

    state = torch.load(source, map_location="cpu", weights_only=False)
    model = state["model"]
    target.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model}, target)

    parameters = sum(tensor.numel() for tensor in model.values())
    report = {
        "source": str(source),
        "target": str(target),
        "source_step": state.get("step"),
        "history_records": len(state.get("history", []) or []),
        "state_dict_parameters": parameters,
        "all_finite": all(bool(torch.isfinite(tensor).all()) for tensor in model.values()),
        "bytes": target.stat().st_size,
        "sha256": sha256(target),
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
