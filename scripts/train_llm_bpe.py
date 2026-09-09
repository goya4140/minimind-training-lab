#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from minimind_lab.llm import MiniMindConfig, MiniMindForCausalLM
from minimind_lab.training import load_config, resolve_device, seed_everything
from minimind_lab.training.utils import environment_info, write_json


def load_tokens(path: Path, tokenizer, repeats: int) -> torch.Tensor:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    token_ids: list[int] = []
    for _ in range(repeats):
        for row in rows:
            encoded = tokenizer.encode(row["text"], add_special_tokens=False)
            token_ids.extend([tokenizer.bos_token_id, *encoded, tokenizer.eos_token_id])
    return torch.tensor(token_ids, dtype=torch.long)


def random_batches(tokens: torch.Tensor, batch_size: int, length: int, device: torch.device):
    max_start = tokens.numel() - length - 1
    if max_start <= 0:
        raise ValueError("dataset must contain more tokens than sequence length")
    while True:
        starts = torch.randint(0, max_start, (batch_size,))
        yield torch.stack([tokens[start : start + length] for start in starts]).to(device)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    seed_everything(config["experiment"]["seed"])
    device = resolve_device(config["experiment"]["device"])
    tokenizer = AutoTokenizer.from_pretrained(ROOT / config["tokenizer"]["path"], local_files_only=True)
    model_config = MiniMindConfig(**config["model"])
    if model_config.vocab_size != len(tokenizer):
        raise ValueError(f"model vocab {model_config.vocab_size} != tokenizer vocab {len(tokenizer)}")
    tokens = load_tokens(ROOT / config["data"]["path"], tokenizer, config["data"]["repeats"])
    model = MiniMindForCausalLM(model_config).to(device)
    training = config["training"]
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=training["learning_rate"], weight_decay=training["weight_decay"]
    )
    batches = random_batches(tokens, training["batch_size"], config["data"]["sequence_length"], device)
    history = []
    started = time.time()
    model.train()
    for step in range(1, training["steps"] + 1):
        input_ids = next(batches)
        loss = model(input_ids, labels=input_ids)["loss"]
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), training["grad_clip"])
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        if step == 1 or step % training["log_interval"] == 0:
            record = {"step": step, "loss": loss.detach().item(), "grad_norm": grad_norm.detach().item()}
            history.append(record)
            print(json.dumps(record, ensure_ascii=False), flush=True)
    checkpoint_path = ROOT / training["checkpoint_path"]
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "config": config, "history": history}, checkpoint_path)
    report = {
        "experiment": config["experiment"]["name"],
        "status": "mini-test-only",
        "parameters": model.parameter_count(),
        "training_tokens_available": tokens.numel(),
        "elapsed_seconds": round(time.time() - started, 3),
        "first_loss": history[0]["loss"],
        "final_loss": history[-1]["loss"],
        "environment": environment_info(device),
        "checkpoint": str(checkpoint_path.relative_to(ROOT)),
    }
    write_json(ROOT / "artifacts/logs/llm-bpe-mini.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
