#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from minimind_lab.data import JsonlPretrainDataset
from minimind_lab.evaluation import distinct_n
from minimind_lab.llm import MiniMindConfig, MiniMindForCausalLM
from minimind_lab.training import load_config, resolve_device, seed_everything
from minimind_lab.training.utils import environment_info, write_json

PROMPTS = [
    "中国的首都是",
    "请用三句话解释什么是机器学习：",
    "用户：如何制定一个可执行的学习计划？\n助手：",
    "Transformer 模型中的注意力机制",
]


@torch.inference_mode()
def corpus_metrics(model, loader, tokenizer, device: torch.device) -> dict[str, float | int]:
    model.eval()
    negative_log_likelihood = 0.0
    predicted_tokens = 0
    utf8_bytes = 0
    started = time.perf_counter()
    special_ids = set(tokenizer.all_special_ids)
    for input_ids, labels in loader:
        input_ids = input_ids.to(device)
        labels = labels.to(device)
        loss = model(input_ids, labels)["loss"]
        valid = labels[:, 1:] != -100
        count = int(valid.sum().item())
        negative_log_likelihood += float(loss.item()) * count
        predicted_tokens += count
        for row in labels[:, 1:].cpu().tolist():
            content_ids = [token for token in row if token != -100 and token not in special_ids]
            utf8_bytes += len(tokenizer.decode(content_ids, skip_special_tokens=True).encode("utf-8"))
    elapsed = time.perf_counter() - started
    mean_loss = negative_log_likelihood / predicted_tokens
    return {
        "validation_loss": mean_loss,
        "validation_perplexity": math.exp(min(mean_loss, 20)),
        "bits_per_byte": negative_log_likelihood / (math.log(2) * utf8_bytes),
        "predicted_tokens": predicted_tokens,
        "utf8_bytes": utf8_bytes,
        "validation_tokens_per_second": predicted_tokens / elapsed,
    }


@torch.inference_mode()
def generation_samples(model, tokenizer, device: torch.device, max_new_tokens: int) -> list[dict]:
    samples = []
    for prompt in PROMPTS:
        prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
        inputs = torch.tensor([[tokenizer.bos_token_id, *prompt_ids]], device=device)
        started = time.perf_counter()
        generated = model.generate(inputs, max_new_tokens=max_new_tokens, temperature=0.0)[0]
        elapsed = time.perf_counter() - started
        completion_ids = generated[inputs.size(1) :].tolist()
        samples.append(
            {
                "prompt": prompt,
                "completion": tokenizer.decode(completion_ids, skip_special_tokens=True),
                "new_tokens": len(completion_ids),
                "seconds": elapsed,
                "tokens_per_second": len(completion_ids) / elapsed,
                "distinct_2": distinct_n(completion_ids, n=2),
            }
        )
    return samples


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--validation-samples", type=int, default=2048)
    parser.add_argument("--max-new-tokens", type=int, default=80)
    parser.add_argument("--device", default=None)
    parser.add_argument("--output", default="artifacts/eval/llm-bpe.json")
    args = parser.parse_args()

    config = load_config(args.config)
    seed_everything(config["experiment"]["seed"])
    device = resolve_device(args.device or config["experiment"]["device"])
    tokenizer = AutoTokenizer.from_pretrained(ROOT / config["tokenizer"]["path"], local_files_only=True)
    dataset = JsonlPretrainDataset(
        ROOT / config["data"]["path"], tokenizer, sequence_length=config["data"]["sequence_length"]
    )
    if not 0 < args.validation_samples < len(dataset):
        raise ValueError("validation-samples must be positive and smaller than the dataset")
    validation = Subset(dataset, range(len(dataset) - args.validation_samples, len(dataset)))
    loader = DataLoader(validation, batch_size=config["training"]["batch_size"], num_workers=0)

    model = MiniMindForCausalLM(MiniMindConfig(**config["model"])).to(device)
    state = torch.load(ROOT / args.checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(state["model"])
    report = {
        "experiment": config["experiment"]["name"],
        "checkpoint": args.checkpoint,
        "parameters": model.parameter_count(),
        "environment": environment_info(device),
        "corpus": corpus_metrics(model, loader, tokenizer, device),
        "generation": generation_samples(model, tokenizer, device, args.max_new_tokens),
    }
    write_json(ROOT / args.output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
