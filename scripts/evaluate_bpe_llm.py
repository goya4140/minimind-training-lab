#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from minimind_lab.data import JsonlPretrainDataset
from minimind_lab.evaluation import LANGUAGE_PROMPTS, language_corpus_metrics, language_generation_samples
from minimind_lab.llm import MiniMindConfig, MiniMindForCausalLM
from minimind_lab.training import load_config, resolve_device, seed_everything
from minimind_lab.training.utils import environment_info, write_json


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
        "corpus": language_corpus_metrics(model, loader, tokenizer, device),
        "generation": language_generation_samples(
            model, tokenizer, device, args.max_new_tokens, prompts=LANGUAGE_PROMPTS
        ),
    }
    write_json(ROOT / args.output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
