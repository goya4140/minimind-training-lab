#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from minimind_lab.llm import MiniMindConfig, MiniMindForCausalLM
from minimind_lab.training import resolve_device
from minimind_lab.training.utils import write_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=80)
    args = parser.parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    device = resolve_device(config["experiment"]["device"])
    tokenizer = AutoTokenizer.from_pretrained(ROOT / config["tokenizer"]["path"], local_files_only=True)
    model = MiniMindForCausalLM(MiniMindConfig(**config["model"])).to(device)
    model.load_state_dict(checkpoint["model"])
    prompts = ["语言模型", "Transformer", "Question: 什么是训练？ Answer:"]
    samples = []
    torch.manual_seed(config["experiment"]["seed"])
    for prompt in prompts:
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
        generated = model.generate(input_ids, args.max_new_tokens, temperature=0.7)[0, input_ids.size(1) :]
        samples.append({"prompt": prompt, "completion": tokenizer.decode(generated, skip_special_tokens=True)})
    report = {"status": "mini-test-only", "checkpoint": args.checkpoint, "samples": samples}
    write_json(ROOT / "artifacts/eval/llm-bpe-mini.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
