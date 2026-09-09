#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from minimind_lab.llm import ByteTokenizer, MiniMindConfig, MiniMindForCausalLM
from minimind_lab.training import load_config, resolve_device
from minimind_lab.training.utils import write_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=80)
    args = parser.parse_args()
    config = load_config(args.config)
    device = resolve_device(config["experiment"]["device"])
    tokenizer = ByteTokenizer()
    model = MiniMindForCausalLM(MiniMindConfig(**config["model"])).to(device)
    state = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(state["model"])
    prompts = ["语言模型", "Transformer", "Question: 什么是训练？ Answer:"]
    samples = []
    torch.manual_seed(config["experiment"]["seed"])
    for prompt in prompts:
        inputs = torch.tensor([tokenizer.encode(prompt)], device=device)
        generated = model.generate(inputs, args.max_new_tokens, temperature=0.7)[0].tolist()
        samples.append({"prompt": prompt, "completion": tokenizer.decode(generated[len(inputs[0]) :])})
    report = {"status": "smoke-test-only", "checkpoint": args.checkpoint, "samples": samples}
    write_json(ROOT / "artifacts/eval/llm-smoke.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
