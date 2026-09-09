#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import Subset
from transformers import AutoTokenizer, SiglipImageProcessor, SiglipVisionModel

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from minimind_lab.data import ParquetVLMDataset, collate_vlm
from minimind_lab.evaluation import distinct_n, evaluate_language_regression
from minimind_lab.training import load_config, resolve_device, seed_everything
from minimind_lab.training.utils import environment_info, write_json
from minimind_lab.vlm import MiniMindVLM, VLMConfig


@torch.inference_mode()
def validation_loss(model, dataset, batch_size: int, device: torch.device) -> float:
    model.eval()
    losses = []
    for start in range(0, len(dataset), batch_size):
        samples = [dataset[index] for index in range(start, min(start + batch_size, len(dataset)))]
        input_ids, labels, pixels, counts = collate_vlm(samples)
        loss = model(
            input_ids.to(device),
            pixels.to(device),
            labels.to(device),
            image_counts=counts.to(device),
        )["loss"]
        if torch.isfinite(loss):
            losses.append(float(loss.item()))
    if not losses:
        raise RuntimeError("VLM evaluation produced no finite losses")
    return sum(losses) / len(losses)


@torch.inference_mode()
def qualitative_samples(model, tokenizer, processor, device, max_new_tokens: int) -> list[dict]:
    eval_dir = ROOT / "data/eval/vlm"
    cases = json.loads((eval_dir / "prompts.json").read_text(encoding="utf-8"))
    results = []
    from PIL import Image

    for case in cases:
        image_names = case.get("images") or [case["image"]]
        if not isinstance(image_names, list) or not image_names:
            raise ValueError(f"VLM evaluation case has no images: {case}")
        placeholders = "\n".join("<image>" for _ in image_names)
        content = f"{placeholders}\n{case['prompt']}".replace(
            "<image>", "<|image_pad|>" * model.config.image_token_length
        )
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": content}], tokenize=False, add_generation_prompt=True
        )
        prompt_ids = tokenizer(prompt).input_ids
        input_ids = torch.tensor([prompt_ids], device=device)
        images = [Image.open(eval_dir / image_name).convert("RGB") for image_name in image_names]
        pixels = processor(images=images, return_tensors="pt")["pixel_values"].unsqueeze(0)
        started = time.perf_counter()
        generated = model.generate(
            input_ids,
            pixels.to(device),
            image_counts=torch.tensor([len(images)], device=device),
            max_new_tokens=max_new_tokens,
            temperature=0.0,
            eos_token_id=tokenizer.eos_token_id,
        )[0]
        elapsed = time.perf_counter() - started
        completion_ids = generated[len(prompt_ids) :].tolist()
        completion = tokenizer.decode(completion_ids, skip_special_tokens=True)
        keywords = case.get("keywords", [])
        results.append(
            {
                **case,
                "image_count": len(images),
                "completion": completion,
                "keyword_recall": sum(keyword in completion for keyword in keywords) / len(keywords)
                if keywords
                else None,
                "new_tokens": len(completion_ids),
                "seconds": elapsed,
                "tokens_per_second": len(completion_ids) / elapsed,
                "distinct_2": distinct_n(completion_ids),
            }
        )
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--validation-samples", type=int, default=128)
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--language-config", default="configs/llm/pretrain-mps.yaml")
    parser.add_argument("--language-validation-samples", type=int, default=2048)
    parser.add_argument("--output", default="artifacts/eval/vlm.json")
    args = parser.parse_args()
    config = load_config(args.config)
    seed_everything(config["experiment"]["seed"])
    device = resolve_device(config["experiment"]["device"])
    model_values = dict(config["model"])
    model_values.pop("alignment_checkpoint", None)
    model_values.pop("language_checkpoint", None)
    vision_path = ROOT / model_values.pop("vision_encoder")
    tokenizer = AutoTokenizer.from_pretrained(ROOT / config["tokenizer"]["path"], local_files_only=True)
    processor = SiglipImageProcessor.from_pretrained(vision_path, local_files_only=True)
    vision = SiglipVisionModel.from_pretrained(vision_path, local_files_only=True)
    model = MiniMindVLM(VLMConfig(**model_values), vision).to(device)
    state = torch.load(ROOT / args.checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(state["model"], strict=False)

    full_dataset = ParquetVLMDataset(
        ROOT / config["data"]["path"],
        tokenizer,
        processor,
        sequence_length=config["data"]["sequence_length"],
        image_token_length=model.config.image_token_length,
    )
    validation = Subset(full_dataset, range(len(full_dataset) - args.validation_samples, len(full_dataset)))
    language_config = load_config(ROOT / args.language_config)
    report = {
        "experiment": config["experiment"]["name"],
        "checkpoint": args.checkpoint,
        "environment": environment_info(device),
        "validation_loss": validation_loss(model, validation, config["training"]["batch_size"], device),
        "qualitative": qualitative_samples(model, tokenizer, processor, device, args.max_new_tokens),
        "language_regression": evaluate_language_regression(
            model.language_model,
            tokenizer,
            ROOT / language_config["data"]["path"],
            sequence_length=language_config["data"]["sequence_length"],
            validation_samples=args.language_validation_samples,
            batch_size=language_config["training"]["batch_size"],
            device=device,
            max_new_tokens=args.max_new_tokens,
        ),
    }
    write_json(ROOT / args.output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
