#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import torch
from transformers import AutoTokenizer, SiglipImageProcessor, SiglipVisionModel

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from minimind_lab.data import QIVDVideoDataset, collate_video, decode_uniform_video
from minimind_lab.data.temporal_benchmark import generate_temporal_benchmark
from minimind_lab.evaluation import evaluate_language_regression, normalize_qa_answer, token_f1
from minimind_lab.training import load_config, resolve_device, seed_everything
from minimind_lab.training.utils import environment_info, write_json
from minimind_lab.video import MiniMindVideoOmni, VideoOmniConfig


@torch.inference_mode()
def loss_ablation(model, dataset, batch_size: int, device: torch.device) -> tuple[float, float]:
    normal, reversed_order = [], []
    for start in range(0, len(dataset), batch_size):
        samples = [dataset[index] for index in range(start, min(start + batch_size, len(dataset)))]
        input_ids, labels, pixels = collate_video(samples)
        input_ids, labels, pixels = input_ids.to(device), labels.to(device), pixels.to(device)
        normal.append(float(model(input_ids, pixels, labels)["loss"].item()))
        reversed_order.append(float(model(input_ids, pixels.flip(1), labels)["loss"].item()))
    return sum(normal) / len(normal), sum(reversed_order) / len(reversed_order)


@torch.inference_mode()
def generate_cases(model, dataset, tokenizer, processor, device, count: int, max_new_tokens: int) -> list[dict]:
    results = []
    for index in range(min(count, len(dataset))):
        row = dataset.metadata(index)
        prompt = tokenizer.apply_chat_template(
            [
                {
                    "role": "user",
                    "content": f"{'<|video_pad|>' * model.config.num_video_tokens}\n{row['question']}",
                }
            ],
            tokenize=False,
            add_generation_prompt=True,
        )
        prompt_ids = tokenizer(prompt).input_ids
        input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
        frames = decode_uniform_video(dataset.root / row["video_file_name"], model.config.num_frames)
        pixels = processor(images=frames, return_tensors="pt")["pixel_values"].unsqueeze(0).to(device)

        outputs = {}
        elapsed = {}
        for label, frame_tensor in (("normal", pixels), ("reversed", pixels.flip(1))):
            started = time.perf_counter()
            generated = model.generate(
                input_ids,
                frame_tensor,
                max_new_tokens=max_new_tokens,
                temperature=0.0,
                eos_token_id=tokenizer.eos_token_id,
            )[0]
            elapsed[label] = time.perf_counter() - started
            outputs[label] = tokenizer.decode(generated[len(prompt_ids) :].tolist(), skip_special_tokens=True).strip()

        short_answer = str(row.get("short_answer") or "")
        reference = (
            str(row["answer"]) if normalize_qa_answer(short_answer) in {"", "na", "n a"} else short_answer
        )
        normalized_reference = normalize_qa_answer(reference)
        normal_prediction = normalize_qa_answer(outputs["normal"])
        reversed_prediction = normalize_qa_answer(outputs["reversed"])
        results.append(
            {
                "id": row["id"],
                "category": row["category"],
                "question": row["question"],
                "answer": row["answer"],
                "short_answer": row["short_answer"],
                "normal_completion": outputs["normal"],
                "reversed_completion": outputs["reversed"],
                "normal_exact_match": normal_prediction == normalized_reference,
                "reversed_exact_match": reversed_prediction == normalized_reference,
                "normal_contains_reference": normalized_reference in normal_prediction,
                "reversed_contains_reference": normalized_reference in reversed_prediction,
                "normal_token_f1": token_f1(outputs["normal"], reference),
                "reversed_token_f1": token_f1(outputs["reversed"], reference),
                "completion_changed_when_reversed": normal_prediction != reversed_prediction,
                "normal_seconds": elapsed["normal"],
            }
        )
    return results


def mean(items: list[float]) -> float:
    return sum(items) / len(items) if items else 0.0


def generation_summary(cases: list[dict]) -> dict:
    by_category = defaultdict(list)
    for case in cases:
        by_category[case["category"]].append(case["normal_token_f1"])
    return {
        "generated_samples": len(cases),
        "normalized_exact_match": mean([float(case["normal_exact_match"]) for case in cases]),
        "reversed_frame_exact_match": mean([float(case["reversed_exact_match"]) for case in cases]),
        "contains_reference": mean([float(case["normal_contains_reference"]) for case in cases]),
        "reversed_frame_contains_reference": mean(
            [float(case["reversed_contains_reference"]) for case in cases]
        ),
        "token_f1": mean([case["normal_token_f1"] for case in cases]),
        "reversed_frame_token_f1": mean([case["reversed_token_f1"] for case in cases]),
        "normal_minus_reversed_token_f1": mean([case["normal_token_f1"] for case in cases])
        - mean([case["reversed_token_f1"] for case in cases]),
        "completion_change_rate_on_reversal": mean(
            [float(case["completion_changed_when_reversed"]) for case in cases]
        ),
        "mean_generation_seconds": mean([case["normal_seconds"] for case in cases]),
        "token_f1_by_category": {key: mean(values) for key, values in sorted(by_category.items())},
        "qualitative": cases[:12],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--generation-samples", type=int, default=100)
    parser.add_argument("--temporal-generation-samples", type=int, default=160)
    parser.add_argument("--max-new-tokens", type=int, default=48)
    parser.add_argument("--language-config", default="configs/llm/pretrain-mps.yaml")
    parser.add_argument("--language-validation-samples", type=int, default=2048)
    parser.add_argument("--output", default="artifacts/eval/video-omni-final.json")
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
    model = MiniMindVideoOmni(VideoOmniConfig(**model_values), vision).to(device)
    checkpoint = torch.load(ROOT / args.checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model"], strict=False)
    model.eval()

    data = config["data"]
    dataset = QIVDVideoDataset(
        ROOT / data["path"],
        tokenizer,
        processor,
        sequence_length=data["sequence_length"],
        num_frames=model.config.num_frames,
        num_video_tokens=model.config.num_video_tokens,
        split="test",
        split_seed=data.get("split_seed", 48),
        train_samples=data.get("train_samples", 2400),
        validation_samples=data.get("validation_samples", 250),
    )
    normal_loss, reversed_loss = loss_ablation(model, dataset, config["training"]["batch_size"], device)
    cases = generate_cases(
        model, dataset, tokenizer, processor, device, args.generation_samples, args.max_new_tokens
    )

    temporal_root = ROOT / "data/eval/temporal-video"
    temporal_manifest = generate_temporal_benchmark(temporal_root)
    temporal_dataset = QIVDVideoDataset(
        temporal_root,
        tokenizer,
        processor,
        sequence_length=data["sequence_length"],
        num_frames=model.config.num_frames,
        num_video_tokens=model.config.num_video_tokens,
        split="test",
        split_seed=20260909,
        train_samples=0,
        validation_samples=0,
    )
    temporal_loss, temporal_reversed_loss = loss_ablation(
        model, temporal_dataset, config["training"]["batch_size"], device
    )
    temporal_cases = generate_cases(
        model,
        temporal_dataset,
        tokenizer,
        processor,
        device,
        args.temporal_generation_samples,
        args.max_new_tokens,
    )
    language_config = load_config(ROOT / args.language_config)
    report = {
        "experiment": config["experiment"]["name"],
        "checkpoint": args.checkpoint,
        "environment": environment_info(device),
        "held_out_test_samples": len(dataset),
        "test_loss": normal_loss,
        "reversed_frame_test_loss": reversed_loss,
        "reversed_minus_normal_loss": reversed_loss - normal_loss,
        "qivd_generation": generation_summary(cases),
        "controlled_temporal": {
            "manifest": temporal_manifest,
            "test_loss": temporal_loss,
            "reversed_frame_test_loss": temporal_reversed_loss,
            "reversed_minus_normal_loss": temporal_reversed_loss - temporal_loss,
            **generation_summary(temporal_cases),
        },
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
