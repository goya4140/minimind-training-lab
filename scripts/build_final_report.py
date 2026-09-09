#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]

REQUIRED = {
    "LLM pretrain log": "artifacts/logs/llm-64m-pretrain-mps.json",
    "LLM SFT log": "artifacts/logs/llm-64m-sft-mps.json",
    "VLM alignment log": "artifacts/logs/vlm-vision-alignment-mps.json",
    "VLM SFT log": "artifacts/logs/vlm-sft-mps.json",
    "Video alignment log": "artifacts/logs/video-omni-alignment-mps.json",
    "Video SFT log": "artifacts/logs/video-omni-sft-mps.json",
    "LLM evaluation": "artifacts/eval/llm-sft-final.json",
    "VLM evaluation": "artifacts/eval/vlm-final.json",
    "Video evaluation": "artifacts/eval/video-omni-final.json",
    "LLM checkpoint": "artifacts/checkpoints/llm-64m-sft-mps.pt",
    "LLM pretrain checkpoint": "artifacts/checkpoints/llm-64m-pretrain-mps.pt",
    "VLM checkpoint": "artifacts/checkpoints/vlm-sft-mps.pt",
    "VLM alignment checkpoint": "artifacts/checkpoints/vlm-alignment-mps.pt",
    "Video checkpoint": "artifacts/checkpoints/video-omni-sft-mps.pt",
    "Video alignment checkpoint": "artifacts/checkpoints/video-omni-alignment-mps.pt",
    "QIVD manifest": "data/manifests/qivd.json",
}


def read_json(relative_path: str) -> dict:
    return json.loads((ROOT / relative_path).read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def duration(seconds: float) -> str:
    hours, remainder = divmod(round(seconds), 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:d}h {minutes:02d}m {seconds:02d}s"


def training_loss(checkpoint_path: str) -> tuple[float, float]:
    checkpoint = torch.load(ROOT / checkpoint_path, map_location="cpu", weights_only=False)
    history = checkpoint.get("history", [])
    if not history:
        raise RuntimeError(f"checkpoint has no training history: {checkpoint_path}")
    return float(history[0]["loss"]), float(history[-1]["loss"])


def release_entry(name: str, relative_path: str) -> dict:
    path = ROOT / relative_path
    return {"name": name, "path": relative_path, "bytes": path.stat().st_size, "sha256": sha256(path)}


def fmt(value: float) -> str:
    return f"{value:.4f}"


def one_line(value: object, limit: int = 180) -> str:
    text = " ".join(str(value).replace("|", "\\|").split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the evidence-backed final report and release manifest.")
    parser.add_argument("--check", action="store_true", help="Only check that every required artifact exists.")
    args = parser.parse_args()
    missing = {name: path for name, path in REQUIRED.items() if not (ROOT / path).is_file()}
    if missing:
        print(json.dumps({"ready": False, "missing": missing}, ensure_ascii=False, indent=2))
        sys.exit(1)
    if args.check:
        print(json.dumps({"ready": True, "artifacts": REQUIRED}, ensure_ascii=False, indent=2))
        return

    logs = {
        "llm_pretrain": read_json(REQUIRED["LLM pretrain log"]),
        "llm_sft": read_json(REQUIRED["LLM SFT log"]),
        "vlm_alignment": read_json(REQUIRED["VLM alignment log"]),
        "vlm_sft": read_json(REQUIRED["VLM SFT log"]),
        "video_alignment": read_json(REQUIRED["Video alignment log"]),
        "video_sft": read_json(REQUIRED["Video SFT log"]),
    }
    llm_eval = read_json(REQUIRED["LLM evaluation"])
    vlm_eval = read_json(REQUIRED["VLM evaluation"])
    video_eval = read_json(REQUIRED["Video evaluation"])
    qivd = read_json(REQUIRED["QIVD manifest"])
    release_assets = [
        release_entry("minimind-lab-llm-sft-mps.pt", REQUIRED["LLM checkpoint"]),
        release_entry("minimind-lab-vlm-sft-mps.pt", REQUIRED["VLM checkpoint"]),
        release_entry("minimind-lab-video-omni-sft-mps.pt", REQUIRED["Video checkpoint"]),
        release_entry("llm-evaluation.json", REQUIRED["LLM evaluation"]),
        release_entry("vlm-evaluation.json", REQUIRED["VLM evaluation"]),
        release_entry("video-omni-evaluation.json", REQUIRED["Video evaluation"]),
        release_entry("qivd-manifest.json", REQUIRED["QIVD manifest"]),
    ]
    losses = {
        key: training_loss(REQUIRED[checkpoint_name])
        for key, checkpoint_name in (
            ("llm_pretrain", "LLM pretrain checkpoint"),
            ("llm_sft", "LLM checkpoint"),
            ("vlm_alignment", "VLM alignment checkpoint"),
            ("vlm_sft", "VLM checkpoint"),
            ("video_alignment", "Video alignment checkpoint"),
            ("video_sft", "Video checkpoint"),
        )
    }
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    qivd_generation = video_eval["qivd_generation"]
    temporal = video_eval["controlled_temporal"]
    qualitative_vlm = vlm_eval.get("qualitative", [])
    vlm_keyword_recall = [item["keyword_recall"] for item in qualitative_vlm if item.get("keyword_recall") is not None]
    total_training_seconds = sum(float(log.get("training_seconds", 0)) for log in logs.values())
    lines = [
        "# MiniMind Training Lab — Final Results",
        "",
        (
            f"Evidence commit: `{commit}`. Hardware: Apple M4 Pro / MPS. Total recorded stage time: "
            f"**{duration(total_training_seconds)}**."
        ),
        "",
        "## Outcome",
        "",
        (
            "This run trains one language core from random initialization, then reuses its instruction-tuned "
            "checkpoint for image and video understanding. SigLIP2 is explicitly frozen and is not claimed as "
            "from-scratch training."
        ),
        "",
        "## Training",
        "",
        "| Stage | Steps | Trainable parameters | First logged loss | Last logged loss | Time |",
        "|---|---:|---:|---:|---:|---:|",
        f"| LLM pretrain | {logs['llm_pretrain']['total_steps']:,} | 63,912,192 | {fmt(losses['llm_pretrain'][0])} | {fmt(losses['llm_pretrain'][1])} | {duration(logs['llm_pretrain']['training_seconds'])} |",
        f"| LLM SFT | {logs['llm_sft']['total_steps']:,} | 63,912,192 | {fmt(losses['llm_sft'][0])} | {fmt(losses['llm_sft'][1])} | {duration(logs['llm_sft']['training_seconds'])} |",
        f"| VLM alignment | {logs['vlm_alignment']['total_steps']:,} | {logs['vlm_alignment']['trainable_parameters']:,} | {fmt(losses['vlm_alignment'][0])} | {fmt(losses['vlm_alignment'][1])} | {duration(logs['vlm_alignment']['training_seconds'])} |",
        f"| VLM SFT | {logs['vlm_sft']['total_steps']:,} | {logs['vlm_sft']['trainable_parameters']:,} | {fmt(losses['vlm_sft'][0])} | {fmt(losses['vlm_sft'][1])} | {duration(logs['vlm_sft']['training_seconds'])} |",
        f"| Video alignment | {logs['video_alignment']['total_steps']:,} | {logs['video_alignment']['trainable_parameters']:,} | {fmt(losses['video_alignment'][0])} | {fmt(losses['video_alignment'][1])} | {duration(logs['video_alignment']['training_seconds'])} |",
        f"| Video SFT | {logs['video_sft']['total_steps']:,} | {logs['video_sft']['trainable_parameters']:,} | {fmt(losses['video_sft'][0])} | {fmt(losses['video_sft'][1])} | {duration(logs['video_sft']['training_seconds'])} |",
        "",
        "## Evaluation",
        "",
        "| Model / set | Primary metrics |",
        "|---|---|",
        f"| LLM held-out text | loss {fmt(llm_eval['corpus']['validation_loss'])}; perplexity {fmt(llm_eval['corpus']['validation_perplexity'])}; BPB {fmt(llm_eval['corpus']['bits_per_byte'])} |",
        f"| VLM validation + fixed images | loss {fmt(vlm_eval['validation_loss'])}; mean keyword recall {fmt(sum(vlm_keyword_recall) / len(vlm_keyword_recall)) if vlm_keyword_recall else 'n/a'} |",
        f"| Video QIVD held-out | loss {fmt(video_eval['test_loss'])}; exact {fmt(qivd_generation['normalized_exact_match'])}; token F1 {fmt(qivd_generation['token_f1'])} |",
        f"| Controlled temporal | exact {fmt(temporal['normalized_exact_match'])}; reversed exact {fmt(temporal['reversed_frame_exact_match'])}; token F1 delta {fmt(temporal['normal_minus_reversed_token_f1'])} |",
        "",
        (
            "A positive controlled temporal normal-minus-reversed result is evidence that the model uses frame "
            "order; a zero or negative result must be interpreted as failure to establish temporal sensitivity."
        ),
        "",
        "## Data and release",
        "",
        (
            f"QIVD: {qivd['video_count']:,} videos, pinned revision `{qivd['revision']}`, aggregate SHA-256 "
            f"`{qivd['aggregate_sha256']}`. QIVD is research-only and is not redistributed."
        ),
        "",
        "| Release asset | Bytes | SHA-256 |",
        "|---|---:|---|",
        *[f"| `{item['name']}` | {item['bytes']:,} | `{item['sha256']}` |" for item in release_assets],
        "",
        "## Fixed qualitative examples",
        "",
        "### LLM",
        "",
        "| Prompt | Completion |",
        "|---|---|",
        *[
            f"| {one_line(item['prompt'])} | {one_line(item['completion'])} |"
            for item in llm_eval.get("generation", [])[:4]
        ],
        "",
        "### VLM",
        "",
        "| Prompt | Completion |",
        "|---|---|",
        *[
            f"| {one_line(item.get('prompt', item.get('id', '')))} | {one_line(item['completion'])} |"
            for item in qualitative_vlm[:6]
        ],
        "",
        "### Video-Omni — QIVD",
        "",
        "| Question | Reference | Normal frames | Reversed frames |",
        "|---|---|---|---|",
        *[
            f"| {one_line(item['question'])} | {one_line(item['answer'])} | "
            f"{one_line(item['normal_completion'])} | {one_line(item['reversed_completion'])} |"
            for item in qivd_generation.get("qualitative", [])[:6]
        ],
        "",
        "### Video-Omni — controlled temporal",
        "",
        "| Category | Question | Reference | Normal frames | Reversed frames |",
        "|---|---|---|---|---|",
        *[
            f"| {one_line(item['category'])} | {one_line(item['question'])} | {one_line(item['answer'])} | "
            f"{one_line(item['normal_completion'])} | {one_line(item['reversed_completion'])} |"
            for item in temporal.get("qualitative", [])[:8]
        ],
        "",
        (
            "See `docs/model-cards/` for intended use and limitations, `docs/evaluation.md` for the protocol, "
            "and the JSON evaluation artifacts attached to the release for full qualitative outputs."
        ),
        "",
    ]
    report_path = ROOT / "reports/final-results.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    manifest = {"tag": "v0.1.0-trained-mps", "commit": commit, "assets": release_assets}
    (ROOT / "reports/release-manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"report": str(report_path), "release_manifest": manifest}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
