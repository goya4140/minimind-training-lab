#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from minimind_lab.reporting import render_temporal_ablation, render_training_curves, validate_final_evaluations

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


def training_history(checkpoint_path: str) -> list[dict]:
    checkpoint = torch.load(ROOT / checkpoint_path, map_location="cpu", weights_only=False)
    history = checkpoint.get("history", [])
    if not history:
        raise RuntimeError(f"checkpoint has no training history: {checkpoint_path}")
    return history


def local_artifact_entry(name: str, relative_path: str) -> dict:
    path = ROOT / relative_path
    return {"name": name, "path": relative_path, "bytes": path.stat().st_size, "sha256": sha256(path)}


def fmt(value: float) -> str:
    return f"{value:.4f}"


def one_line(value: object, limit: int = 180) -> str:
    text = " ".join(str(value).replace("|", "\\|").split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the evidence-backed final handbook report.")
    parser.add_argument("--check", action="store_true", help="Only check that every required artifact exists.")
    args = parser.parse_args()
    missing = {name: path for name, path in REQUIRED.items() if not (ROOT / path).is_file()}
    if missing:
        print(json.dumps({"ready": False, "missing": missing}, ensure_ascii=False, indent=2))
        sys.exit(1)

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
    validate_final_evaluations(llm_eval, vlm_eval, video_eval)
    qivd = read_json(REQUIRED["QIVD manifest"])
    local_artifacts = [
        local_artifact_entry("llm-64m-sft-mps.pt", REQUIRED["LLM checkpoint"]),
        local_artifact_entry("vlm-sft-mps.pt", REQUIRED["VLM checkpoint"]),
        local_artifact_entry("video-omni-sft-mps.pt", REQUIRED["Video checkpoint"]),
        local_artifact_entry("llm-sft-final.json", REQUIRED["LLM evaluation"]),
        local_artifact_entry("vlm-final.json", REQUIRED["VLM evaluation"]),
        local_artifact_entry("video-omni-final.json", REQUIRED["Video evaluation"]),
        local_artifact_entry("qivd.json", REQUIRED["QIVD manifest"]),
    ]
    histories = {
        key: training_history(REQUIRED[checkpoint_name])
        for key, checkpoint_name in (
            ("llm_pretrain", "LLM pretrain checkpoint"),
            ("llm_sft", "LLM checkpoint"),
            ("vlm_alignment", "VLM alignment checkpoint"),
            ("vlm_sft", "VLM checkpoint"),
            ("video_alignment", "Video alignment checkpoint"),
            ("video_sft", "Video checkpoint"),
        )
    }
    if args.check:
        print(json.dumps({"ready": True, "artifacts": REQUIRED}, ensure_ascii=False, indent=2))
        return
    losses = {key: (float(history[0]["loss"]), float(history[-1]["loss"])) for key, history in histories.items()}
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    qivd_generation = video_eval["qivd_generation"]
    temporal = video_eval["controlled_temporal"]
    llm_language = llm_eval["corpus"]
    vlm_language = vlm_eval["language_regression"]["corpus"]
    video_language = video_eval["language_regression"]["corpus"]
    qualitative_vlm = vlm_eval.get("qualitative", [])
    visual_ablation = vlm_eval["visual_ablation"]
    total_training_seconds = sum(float(log.get("training_seconds", 0)) for log in logs.values())
    assets = ROOT / "reports/assets"
    render_training_curves(
        {
            "LLM pretrain": histories["llm_pretrain"],
            "LLM SFT": histories["llm_sft"],
            "VLM alignment": histories["vlm_alignment"],
            "VLM SFT": histories["vlm_sft"],
            "Video alignment": histories["video_alignment"],
            "Video SFT": histories["video_sft"],
        },
        assets / "training-curves.svg",
    )
    render_temporal_ablation(temporal, assets / "temporal-ablation.svg")
    lines = [
        "# MiniMind Training Lab — 最终训练与评估结果",
        "",
        (
            f"证据对应提交：`{commit}`。硬件：Apple M4 Pro / MPS。六阶段记录的训练总耗时："
            f"**{duration(total_training_seconds)}**。"
        ),
        "",
        "## 结论",
        "",
        (
            "本次实验从随机初始化训练一个语言核心，再从其指令微调 checkpoint 分别扩展图像与视频理解。"
            "SigLIP2 始终冻结，因此不把视觉编码器描述为从零训练。"
        ),
        "",
        "## 训练结果",
        "",
        "| 阶段 | Micro-steps | 可训练参数 | 首次记录 loss | 最后记录 loss | 耗时 |",
        "|---|---:|---:|---:|---:|---:|",
        f"| LLM 预训练 | {logs['llm_pretrain']['total_steps']:,} | 63,912,192 | {fmt(losses['llm_pretrain'][0])} | {fmt(losses['llm_pretrain'][1])} | {duration(logs['llm_pretrain']['training_seconds'])} |",
        f"| LLM SFT | {logs['llm_sft']['total_steps']:,} | 63,912,192 | {fmt(losses['llm_sft'][0])} | {fmt(losses['llm_sft'][1])} | {duration(logs['llm_sft']['training_seconds'])} |",
        f"| VLM 对齐 | {logs['vlm_alignment']['total_steps']:,} | {logs['vlm_alignment']['trainable_parameters']:,} | {fmt(losses['vlm_alignment'][0])} | {fmt(losses['vlm_alignment'][1])} | {duration(logs['vlm_alignment']['training_seconds'])} |",
        f"| VLM SFT | {logs['vlm_sft']['total_steps']:,} | {logs['vlm_sft']['trainable_parameters']:,} | {fmt(losses['vlm_sft'][0])} | {fmt(losses['vlm_sft'][1])} | {duration(logs['vlm_sft']['training_seconds'])} |",
        f"| Video-Omni 对齐 | {logs['video_alignment']['total_steps']:,} | {logs['video_alignment']['trainable_parameters']:,} | {fmt(losses['video_alignment'][0])} | {fmt(losses['video_alignment'][1])} | {duration(logs['video_alignment']['training_seconds'])} |",
        f"| Video-Omni SFT | {logs['video_sft']['total_steps']:,} | {logs['video_sft']['trainable_parameters']:,} | {fmt(losses['video_sft'][0])} | {fmt(losses['video_sft'][1])} | {duration(logs['video_sft']['training_seconds'])} |",
        "",
        "![六阶段训练 loss 曲线](assets/training-curves.svg)",
        "",
        "## 评估结果",
        "",
        "| 模型 / 数据集 | 主要指标 |",
        "|---|---|",
        f"| LLM 文本留出集 | loss {fmt(llm_language['validation_loss'])}; PPL {fmt(llm_language['validation_perplexity'])}; BPB {fmt(llm_language['bits_per_byte'])} |",
        f"| VLM validation + 固定图像 | loss {fmt(vlm_eval['validation_loss'])}; 正确图关键词召回 {fmt(visual_ablation['correct_image_keyword_recall'])} |",
        f"| VLM 视觉反事实 | 错图/倒序召回 {fmt(visual_ablation['counterfactual_keyword_recall'])}; 正确减反事实 {fmt(visual_ablation['correct_minus_counterfactual_recall'])}; 回答变化率 {fmt(visual_ablation['completion_change_rate'])} |",
        f"| VLM 语言回归 | PPL {fmt(vlm_language['validation_perplexity'])}; 相对 LLM 变化 {fmt(vlm_language['validation_perplexity'] - llm_language['validation_perplexity'])} |",
        f"| Video QIVD 留出集 | loss {fmt(video_eval['test_loss'])}; exact {fmt(qivd_generation['normalized_exact_match'])}; token F1 {fmt(qivd_generation['token_f1'])} |",
        f"| 受控时序集 | exact {fmt(temporal['normalized_exact_match'])}; 倒序 exact {fmt(temporal['reversed_frame_exact_match'])}; token F1 差 {fmt(temporal['normal_minus_reversed_token_f1'])} |",
        f"| Video-Omni 语言回归 | PPL {fmt(video_language['validation_perplexity'])}; 相对 LLM 变化 {fmt(video_language['validation_perplexity'] - llm_language['validation_perplexity'])} |",
        "",
        "![受控时序正常帧与倒序帧指标](assets/temporal-ablation.svg)",
        "",
        (
            "受控时序集的 normal-minus-reversed 为正，才是模型利用帧序的证据；若为零或负值，必须解释为"
            "尚未建立时序敏感性。"
        ),
        "",
        "## 数据与本地产物",
        "",
        (
            f"QIVD：{qivd['video_count']:,} 个视频，固定 revision `{qivd['revision']}`，聚合 SHA-256 "
            f"`{qivd['aggregate_sha256']}`。QIVD 仅限研究使用，本项目不重新分发。"
        ),
        "",
        "下列文件仅保存在本机，**不会上传 GitHub**；哈希用于审计本地运行。",
        "",
        "| 本地产物 | Bytes | SHA-256 |",
        "|---|---:|---|",
        *[f"| `{item['name']}` | {item['bytes']:,} | `{item['sha256']}` |" for item in local_artifacts],
        "",
        "## 固定定性样例",
        "",
        "### LLM",
        "",
        "| 提示 | 模型续写 |",
        "|---|---|",
        *[
            f"| {one_line(item['prompt'])} | {one_line(item['completion'])} |"
            for item in llm_eval.get("generation", [])[:4]
        ],
        "",
        "### 三模型语言能力保持情况",
        "",
        "| 提示 | LLM | VLM 语言核心 | Video-Omni 语言核心 |",
        "|---|---|---|---|",
        *[
            f"| {one_line(llm_row['prompt'])} | {one_line(llm_row['completion'])} | "
            f"{one_line(vlm_row['completion'])} | {one_line(video_row['completion'])} |"
            for llm_row, vlm_row, video_row in zip(
                llm_eval["generation"],
                vlm_eval["language_regression"]["generation"],
                video_eval["language_regression"]["generation"],
                strict=True,
            )
        ],
        "",
        "### VLM",
        "",
        "| 图像数 | 提示 | 正确图像回答 | 错图/倒序图像回答 |",
        "|---:|---|---|---|",
        *[
            f"| {item.get('image_count', 1)} | {one_line(item.get('prompt', item.get('id', '')))} | "
            f"{one_line(item['completion'])} | {one_line(item['counterfactual_completion'])} |"
            for item in qualitative_vlm
        ],
        "",
        "### Video-Omni — QIVD 真实视频",
        "",
        "| 问题 | 参考答案 | 正常帧回答 | 倒序帧回答 |",
        "|---|---|---|---|",
        *[
            f"| {one_line(item['question'])} | {one_line(item['answer'])} | "
            f"{one_line(item['normal_completion'])} | {one_line(item['reversed_completion'])} |"
            for item in qivd_generation.get("qualitative", [])[:6]
        ],
        "",
        "### Video-Omni — 受控时序视频",
        "",
        "| 类别 | 问题 | 参考答案 | 正常帧回答 | 倒序帧回答 |",
        "|---|---|---|---|---|",
        *[
            f"| {one_line(item['category'])} | {one_line(item['question'])} | {one_line(item['answer'])} | "
            f"{one_line(item['normal_completion'])} | {one_line(item['reversed_completion'])} |"
            for item in temporal.get("qualitative", [])[:8]
        ],
        "",
        (
            "预期用途和限制见 `docs/model-cards/`，评估协议见 `docs/evaluation.md`；完整定性输出保存在"
            "本机 JSON 评估产物中。"
        ),
        "",
    ]
    report_path = ROOT / "reports/final-results.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    manifest = {"source_commit": commit, "uploaded": False, "local_artifacts": local_artifacts}
    (ROOT / "reports/local-artifact-manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    readme_path = ROOT / "README.md"
    readme = readme_path.read_text(encoding="utf-8")
    status = """<!-- STATUS_START -->
| 阶段 | 架构 | 训练 | 评估 |
|---|---|---|---|
| LLM | ✅ 63,912,192 参数 | ✅ Pretrain + SFT | ✅ loss / PPL / BPB / 固定生成 |
| VLM | ✅ SigLIP2 + Projector + LLM | ✅ Alignment + SFT | ✅ validation + 固定图像样例 |
| Video-Omni | ✅ Spatial + Temporal Adapter + LLM | ✅ Alignment + SFT | ✅ QIVD + 受控时序 + 倒序消融 |
<!-- STATUS_END -->"""
    readme, replacements = re.subn(
        r"<!-- STATUS_START -->.*?<!-- STATUS_END -->", status, readme, count=1, flags=re.DOTALL
    )
    if replacements != 1:
        raise RuntimeError("README status markers are missing or duplicated")
    readme_path.write_text(readme, encoding="utf-8")
    print(json.dumps({"report": str(report_path), "local_artifact_manifest": manifest}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
