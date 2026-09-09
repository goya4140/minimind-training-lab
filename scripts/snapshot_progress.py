#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import os
import sys
from datetime import UTC, datetime
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from preflight_mps_pipeline import STAGES


def live_pid(checkpoint: Path) -> int | None:
    try:
        pid = int(checkpoint.with_suffix(".lock").read_text(encoding="utf-8").strip())
        os.kill(pid, 0)
        return pid
    except (FileNotFoundError, ProcessLookupError, PermissionError, ValueError):
        return None


def records(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    output = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict) and isinstance(value.get("step"), int) and "loss" in value:
            output.append(value)
    return output


def duration(seconds: float | None) -> str:
    if seconds is None or not math.isfinite(seconds):
        return "—"
    hours, remainder = divmod(max(0, round(seconds)), 3600)
    minutes, _ = divmod(remainder, 60)
    return f"{hours:d}h {minutes:02d}m"


def main() -> None:
    rows = []
    prior_complete = True
    for stage, _, config_path in STAGES:
        with (ROOT / config_path).open(encoding="utf-8") as handle:
            config = yaml.safe_load(handle)
        experiment = config["experiment"]["name"]
        checkpoint = ROOT / config["training"]["checkpoint_path"]
        total_steps = int(config["training"]["steps"])
        history = records(ROOT / f"artifacts/logs/{experiment}.console.log")
        if checkpoint.is_file():
            status = "complete"
        elif live_pid(checkpoint) is not None:
            status = "active"
        elif prior_complete:
            status = "ready"
        else:
            status = "pending"
        prior_complete = status == "complete"
        last = history[-1] if history else None
        recent = history[-20:]
        current_step = int(last["step"]) if last else 0
        seconds_per_step = float(last.get("seconds_per_step", math.nan)) if last else math.nan
        eta = (total_steps - current_step) * seconds_per_step if current_step and status == "active" else None
        rows.append(
            {
                "stage": stage,
                "status": status,
                "current_step": current_step,
                "total_steps": total_steps,
                "percent": 100 * current_step / total_steps if current_step else 0,
                "first_loss": float(history[0]["loss"]) if history else None,
                "recent_loss": sum(float(item["loss"]) for item in recent) / len(recent) if recent else None,
                "seconds_per_step": seconds_per_step if math.isfinite(seconds_per_step) else None,
                "eta": eta,
            }
        )

    lines = [
        "# 正式训练进度快照",
        "",
        f"生成时间：{datetime.now(UTC).isoformat()}。数据来自本机 Git 忽略日志；不包含权重或训练数据。",
        "",
        "| Stage | Status | Micro-step | Progress | First loss | Recent mean loss | Seconds/step | ETA |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        first_loss = f"{row['first_loss']:.4f}" if row["first_loss"] is not None else "—"
        recent_loss = f"{row['recent_loss']:.4f}" if row["recent_loss"] is not None else "—"
        seconds_per_step = f"{row['seconds_per_step']:.4f}" if row["seconds_per_step"] is not None else "—"
        lines.append(
            f"| {row['stage']} | {row['status']} | {row['current_step']:,} / {row['total_steps']:,} | "
            f"{row['percent']:.1f}% | {first_loss} | {recent_loss} | {seconds_per_step} | {duration(row['eta'])} |"
        )
    lines.extend(
        [
            "",
            "`Recent mean loss` 是最近 20 条日志记录的均值，仅用于观察趋势；不同阶段的 loss mask 与数据不同，",
            "不能把数值直接横向排名。ETA 按当前进程的累计 seconds/step 外推，不包含验证、评估和后续阶段。",
            "",
        ]
    )
    output = ROOT / "reports/training-progress.md"
    output.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"output": str(output), "stages": rows}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
