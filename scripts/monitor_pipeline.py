#!/usr/bin/env python3
"""持续监控六阶段训练流水线：采集状态、检测异常、记录阶段切换事件。

这是一个**只读观测**脚本：不启动、不停止、不修改任何训练过程，也不会触碰
checkpoint 或日志。它的职责是把「现在发生了什么」压缩成一段结构化结论，
供人工查看或由上层自动化（定时任务）解读。

检测的事件
----------
- ``stage_started``    某个阶段从 pending/ready 变为 active
- ``stage_finished``   某个阶段产出最终 checkpoint（status 变为 complete）
- ``pipeline_stalled`` 存在 active 阶段，但训练日志长时间无新记录
                       （常见原因是系统休眠或数据加载阻塞，不一定是故障）
- ``pipeline_dead``    流水线主进程不存在，且仍有未完成的阶段（严重）
- ``loss_invalid``     最近日志记录中出现非有限 loss（NaN / inf）
- ``pipeline_finished`` 六个阶段全部完成

输出
----
- stdout：JSON（``--quiet`` 时只写文件不打印）
- ``artifacts/monitor-state.json``：最近一次观测的完整快照，用于与下一次对比
- ``artifacts/monitor-events.jsonl``：追加写入的事件历史（Git 忽略）

退出码
------
- ``0`` 无异常
- ``1`` 有 warn 级事件（需要关注，但不紧急）
- ``2`` 有 critical 级事件（流水线中断或 loss 非法）

用法
----
    uv run python scripts/monitor_pipeline.py
    uv run python scripts/monitor_pipeline.py --stall-minutes 20 --quiet
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from preflight_mps_pipeline import STAGES

DEFAULT_STATE = "artifacts/monitor-state.json"
DEFAULT_EVENTS = "artifacts/monitor-events.jsonl"
PIPELINE_PATTERN = "run_mps_pipeline.py"

# 事件类型 -> 严重级别
SEVERITY = {
    "stage_started": "info",
    "stage_finished": "info",
    "pipeline_finished": "info",
    "pipeline_stalled": "warn",
    "pipeline_dead": "critical",
    "loss_invalid": "critical",
}


def read_records(path: Path) -> list[dict]:
    """读取 JSONL 训练日志，只保留含 step 与 loss 的记录。"""
    if not path.is_file():
        return []
    output: list[dict] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict) and isinstance(value.get("step"), int) and "loss" in value:
            output.append(value)
    return output


def live_pid(checkpoint: Path) -> int | None:
    """返回持有该阶段运行锁的存活进程 PID，锁不存在或进程已退出时返回 None。"""
    try:
        pid = int(checkpoint.with_suffix(".lock").read_text(encoding="utf-8").strip())
        os.kill(pid, 0)
        return pid
    except (FileNotFoundError, ProcessLookupError, PermissionError, ValueError):
        return None


def pipeline_pids() -> list[int]:
    """返回流水线主进程的 PID 列表。"""
    try:
        result = subprocess.run(
            ["pgrep", "-f", PIPELINE_PATTERN],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return []
    pids: list[int] = []
    for token in result.stdout.split():
        try:
            pids.append(int(token))
        except ValueError:
            continue
    return pids


def collect_stage(stage: str, config_path: str, prior_complete: bool) -> dict:
    """采集单个阶段的状态。状态判定与 snapshot_progress.py 保持一致。"""
    with (ROOT / config_path).open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    experiment = config["experiment"]["name"]
    checkpoint = ROOT / config["training"]["checkpoint_path"]
    total_steps = int(config["training"]["steps"])
    log_path = ROOT / f"artifacts/logs/{experiment}.console.log"
    records = read_records(log_path)

    if checkpoint.is_file():
        status = "complete"
    elif live_pid(checkpoint) is not None:
        status = "active"
    elif prior_complete:
        status = "ready"
    else:
        status = "pending"

    last = records[-1] if records else None
    recent = records[-20:]
    current_step = int(last["step"]) if last else 0
    seconds_per_step = float(last.get("seconds_per_step", math.nan)) if last else math.nan
    recent_loss = sum(float(item["loss"]) for item in recent) / len(recent) if recent else None
    loss_finite = all(math.isfinite(float(item["loss"])) for item in records[-100:]) if records else True

    eta = None
    if status == "active" and current_step and math.isfinite(seconds_per_step):
        eta = (total_steps - current_step) * seconds_per_step

    mtime = log_path.stat().st_mtime if log_path.is_file() else None

    return {
        "stage": stage,
        "status": status,
        "current_step": current_step,
        "total_steps": total_steps,
        "percent": round(100 * current_step / total_steps, 2) if current_step else 0.0,
        "recent_loss": round(recent_loss, 4) if recent_loss is not None else None,
        "seconds_per_step": round(seconds_per_step, 4) if math.isfinite(seconds_per_step) else None,
        "eta_seconds": round(eta) if eta is not None else None,
        "log_age_seconds": round(time.time() - mtime) if mtime is not None else None,
        "loss_finite": loss_finite,
    }


def duration(seconds: float | None) -> str:
    if not seconds or not math.isfinite(seconds):
        return "—"
    hours, remainder = divmod(max(0, round(seconds)), 3600)
    minutes, _ = divmod(remainder, 60)
    return f"{hours:d}h {minutes:02d}m"


def build_events(stages: list[dict], previous: dict, pipeline_alive: bool, stall_minutes: float) -> list[dict]:
    """对比上一次观测，生成本次新增的事件。"""
    now = datetime.now(UTC).isoformat()
    events: list[dict] = []
    prior_status = previous.get("stage_status", {})
    prior_alert = previous.get("last_alert_at", {})

    def emit(kind: str, stage: str | None, detail: str, cooldown_minutes: float | None = None) -> None:
        if cooldown_minutes is not None:
            last = prior_alert.get(f"{kind}:{stage}")
            if last is not None:
                try:
                    elapsed = (datetime.now(UTC) - datetime.fromisoformat(last)).total_seconds() / 60
                except ValueError:
                    elapsed = math.inf
                if elapsed < cooldown_minutes:
                    return
            prior_alert[f"{kind}:{stage}"] = now
        events.append(
            {
                "at": now,
                "kind": kind,
                "stage": stage,
                "severity": SEVERITY[kind],
                "detail": detail,
            }
        )

    for row in stages:
        name = row["stage"]
        before = prior_status.get(name)
        after = row["status"]
        if before is not None and before != after:
            if after == "active":
                emit("stage_started", name, f"{name} 开始训练")
            elif after == "complete":
                emit(
                    "stage_finished",
                    name,
                    f"{name} 完成，checkpoint 已落盘（最终步数 {row['current_step']:,}）",
                )
        if after == "active" and not row["loss_finite"]:
            emit("loss_invalid", name, f"{name} 最近日志出现非有限 loss（NaN/inf）")
        if after == "active" and row["log_age_seconds"] is not None:
            age_minutes = row["log_age_seconds"] / 60
            if age_minutes >= stall_minutes:
                emit(
                    "pipeline_stalled",
                    name,
                    f"{name} 日志已 {age_minutes:.0f} 分钟无更新"
                    "（常见原因为系统休眠或数据加载阻塞，需确认进程是否仍在推进）",
                    cooldown_minutes=60,
                )

    all_complete = all(row["status"] == "complete" for row in stages)
    if all_complete and not previous.get("all_complete"):
        emit("pipeline_finished", None, "六个阶段全部完成")
    if not all_complete and not pipeline_alive:
        pending = [row["stage"] for row in stages if row["status"] != "complete"]
        emit(
            "pipeline_dead",
            None,
            f"流水线主进程不存在，但仍有未完成阶段：{', '.join(pending)}",
            cooldown_minutes=60,
        )

    return events


def main() -> None:
    parser = argparse.ArgumentParser(description="监控六阶段训练流水线的运行状态。")
    parser.add_argument("--stall-minutes", type=float, default=15.0, help="日志停滞多少分钟算异常，默认 15")
    parser.add_argument("--state", default=DEFAULT_STATE, help="状态快照路径")
    parser.add_argument("--events", default=DEFAULT_EVENTS, help="事件历史路径（JSONL 追加）")
    parser.add_argument("--quiet", action="store_true", help="不向 stdout 打印 JSON")
    args = parser.parse_args()

    state_path = ROOT / args.state
    events_path = ROOT / args.events
    previous: dict = {}
    if state_path.is_file():
        try:
            previous = json.loads(state_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            previous = {}

    stages: list[dict] = []
    prior_complete = True
    for stage, _runner, config_path in STAGES:
        row = collect_stage(stage, config_path, prior_complete)
        prior_complete = row["status"] == "complete"
        stages.append(row)

    pipeline_alive = bool(pipeline_pids())
    events = build_events(stages, previous, pipeline_alive, args.stall_minutes)

    active = next((row for row in stages if row["status"] == "active"), None)
    all_complete = all(row["status"] == "complete" for row in stages)
    severities = {event["severity"] for event in events}
    if "critical" in severities:
        severity = "critical"
    elif "warn" in severities:
        severity = "warn"
    else:
        severity = "ok"

    if events:
        events_path.parent.mkdir(parents=True, exist_ok=True)
        with events_path.open("a", encoding="utf-8") as handle:
            for event in events:
                handle.write(json.dumps(event, ensure_ascii=False) + "\n")

    state = {
        "checked_at": datetime.now(UTC).isoformat(),
        "pipeline_alive": pipeline_alive,
        "all_complete": all_complete,
        "active_stage": active["stage"] if active else None,
        "stage_status": {row["stage"]: row["status"] for row in stages},
        "last_alert_at": previous.get("last_alert_at", {}),
        "stages": stages,
    }
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")

    if active:
        summary = (
            f"{active['stage']} 进行中 {active['current_step']:,}/{active['total_steps']:,} "
            f"({active['percent']:.1f}%)，剩余约 {duration(active['eta_seconds'])}"
        )
    elif all_complete:
        summary = "六个阶段全部完成"
    else:
        pending = [row["stage"] for row in stages if row["status"] != "complete"]
        summary = f"无活动阶段，未完成：{', '.join(pending)}"

    payload = {
        "checked_at": state["checked_at"],
        "severity": severity,
        "summary": summary,
        "pipeline_alive": pipeline_alive,
        "active_stage": state["active_stage"],
        "all_complete": all_complete,
        "new_events": events,
        "stages": stages,
    }
    if not args.quiet:
        print(json.dumps(payload, ensure_ascii=False, indent=2))

    sys.exit(0 if severity == "ok" else 1 if severity == "warn" else 2)


if __name__ == "__main__":
    main()
