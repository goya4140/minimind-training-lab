"""Export checkpoint-authoritative, binned training evidence and static PNGs.

Uses existing server torch/yaml for export and matplotlib for PNG rendering.
Never reads append-only telemetry as the authority after a resumed rollback.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
import statistics
from collections import defaultdict
from pathlib import Path

FIELDS = (
    "loss", "learning_rate", "grad_norm", "padded_tokens_per_second",
    "cuda_allocated_bytes", "cuda_reserved_bytes", "cuda_peak_allocated_bytes", "training_seconds",
)


def digest(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def nonnegative_plot_upper(values) -> float:
    """Keep zero visible and give almost-flat, high-valued series real headroom."""
    finite = [value for value in values if math.isfinite(value)]
    return max(finite, default=0.0) * 1.08 or 1.0


def aggregate(history: list[dict], *, run_id: str, checkpoint_step: int, bin_width: int) -> list[dict]:
    if bin_width < 1 or not history:
        raise ValueError("Nonempty history and positive bin width required")
    grouped = defaultdict(list)
    previous = 0
    for row in history:
        step = row["step"]
        if not isinstance(step, int) or not previous < step <= checkpoint_step:
            raise ValueError("History steps must be unique, increasing and within checkpoint")
        if row.get("run_id") != run_id:
            raise ValueError("History belongs to another run or an untagged probe")
        for name in FIELDS:
            value = row.get(name)
            if value is None and name != "grad_norm":
                raise ValueError(f"Missing required metric {name}")
            if value is not None and (not math.isfinite(value) or value < 0):
                raise ValueError(f"Invalid metric {name}")
        if row["cuda_peak_allocated_bytes"] < row["cuda_allocated_bytes"]:
            raise ValueError("CUDA peak cannot be below current allocated memory")
        grouped[(step - 1) // bin_width].append(row)
        previous = step
    output = []
    for index, rows in sorted(grouped.items()):
        grads = [r["grad_norm"] for r in rows if r.get("grad_norm") is not None]
        output.append({
            "bin_start": index * bin_width + 1,
            "bin_end": (index + 1) * bin_width,
            "first_observed_step": rows[0]["step"],
            "last_observed_step": rows[-1]["step"],
            "logged_microbatches": len(rows),
            "loss_mean": statistics.fmean(r["loss"] for r in rows),
            "loss_min": min(r["loss"] for r in rows),
            "loss_max": max(r["loss"] for r in rows),
            "learning_rate_last": rows[-1]["learning_rate"],
            "grad_norm_median": statistics.median(grads) if grads else None,
            "padded_tokens_per_second_median": statistics.median(r["padded_tokens_per_second"] for r in rows),
            "cuda_allocated_gib_last": rows[-1]["cuda_allocated_bytes"] / 2**30,
            "cuda_reserved_gib_last": rows[-1]["cuda_reserved_bytes"] / 2**30,
            "cuda_peak_allocated_gib_last": rows[-1]["cuda_peak_allocated_bytes"] / 2**30,
            "active_seconds_last": rows[-1]["training_seconds"],
        })
    return output


def export(args) -> None:
    import torch
    import yaml

    run_dir, checkpoint = args.run_dir.resolve(), args.checkpoint.resolve()
    root = run_dir.parents[2]
    manifest_path = run_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest["run_id"] != run_dir.name:
        raise ValueError("Run directory name and manifest run ID differ")
    # Hash and load the same open inode: training may atomically replace its pathname.
    with checkpoint.open("rb") as handle:
        checkpoint_sha256 = hashlib.file_digest(handle, "sha256").hexdigest()
        checkpoint_bytes = handle.tell()
        handle.seek(0)
        state = torch.load(handle, map_location="cpu", weights_only=False)
    if not {"step", "config", "optimizer", "history", "torch_rng_state"} <= state.keys():
        raise ValueError("Use a trusted complete resume checkpoint, not model-only weights")
    config = state["config"]
    experiments = {
        "llm-pretrain": "llm-64m-pretrain-cuda", "llm-sft": "llm-64m-sft-cuda",
        "vlm-alignment": "vlm-alignment-cuda", "vlm-sft": "vlm-sft-cuda",
        "video-omni-alignment": "video-omni-alignment-cuda", "video-omni-sft": "video-omni-sft-cuda",
    }
    if experiments.get(args.stage) != config["experiment"]["name"]:
        raise ValueError("Stage label does not match checkpoint experiment")
    configs = [
        yaml.safe_load((run_dir / "configs" / name.removeprefix("configs/")).read_text())
        for name in manifest["config_paths"]
    ]
    if config not in configs:
        raise ValueError("Embedded config differs from immutable run snapshot")
    step, target = state["step"], config["training"]["steps"]
    if not 0 < step <= target:
        raise ValueError("Checkpoint is outside configured budget")
    if args.final:
        events = [json.loads(line) for line in (run_dir / "events.jsonl").read_text().splitlines()]
        archived = [entry for event in events if event.get("kind") == "artifacts_archived"
                    and event.get("stage") == args.stage for entry in event["artifacts"]]
        bound = any(entry["sha256"] == checkpoint_sha256 and (root / entry["archive"]).resolve() == checkpoint
                    for entry in archived)
        completed = any(e.get("kind") == "stage_completed" and e.get("stage") == args.stage for e in events)
        if step != target or not bound or not completed:
            raise ValueError("Final curves require completed stage and its exact archived resume artifact")
    log_interval = config["training"]["log_interval"]
    if args.bin_width is not None and args.bin_width < 1:
        raise ValueError("Bin width must be positive")
    bin_width = args.bin_width or max(log_interval, math.ceil(target / 160 / log_interval) * log_interval)
    history = state["history"]
    rows = aggregate(history, run_id=manifest["run_id"], checkpoint_step=step, bin_width=bin_width)
    args.output.mkdir(parents=True, exist_ok=False)
    history_path = args.output / "history-records.json"
    history_path.write_text(json.dumps(history, ensure_ascii=False) + "\n")
    shutil.copy2(Path(__file__), args.output / "curve-tool.py")
    csv_path = args.output / "curve-data.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    metadata = {
        "schema": 1, "status": "final-stage" if args.final else "intermediate-not-final",
        "run_id": manifest["run_id"], "training_commit": manifest["code_commit"],
        "stage": args.stage, "experiment": config["experiment"]["name"],
        "checkpoint": str(checkpoint.relative_to(root)), "checkpoint_sha256": checkpoint_sha256,
        "checkpoint_bytes": checkpoint_bytes, "checkpoint_step": step, "target_microsteps": target,
        "run_manifest_sha256": digest(manifest_path), "curve_data_sha256": digest(csv_path),
        "history_sha256": digest(history_path), "curve_tool_sha256": digest(Path(__file__)),
        "raw_history_retention": "Server only; publish binned CSV and figures, never history-records.json",
        "history_records": len(history), "bins": len(rows), "bin_width_microsteps": bin_width,
        "first_observed_step": history[0]["step"], "last_observed_step": history[-1]["step"],
        "first_recorded_at_utc": history[0]["recorded_at_utc"],
        "last_recorded_at_utc": history[-1]["recorded_at_utc"],
        "segments": sorted({r["segment_id"] for r in history}),
        "definitions": {
            "loss_mean": "Unweighted mean of logged microbatch losses within fixed microstep bins; not token-weighted validation loss or every training batch",
            "learning_rate_last": "Last logged optimizer LR in the bin",
            "grad_norm_median": "Median last-update gradient norms observed at logged microsteps",
            "padded_tokens_per_second_median": "Median of logged segment-average padded throughput; includes padding and unsupervised context, not instantaneous supervised-token throughput",
            "cuda_memory": "Last PyTorch allocated/reserved/process-peak values in each bin, GiB; not nvidia-smi device total",
            "bin_policy": "Fixed from full configured budget, including partial final bin; gaps remain gaps, never zero-filled",
        },
        "limits": "Single run and partial-stage evidence unless final-stage. Loss decline alone does not prove model capability. This exporter does not certify the whole experiment.",
        "export_environment": {"torch": torch.__version__, "yaml": yaml.__version__},
    }
    (args.output / "curve-source.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"directory": str(args.output), "step": step, "bins": len(rows), "status": metadata["status"]}))


def plot(args) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter

    source_path, csv_path = args.source / "curve-source.json", args.source / "curve-data.csv"
    meta = json.loads(source_path.read_text())
    if digest(csv_path) != meta["curve_data_sha256"]:
        raise ValueError("Curve data checksum mismatch")
    with csv_path.open(newline="") as handle:
        rows = [{key: (float(value) if value else math.nan) for key, value in row.items()}
                for row in csv.DictReader(handle)]
    args.output.mkdir(parents=True, exist_ok=False)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 12, "axes.spines.top": False,
                         "axes.spines.right": False, "axes.labelcolor": "#253746", "text.color": "#253746"})
    blue, gold, neutral = "#216A9A", "#B17B22", "#5E6871"
    panels = [
        ("loss_mean", "Training loss", "Mean logged microbatch loss", blue),
        ("learning_rate_last", "Learning rate", "Last logged optimizer LR", blue),
        ("grad_norm_median", "Gradient norm", "Median last-update norm", blue),
        ("padded_tokens_per_second_median", "Padded training throughput", "Segment-average padded tokens/s", blue),
    ]
    outputs = []
    for field, title, ylabel, color in panels:
        fig, ax = plt.subplots(figsize=(10, 5.8), layout="constrained")
        segments = []
        chunk = []
        for row in rows:
            if chunk and row["bin_start"] > chunk[-1]["bin_end"] + 1:
                segments.append(chunk); chunk = []
            chunk.append(row)
        segments.append(chunk)
        for chunk in segments:
            ax.plot([r["last_observed_step"] for r in chunk], [r[field] for r in chunk],
                    color=color, linewidth=1.8, marker="o", markersize=3)
        ax.set_title(title, loc="left", fontsize=15, pad=15)
        ax.set_xlabel("Micro-step (not optimizer update)")
        ax.set_ylabel(ylabel)
        ax.set_ylim(0, nonnegative_plot_upper(row[field] for row in rows))
        ax.set_xlim(0, meta["last_observed_step"] * 1.04)
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{int(v):,}"))
        ax.grid(axis="y", color="#E3E7EA", linewidth=0.7)
        fig.suptitle(f"{meta['experiment']}   {meta['status']}", fontsize=11, color=neutral)
        fig.text(0.01, -0.04, f"Run {meta['run_id']}  |  CSV SHA256 {meta['curve_data_sha256'][:16]}\n"
                 f"Checkpoint {meta['checkpoint_step']:,}/{meta['target_microsteps']:,}; bin width {meta['bin_width_microsteps']:,}. "
                 "Means/medians of logged records, not every batch.",
                 fontsize=9, color=neutral)
        target = args.output / f"{field}.png"
        fig.savefig(target, dpi=180, bbox_inches="tight")
        plt.close(fig)
        outputs.append({"file": target.name, "sha256": digest(target), "metric": field})
    fig, ax = plt.subplots(figsize=(10, 5.8), layout="constrained")
    x = [r["last_observed_step"] for r in rows]
    for key, label, color, style in [
        ("cuda_allocated_gib_last", "Allocated", blue, "-"),
        ("cuda_reserved_gib_last", "Reserved", gold, "--"),
        ("cuda_peak_allocated_gib_last", "Process peak allocated", neutral, ":"),
    ]:
        # NaNs break lines at missing fixed bins rather than inventing continuity.
        xx, yy = [], []
        for index, row in enumerate(rows):
            if index and row["bin_start"] > rows[index - 1]["bin_end"] + 1:
                xx.append(math.nan); yy.append(math.nan)
            xx.append(x[index]); yy.append(row[key])
        ax.plot(xx, yy, label=label, color=color, linestyle=style, linewidth=1.8)
    ax.set_title("PyTorch CUDA memory", loc="left", fontsize=15, pad=15)
    ax.set(xlabel="Micro-step (not optimizer update)", ylabel="GiB", ylim=(0, None))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{int(v):,}"))
    ax.grid(axis="y", color="#E3E7EA", linewidth=0.7)
    ax.legend(loc="lower right", frameon=False)
    fig.suptitle(f"{meta['experiment']}   {meta['status']}",
                 fontsize=10, color=neutral)
    fig.text(0.01, -0.04, f"Run {meta['run_id']}  |  Last logged values per bin; not nvidia-smi total.\n"
             "Peak is process-segment scoped, and can restart after a resumed process.", fontsize=9, color=neutral)
    target = args.output / "cuda-memory.png"
    fig.savefig(target, dpi=180, bbox_inches="tight")
    plt.close(fig)
    outputs.append({"file": target.name, "sha256": digest(target), "metric": "cuda-memory"})
    shutil.copy2(Path(__file__), args.output / "renderer-tool.py")
    receipt = {"source_sha256": digest(source_path), "csv_sha256": digest(csv_path),
               "renderer_tool_sha256": digest(Path(__file__)),
               "matplotlib": matplotlib.__version__, "figures": outputs, "status": meta["status"]}
    (args.output / "figure-manifest.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps({"output": str(args.output), "figures": len(outputs), "status": meta["status"]}))


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    exporter = sub.add_parser("export")
    exporter.add_argument("--run-dir", type=Path, required=True)
    exporter.add_argument("--checkpoint", type=Path, required=True)
    exporter.add_argument("--stage", required=True)
    exporter.add_argument("--bin-width", type=int)
    exporter.add_argument("--final", action="store_true")
    exporter.add_argument("--output", type=Path, required=True)
    renderer = sub.add_parser("plot")
    renderer.add_argument("--source", type=Path, required=True)
    renderer.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    (export if args.command == "export" else plot)(args)


if __name__ == "__main__":
    main()
