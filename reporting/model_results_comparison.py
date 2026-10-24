"""Extract archive-bound aggregate results and render descriptive comparison charts."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

from audit_stage_lineage import STAGES, audit_stage, events

LANGUAGE_STAGES = ("llm-pretrain", "llm-sft", "vlm-sft", "video-omni-sft")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_language(rows):
    if [r["stage"] for r in rows] != list(LANGUAGE_STAGES):
        raise ValueError("Expected the four designated language checkpoints in fixed order")
    counts = {(r["predicted_tokens"], r["utf8_bytes"]) for r in rows}
    if counts != {(670079, 1906560)}:
        raise ValueError("Language comparison must use the same fixed token/byte denominators")
    for row in rows:
        if any(not math.isfinite(row[k]) or row[k] <= 0 for k in ("nll", "ppl", "bpb")):
            raise ValueError("Non-finite or non-positive corpus metric")
        if not math.isclose(row["ppl"], math.exp(row["nll"]), rel_tol=1e-12):
            raise ValueError("PPL/NLL mismatch")
        bpb = row["nll"] * row["predicted_tokens"] / (math.log(2) * row["utf8_bytes"])
        if not math.isclose(row["bpb"], bpb, rel_tol=1e-12):
            raise ValueError("BPB denominator mismatch")


def extract(run, output):
    run = run.resolve()
    root = run.parents[2]
    rows = events(run)
    if not any(e.get("kind") == "training_evaluation_report_completed" for e in rows):
        raise ValueError("Original pipeline has not completed")
    audits = [audit_stage(run, rows, stage) for stage in STAGES]
    if not all(x["status"] == "archive-verified" for x in audits):
        raise ValueError("All six completed stage archives are required")
    sources = []

    def read(stage, filename):
        entries = [a for e in rows if e.get("kind") == "artifacts_archived" and e.get("stage") == stage
                   for a in e["artifacts"] if Path(a["archive"]).name == filename]
        if len(entries) != 1:
            raise ValueError("Ambiguous archive result")
        entry = entries[0]
        path = (root / entry["archive"]).resolve()
        if not path.is_relative_to((run / "stages" / stage).resolve()) or sha(path) != entry["sha256"]:
            raise ValueError("Result archive binding failed")
        sources.append({"stage": stage, "archive": entry["archive"], "sha256": entry["sha256"]})
        return json.loads(path.read_text())

    training = []
    for stage, (_, summary_name, _) in STAGES.items():
        d = read(stage, summary_name + ".json")
        training.append({"stage": stage, "microsteps": d["total_steps"],
                         "active_training_seconds": d["training_seconds"],
                         "original_training_run_id": next(x["original_training_run_id"] for x in audits if x["stage"] == stage)})
    language, video = [], None
    for stage in LANGUAGE_STAGES:
        d = read(stage, STAGES[stage][2])
        corpus = d["corpus"] if stage.startswith("llm-") else d["language_regression"]["corpus"]
        language.append({"stage": stage, "nll": corpus["validation_loss"], "ppl": corpus["validation_perplexity"],
                         "bpb": corpus["bits_per_byte"], "predicted_tokens": corpus["predicted_tokens"], "utf8_bytes": corpus["utf8_bytes"]})
        if stage == "video-omni-sft":
            video = [{"dataset": name, "generated_samples": d[key]["generated_samples"],
                      "normal_f1": d[key]["token_f1"], "reversed_f1": d[key]["reversed_frame_token_f1"],
                      "normalized_change_rate": d[key]["completion_change_rate_on_reversal"]}
                     for name, key in (("QIVD", "qivd_generation"), ("Controlled temporal", "controlled_temporal"))]
    validate_language(language)
    result = {"schema": 1, "run_id": run.name, "training": training, "language": language, "video": video,
              "sources": sources, "exporter_sha256": sha(__file__),
              "limits": "Single training trajectory, no repeat-run uncertainty or causal attribution. Equal token/byte counts are checked, not semantic corpus deduplication. Video F1 is lexical, reverse scores use original references. Figures are data plots, not screenshots."}
    output.mkdir(parents=True, exist_ok=False)
    (output / "comparison-source.json").write_text(json.dumps(result, indent=2) + "\n")


def plot(source, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    d = json.loads(source.read_text())
    validate_language(d["language"])
    output.mkdir(parents=True, exist_ok=False)
    plt.rcParams.update({"font.size": 14, "axes.spines.top": False, "axes.spines.right": False})
    digest = sha(source)
    figures = []
    fig, ax = plt.subplots(figsize=(11, 6.6), layout="constrained")
    values = [r["ppl"] for r in d["language"]]
    bars = ax.bar(["LLM pretrain", "LLM SFT", "VLM SFT", "Video SFT"], values, color=["#7a8a99", "#356fa0", "#268477", "#9268a0"])
    ax.bar_label(bars, fmt="%.4f", padding=5)
    ax.set_ylim(0, max(values) * 1.2)
    ax.set_ylabel("Perplexity (lower is better on this corpus)")
    ax.set_title("Same fixed text corpus: language-modeling regression", pad=20)
    fig.supxlabel("2,048 rows | 670,079 predicted tokens | 1,906,560 UTF-8 bytes\nSingle trajectory; not a general ability ranking. Source SHA256 " + digest[:16], fontsize=11)
    path = output / "language-ppl.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    figures.append({"file": path.name, "sha256": sha(path)})
    fig, ax = plt.subplots(figsize=(11, 6.6), layout="constrained")
    normal = [r["normal_f1"] for r in d["video"]]
    reverse = [r["reversed_f1"] for r in d["video"]]
    a = ax.bar([-0.18, 0.82], normal, width=.34, label="Normal frames", color="#268477")
    b = ax.bar([0.18, 1.18], reverse, width=.34, label="Reversed frames", color="#356fa0")
    ax.bar_label(a, fmt="%.4f", padding=4)
    ax.bar_label(b, fmt="%.4f", padding=4)
    ax.set_xticks([0, 1], [f"{r['dataset']} (n={r['generated_samples']})" for r in d["video"]])
    ax.set_ylim(0, .25)
    ax.set_ylabel("Mean bag-of-token F1 (not semantic accuracy)")
    ax.set_title("Fixed video generation: no normal/reversed F1 gap", pad=20)
    ax.legend(loc="upper left")
    fig.supxlabel("Both conditions scored against ORIGINAL references; dataset means are not interchangeable.\nNo error bars: one run, no uncertainty estimate. Source SHA256 " + digest[:16], fontsize=11)
    path = output / "video-frame-order-f1.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    figures.append({"file": path.name, "sha256": sha(path)})
    (output / "figure-manifest.json").write_text(json.dumps({"source_sha256": digest, "renderer_sha256": sha(__file__), "matplotlib": matplotlib.__version__, "figures": figures}, indent=2) + "\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    e = sub.add_parser("extract")
    e.add_argument("--run-dir", type=Path, required=True)
    e.add_argument("--output", type=Path, required=True)
    v = sub.add_parser("plot")
    v.add_argument("--source", type=Path, required=True)
    v.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    extract(args.run_dir, args.output) if args.command == "extract" else plot(args.source, args.output)
