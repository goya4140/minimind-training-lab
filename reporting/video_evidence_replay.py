"""Replay the pinned video evaluation while retaining every generated case.

This is a separately identified evidence replay, never an overwrite of baseline
evaluation. Its hook changes output retention only, not predictions or scoring.
"""
from __future__ import annotations

import argparse
import functools
import hashlib
import importlib.util
import json
import os
import re
import shutil
import sys
from pathlib import Path


def digest(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def validate_gpu_selection(run_manifest: dict, selected: str | None) -> str:
    expected = run_manifest.get("runtime", {}).get("cuda_visible_devices")
    if not isinstance(expected, str) or not expected.strip():
        raise ValueError("The immutable run has no explicit authorized GPU binding")
    if selected != expected:
        raise ValueError("CUDA_VISIBLE_DEVICES must exactly match the immutable run GPU binding")
    return expected


def retain_cases(module, output: Path):
    original = module.generation_summary
    receipts = []
    expected = [("qivd", 100), ("controlled_temporal", 160)]

    @functools.wraps(original)
    def summary(cases):
        index = len(receipts)
        if index >= len(expected):
            raise ValueError("Unexpected summary call; inspect the pinned evaluator")
        name, count = expected[index]
        if len(cases) != count or len({row["id"] for row in cases}) != count:
            raise ValueError(f"{name}: expected {count} unique fixed cases")
        result = original(cases)
        path = output / f"{name}-predictions.jsonl"
        with path.open("x", encoding="utf-8") as handle:
            for case in cases:
                handle.write(json.dumps(case, ensure_ascii=False, allow_nan=False) + "\n")
        receipts.append({"dataset": name, "cases": count, "file": path.name, "sha256": digest(path)})
        return result

    module.generation_summary = summary
    return receipts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--eval-id", required=True)
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,79}", args.eval_id):
        raise ValueError("Invalid independent evaluation ID")
    run = args.run_dir.resolve()
    root = run.parents[2]
    sys.path.insert(0, str(root / "src"))
    from minimind_lab.training.evidence import RunEvidence

    evidence = RunEvidence(root, run.name)
    if evidence.directory != run:
        raise ValueError("Not a canonical run directory")
    evidence.assert_sources_unchanged()
    events = [json.loads(line) for line in (run / "events.jsonl").read_text().splitlines()]
    if not any(e.get("kind") == "training_evaluation_report_completed" for e in events):
        raise ValueError("Wait for the original six-stage training/evaluation pipeline to complete")
    stage = "video-omni-sft"
    if not any(e.get("kind") == "stage_completed" and e.get("stage") == stage for e in events):
        raise ValueError("Video SFT stage is not completed")
    entries = [a for e in events if e.get("kind") == "artifacts_archived" and e.get("stage") == stage
               for a in e["artifacts"] if Path(a["archive"]).name == "video-omni-sft-cuda.pt"]
    if len(entries) != 1:
        raise ValueError("Expected one unambiguous archived final CUDA checkpoint")
    checkpoint = (root / entries[0]["archive"]).resolve()
    if not checkpoint.is_relative_to(run / "stages" / stage) or digest(checkpoint) != entries[0]["sha256"]:
        raise ValueError("Archived checkpoint binding failed")
    run_manifest = json.loads((run / "manifest.json").read_text())
    authorized_gpu = validate_gpu_selection(run_manifest, os.environ.get("CUDA_VISIBLE_DEVICES"))
    script = root / "scripts/evaluate_video_omni.py"
    output = run / "supplemental-evaluations" / args.eval_id
    output.mkdir(parents=True, exist_ok=False)
    manifest = {"schema": 1, "eval_id": args.eval_id, "run_id": run.name,
                "kind": "fixed-video-evaluation-full-output-replay-v1", "status": "started",
                "checkpoint": str(checkpoint.relative_to(root)), "checkpoint_sha256": digest(checkpoint),
                "evaluator_sha256": digest(script), "retention_tool_sha256": digest(__file__),
                "baseline_evaluation_unchanged": True, "max_new_tokens": 48,
                "cuda_visible_devices": authorized_gpu,
                "caveat": "Reversed predictions are scored against original references, not reversed-video truth."}

    def save():
        (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    save()
    shutil.copy2(__file__, output / "retention-tool.py")
    shutil.copy2(script, output / "pinned-evaluator.py")
    evidence.event("video_evidence_replay_started", eval_id=args.eval_id)
    previous_argv = sys.argv[:]
    try:
        spec = importlib.util.spec_from_file_location("pinned_video_evaluation", script)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        receipts = retain_cases(module, output)
        sys.argv = [str(script), "--config", str(root / "configs/video/sft-cuda.yaml"),
                    "--checkpoint", str(checkpoint), "--language-config", str(root / "configs/llm/pretrain-cuda.yaml"),
                    "--output", str(output / "summary.json")]
        module.main()
        if len(receipts) != 2:
            raise ValueError("Both fixed video generation sets must be retained")
        evidence.assert_sources_unchanged()
        if digest(checkpoint) != manifest["checkpoint_sha256"]:
            raise ValueError("Checkpoint changed during evaluation")
        manifest.update(status="completed", predictions=receipts, summary_sha256=digest(output / "summary.json"))
        save()
        evidence.event("video_evidence_replay_completed", eval_id=args.eval_id,
                       manifest=str((output / "manifest.json").relative_to(root)),
                       manifest_sha256=digest(output / "manifest.json"), generated_cases=260)
    except Exception as error:
        manifest.update(status="failed", error_type=type(error).__name__, error=str(error))
        save()
        evidence.event("video_evidence_replay_failed", eval_id=args.eval_id, error_type=type(error).__name__)
        raise
    finally:
        sys.argv = previous_argv


if __name__ == "__main__":
    main()
