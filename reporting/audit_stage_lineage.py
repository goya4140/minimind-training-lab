"""Read-only archive/lineage audit; not a model capability or goal-completion gate.

No torch import, checkpoint deserialization, GPU access, file writes or polling.
Training tensor/config/RNG audits and final report interpretation remain separate.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

STAGES = {
    "llm-pretrain": ("llm-64m-pretrain-cuda", "llm-64m-pretrain-cuda", "llm-pretrain-final.json"),
    "llm-sft": ("llm-64m-sft-cuda", "llm-64m-sft-cuda", "llm-sft-final.json"),
    "vlm-alignment": ("vlm-alignment-cuda", "vlm-vision-alignment-cuda", None),
    "vlm-sft": ("vlm-sft-cuda", "vlm-sft-cuda", "vlm-final.json"),
    "video-omni-alignment": ("video-omni-alignment-cuda", "video-omni-alignment-cuda", None),
    "video-omni-sft": ("video-omni-sft-cuda", "video-omni-sft-cuda", "video-omni-final.json"),
}


def digest(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def events(run):
    rows = [json.loads(line) for line in (run / "events.jsonl").read_text().splitlines()]
    if any(row.get("run_id") != run.name for row in rows):
        raise ValueError("Mixed run IDs in event ledger")
    return rows


def required(stage):
    checkpoint, log, evaluation = STAGES[stage]
    paths = {f"artifacts/checkpoints/{checkpoint}.pt", f"artifacts/checkpoints/{checkpoint}.resume.pt",
             f"artifacts/logs/{log}.json", f"artifacts/logs/{log}.console.log"}
    if evaluation:
        paths.add(f"artifacts/eval/{evaluation}")
    return paths


def audit_stage(run, rows, stage):
    run = Path(run).resolve()
    root = run.parents[2]
    completed = any(row.get("kind") == "stage_completed" and row.get("stage") == stage for row in rows)
    if not completed:
        return {"stage": stage, "status": "incomplete", "archive_files_verified": 0}
    entries = [entry for row in rows if row.get("kind") == "artifacts_archived" and row.get("stage") == stage
               for entry in row["artifacts"]]
    if len(entries) != len(required(stage)) or {entry["source"] for entry in entries} != required(stage):
        raise ValueError(f"{stage}: missing, duplicate or unexpected archive sources")
    for entry in entries:
        archive = (root / entry["archive"]).resolve()
        if not archive.is_relative_to(run / "stages" / stage):
            raise ValueError(f"{stage}: archive escapes stage directory")
        if digest(archive) != entry["sha256"]:
            raise ValueError(f"{stage}: archived content hash mismatch")
    inherited = [row for row in rows if row.get("kind") == "stage_inherited" and row.get("stage") == stage]
    origin = run.name
    if inherited:
        if len(inherited) != 1:
            raise ValueError(f"{stage}: ambiguous inheritance")
        lineage = [row for row in rows if row.get("kind") == "repair_lineage_registered"]
        if len(lineage) != 1 or lineage[0]["parent_run_id"] != inherited[0]["parent_run_id"]:
            raise ValueError(f"{stage}: missing or inconsistent parent lineage")
        parent = Path(lineage[0]["parent_root"]).resolve() / "artifacts/runs" / lineage[0]["parent_run_id"]
        manifest = json.loads((parent / "manifest.json").read_text())
        if manifest["run_id"] != parent.name or manifest["code_commit"] != lineage[0]["parent_code_commit"]:
            raise ValueError(f"{stage}: parent manifest binding mismatch")
        parent_rows = events(parent)
        if not any(row.get("kind") == "stage_completed" and row.get("stage") == stage for row in parent_rows):
            raise ValueError(f"{stage}: parent stage not completed")
        parent_entries = [entry for row in parent_rows if row.get("kind") == "artifacts_archived"
                          and row.get("stage") == stage for entry in row["artifacts"]]
        parent_map = {entry["source"]: entry for entry in parent_entries}
        if len(parent_entries) != len(entries) or set(parent_map) != required(stage):
            raise ValueError(f"{stage}: parent archive set differs")
        for entry in entries:
            original = parent_map[entry["source"]]
            path = (parent.parents[2] / original["archive"]).resolve()
            if not path.is_relative_to(parent / "stages" / stage):
                raise ValueError(f"{stage}: parent archive escapes stage directory")
            if original["sha256"] != entry["sha256"] or digest(path) != entry["sha256"]:
                raise ValueError(f"{stage}: inherited artifact differs from parent")
        origin = parent.name
    return {"stage": stage, "status": "archive-verified", "archive_files_verified": len(entries),
            "inherited": bool(inherited), "original_training_run_id": origin,
            "artifact_sha256": {entry["source"]: entry["sha256"] for entry in entries}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    args = parser.parse_args()
    run = args.run_dir.resolve()
    manifest = json.loads((run / "manifest.json").read_text())
    if manifest["run_id"] != run.name or run.parent.name != "runs" or run.parent.parent.name != "artifacts":
        raise ValueError("Not a canonical run directory")
    rows = events(run)
    stages = [audit_stage(run, rows, stage) for stage in STAGES]
    complete = all(row["status"] == "archive-verified" for row in stages)
    pipeline = any(row.get("kind") == "training_evaluation_report_completed" for row in rows)
    print(json.dumps({"run_id": run.name, "training_commit": manifest["code_commit"], "stages": stages,
                      "six_stage_archives_verified": complete, "pipeline_completion_event": pipeline,
                      "goal_completion_proven": False,
                      "limits": "Archive bytes and parent lineage only. Does not audit tensor/RNG/config contents, "
                      "semantic metrics, data leakage, screenshots, full per-sample retention or final deliverables. "
                      "Automatically generated final-results.md requires independent interpretation review."},
                     ensure_ascii=False, indent=2))
    raise SystemExit(0 if complete and pipeline else 2)


if __name__ == "__main__":
    main()
