"""Read-only final training audit. Does not certify capability or report delivery.

CPU deserialization only: never restores RNG, initializes CUDA, or starts training.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

from audit_stage_lineage import STAGES, audit_stage, events

CONFIGS = {"llm-pretrain": "llm/pretrain-cuda.yaml", "llm-sft": "llm/sft-cuda.yaml",
           "vlm-alignment": "vlm/alignment-cuda.yaml", "vlm-sft": "vlm/sft-cuda.yaml",
           "video-omni-alignment": "video/alignment-cuda.yaml", "video-omni-sft": "video/sft-cuda.yaml"}


def digest(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def validate_history(history, origin, total, accumulation):
    if not history or any(r.get("run_id") != origin for r in history):
        raise ValueError("History missing or original training run identity mismatch")
    steps = [r["step"] for r in history]
    if steps != sorted(set(steps)) or steps[-1] > total:
        raise ValueError("History steps are duplicated, out of order, or beyond budget")
    for row in history:
        step = row["step"]
        updates = step // accumulation + int(step == total and step % accumulation != 0)
        if row.get("optimizer_updates") != updates:
            raise ValueError("Logged optimizer update counter mismatch")
    return {"history_original_run_id": origin, "history_last_logged_step": steps[-1],
            "unlogged_tail_microsteps": total - steps[-1],
            "configured_final_optimizer_updates": (total + accumulation - 1) // accumulation}


def validate_rng(cpu, cuda, tensor_type, uint8):
    def valid(tensor):
        return isinstance(tensor, tensor_type) and tensor.dtype == uint8 and tensor.ndim == 1 and tensor.numel() > 0
    if not valid(cpu):
        raise ValueError("Invalid CPU RNG byte tensor")
    if not isinstance(cuda, list) or len(cuda) != 1 or not all(valid(t) for t in cuda):
        raise ValueError("Expected one CUDA RNG byte tensor for the pinned single-device run")
    return {"cpu_rng_bytes": cpu.numel(), "cuda_rng_devices_saved": len(cuda),
            "cuda_rng_bytes": [t.numel() for t in cuda],
            "rng_restored_or_cuda_initialized": False}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir", required=True, type=Path)
    p.add_argument("--rehash-inputs", action="store_true")
    args = p.parse_args()
    run = args.run_dir.resolve()
    root = run.parents[2]
    sys.path.insert(0, str(root / "src"))
    import torch
    import yaml
    from minimind_lab.training.evidence import RunEvidence
    from minimind_lab.training.artifacts import verify_resume_checkpoint

    evidence = RunEvidence(root, run.name)
    if evidence.directory != run:
        raise ValueError("Not a canonical run directory")
    evidence.assert_sources_unchanged()
    manifest = json.loads(evidence.manifest_path.read_text())
    ledger = events(run)
    if not any(e.get("kind") == "training_evaluation_report_completed" for e in ledger):
        raise ValueError("Original pipeline completion event missing")
    input_bytes = 0
    if args.rehash_inputs:
        for entry in manifest["data_files"]:
            path = root / entry["path"]
            if path.stat().st_size != entry["bytes"] or digest(path) != entry["sha256"]:
                raise ValueError(f"Pinned input byte hash mismatch: {entry['path']}")
            input_bytes += entry["bytes"]
    stages = []
    for stage, (checkpoint, _, _) in STAGES.items():
        lineage = audit_stage(run, ledger, stage)
        if lineage["status"] != "archive-verified":
            raise ValueError("Stage not completed and archived")
        path = run / "stages" / stage / f"{checkpoint}.resume.pt"
        before = digest(path)
        config_path = run / "configs" / CONFIGS[stage]
        config = yaml.safe_load(config_path.read_text())
        status = verify_resume_checkpoint(path, config, require_complete=True)
        if not status["all_finite"] or not status["optimizer_all_finite"]:
            raise ValueError("Non-finite model/optimizer tensors")
        state = torch.load(path, map_location="cpu", weights_only=False)
        history = validate_history(state["history"], lineage["original_training_run_id"],
                                   state["step"], config["training"]["gradient_accumulation_steps"])
        rng = validate_rng(state["torch_rng_state"], state.get("cuda_rng_state"), torch.Tensor, torch.uint8)
        del state
        if digest(path) != before or status["checkpoint_sha256"] != before:
            raise ValueError("Resume bytes changed during read-only audit")
        stages.append({"stage": stage, "archive": str(path.relative_to(root)), "pinned_config_sha256": digest(config_path),
                       "lineage": lineage, "resume": status, "history": history, "rng": rng})
        print(f"Verified {stage}: step {status['resume_step']}", file=sys.stderr, flush=True)
    evidence.assert_sources_unchanged()
    print(json.dumps({"run_id": run.name, "audited_at_utc": datetime.now(UTC).isoformat(),
                      "manifest_sha256": digest(evidence.manifest_path), "auditor_sha256": digest(__file__),
                      "pinned_sources_and_dependencies_unchanged": True,
                      "inputs": {"registered_files": len(manifest["data_files"]), "full_sha256_recomputed": args.rehash_inputs,
                                 "bytes_rehashed": input_bytes, "excludes": "Evaluation-generated controlled videos not in original data_files"},
                      "stages": stages, "all_six_training_archives_and_resumes_verified": True,
                      "goal_completion_proven": False,
                      "limits": "Training/archive/config/finite-state/RNG-structure/history/input-byte audit only. No actual restart, CUDA determinism, semantic capability, dataset deduplication, visual QA or document delivery certification."},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
