#!/usr/bin/env python3
"""Prepare an isolated repair run, with verified inherited LLM stage evidence."""
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from minimind_lab.training import load_config, verify_checkpoint, verify_resume_checkpoint
from minimind_lab.training.artifacts import file_sha256
from minimind_lab.training.evidence import RunEvidence
from preflight_mps_pipeline import STAGES


def link_resources(parent: Path):
    """Create only new ignored links; never overwrite an existing resource."""
    def link(source, target):
        if target.exists() or target.is_symlink():
            if target.resolve() != source.resolve():
                raise FileExistsError(f"Resource already belongs to another source: {target}")
            return
        target.parent.mkdir(parents=True, exist_ok=True)
        target.symlink_to(source.resolve(), target_is_directory=source.is_dir())

    for relative in ("data/raw", "data/processed", "data/manifests", "assets/models"):
        destination = ROOT / relative
        destination.mkdir(parents=True, exist_ok=True)
        for source in sorted((parent / relative).iterdir()):
            link(source, destination / source.name)
    for source in (parent / "assets/tokenizer").glob("*.json"):
        link(source, ROOT / "assets/tokenizer" / source.name)
    link(parent / ".venv", ROOT / ".venv")
    link(parent / ".cache/huggingface", ROOT / ".cache/huggingface")


def archive_entries(parent, stage):
    events = [json.loads(line) for line in (parent.directory / "events.jsonl").read_text().splitlines()]
    if not any(e.get("kind") == "stage_completed" and e.get("stage") == stage for e in events):
        raise ValueError(f"Parent stage is not complete: {stage}")
    entries = [a for e in events if e.get("kind") == "artifacts_archived" and e.get("stage") == stage
               for a in e["artifacts"]]
    names = [a["source"] for a in entries]
    if len(entries) != 5 or len(set(names)) != 5:
        raise ValueError(f"Parent stage needs five unambiguous archived artifacts: {stage}")
    for entry in entries:
        source = (parent.root / entry["archive"]).resolve()
        if not source.is_relative_to(parent.directory / "stages" / stage):
            raise ValueError("Parent artifact escapes its stage archive")
        if file_sha256(source) != entry["sha256"] or source.stat().st_size != entry["bytes"]:
            raise ValueError("Parent archived artifact fingerprint changed")
    return entries


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-root", required=True, type=Path)
    parser.add_argument("--parent-run-id", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--link-only", action="store_true")
    parser.add_argument("--probe-json", type=Path)
    args = parser.parse_args()
    parent = RunEvidence(args.parent_root, args.parent_run_id)
    if parent.root == ROOT:
        raise ValueError("Use a separate checkout; never mutate the original pinned run")
    parent.assert_sources_unchanged()
    parent_manifest = json.loads(parent.manifest_path.read_text())
    if os.environ.get("CUDA_VISIBLE_DEVICES") != parent_manifest["runtime"]["cuda_visible_devices"]:
        raise ValueError("Select exactly the parent run's authorized GPU")
    link_resources(parent.root)
    if args.link_only:
        print("Shared resources linked without copying data or changing parent sources")
        return
    if not args.probe_json:
        raise ValueError("Require the successful CUDA regression probe JSON")
    probe = json.loads(args.probe_json.read_text())
    if (probe.get("status") != "passed" or probe.get("image_slots") != [64] * 4
            or probe.get("cuda_visible_devices") != os.environ["CUDA_VISIBLE_DEVICES"]
            or probe.get("vlm_data_sha256") != file_sha256(ROOT / "src/minimind_lab/data/vlm.py")
            or not probe.get("projector_changed") or not math.isfinite(probe.get("loss", float("nan")))):
        raise ValueError("CUDA probe does not bind this repaired implementation")
    configs = [c.replace("-mps", "-cuda") for _, _, c in STAGES]
    for name in configs:
        if load_config(ROOT / name) != load_config(parent.root / name):
            raise ValueError("This repair must not change the six-stage hyperparameters")
        output = ROOT / load_config(ROOT / name)["training"]["checkpoint_path"]
        if output.exists() or output.with_suffix(".resume.pt").exists():
            raise FileExistsError("Repair output already exists; do not overwrite or silently re-import")
    inherited = {stage: archive_entries(parent, stage) for stage in ("llm-pretrain", "llm-sft")}
    if shutil.disk_usage(ROOT).free < 12 * 1024**3:
        raise RuntimeError("Less than 12 GiB free; stop before creating formal repair outputs")
    data = sorted({*ROOT.glob("data/raw/*.jsonl"), *ROOT.glob("data/raw/*.parquet"),
                   *ROOT.glob("data/raw/qivd/*"), *ROOT.glob("data/raw/qivd/videos/*.mp4"),
                   *ROOT.glob("assets/tokenizer/*.json"), *ROOT.glob("assets/models/siglip2-base-p32-256-ve/*"),
                   *ROOT.glob("data/manifests/*.json"), *ROOT.glob("data/processed/qivd-siglip2-patch-fp16.*")})
    evidence = RunEvidence(ROOT, args.run_id)
    evidence.create(configs, [p for p in data if p.is_file() and not p.name.endswith((".part", ".lock"))])
    shutil.copy2(args.probe_json, evidence.directory / "cuda-repair-probe.json")
    evidence.event("repair_lineage_registered", parent_root=str(parent.root), parent_run_id=parent.run_id,
                   parent_code_commit=parent_manifest["code_commit"],
                   policy="full-visual-slots-answer-window-v2", multimodal_alignment_restarted_from_step=0,
                   failed_alignment_optimizer_inherited=False, cuda_probe_sha256=file_sha256(args.probe_json))
    for stage, entries in inherited.items():
        paths = []
        for entry in entries:
            destination = ROOT / entry["source"]
            if not destination.resolve().is_relative_to(ROOT / "artifacts"):
                raise ValueError("Inherited output escapes repair artifacts")
            destination.parent.mkdir(parents=True, exist_ok=True)
            with (parent.root / entry["archive"]).open("rb") as source, destination.open("xb") as target:
                shutil.copyfileobj(source, target)
            shutil.copystat(parent.root / entry["archive"], destination)
            if file_sha256(destination) != entry["sha256"]:
                raise ValueError("Inherited copy checksum mismatch")
            paths.append(destination)
        config = load_config(ROOT / ("configs/llm/pretrain-cuda.yaml" if stage == "llm-pretrain"
                                    else "configs/llm/sft-cuda.yaml"))
        checkpoint = ROOT / config["training"]["checkpoint_path"]
        if not verify_checkpoint(checkpoint)["all_finite"]:
            raise ValueError("Inherited model contains non-finite values")
        audit = verify_resume_checkpoint(checkpoint.with_suffix(".resume.pt"), expected_config=config,
                                         require_complete=True)
        evidence.archive(stage, paths)
        evidence.event("stage_inherited", stage=stage, parent_run_id=parent.run_id,
                       artifacts=entries, resume_audit=audit, trained_in_this_run=False,
                       original_metric_run_id_preserved=True)
    evidence.assert_sources_unchanged()
    parent.assert_sources_unchanged()
    print(json.dumps({"run_id": args.run_id, "status": "prepared-not-started",
                      "inherited_stages": list(inherited), "remaining_stages": 4}))


if __name__ == "__main__":
    main()
