"""CPU-only comparison of archived LLM/Alignment and an immutable SFT snapshot.

Prints an audit without changing training, files, tensors or optimizer state.
Does not prove future freezing, external vision-weight equality or capability.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import UTC, datetime
from pathlib import Path


def group(name):
    if name.startswith("vision_projector."):
        return "projector"
    if name.startswith(("language_model.layers.0.", "language_model.layers.7.")):
        return "language_boundary"
    return "frozen_language"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--sft-snapshot", type=Path, required=True)
    args = parser.parse_args()
    run = args.run_dir.resolve()
    root = run.parents[2]
    sys.path.insert(0, str(root / "src"))
    import torch
    from minimind_lab.training import load_config, verify_resume_checkpoint
    from minimind_lab.training.evidence import RunEvidence

    evidence = RunEvidence(root, run.name)
    if evidence.directory != run:
        raise ValueError("Not a canonical run directory")
    evidence.assert_sources_unchanged()
    events = [json.loads(line) for line in (run / "events.jsonl").read_text().splitlines()]

    def archived(directory, stage, filename):
        ledger = [json.loads(line) for line in (directory / "events.jsonl").read_text().splitlines()]
        if not any(e.get("kind") == "stage_completed" and e.get("stage") == stage for e in ledger):
            raise ValueError("Reference stage is incomplete")
        entries = [a for e in ledger if e.get("kind") == "artifacts_archived" and e.get("stage") == stage
                   for a in e["artifacts"] if Path(a["archive"]).name == filename]
        if len(entries) != 1:
            raise ValueError("Ambiguous reference archive")
        path = (directory.parents[2] / entries[0]["archive"]).resolve()
        if not path.is_relative_to(directory / "stages" / stage):
            raise ValueError("Reference archive escapes stage directory")
        with path.open("rb") as handle:
            digest = hashlib.file_digest(handle, "sha256").hexdigest()
            if digest != entries[0]["sha256"]:
                raise ValueError("Reference archive hash mismatch")
            handle.seek(0)
            return torch.load(handle, map_location="cpu", weights_only=False)["model"], digest

    lineage = [e for e in events if e.get("kind") == "repair_lineage_registered"]
    if len(lineage) != 1:
        raise ValueError("Expected one explicit parent lineage")
    parent = Path(lineage[0]["parent_root"]).resolve() / "artifacts/runs" / lineage[0]["parent_run_id"]
    parent_manifest = json.loads((parent / "manifest.json").read_text())
    if parent_manifest["code_commit"] != lineage[0]["parent_code_commit"] or parent_manifest["run_id"] != parent.name:
        raise ValueError("Parent identity mismatch")
    llm, llm_hash = archived(parent, "llm-sft", "llm-64m-sft-cuda.pt")
    alignment, alignment_hash = archived(run, "vlm-alignment", "vlm-alignment-cuda.pt")
    snapshot = args.sft_snapshot.resolve()
    if not snapshot.is_relative_to(run / "audits"):
        raise ValueError("Use a retained run-scoped immutable audit snapshot, not the live checkpoint path")
    audit = verify_resume_checkpoint(snapshot, expected_config=load_config(root / "configs/vlm/sft-cuda.yaml"))
    state = torch.load(snapshot, map_location="cpu", weights_only=False)
    if not state["history"] or {r.get("run_id") for r in state["history"]} != {run.name}:
        raise ValueError("SFT snapshot history belongs to another run")
    current = state["model"]
    if set(current) != set(alignment):
        raise ValueError("Alignment and SFT state keys differ")
    language = {name.removeprefix("language_model."): tensor for name, tensor in alignment.items()
                if name.startswith("language_model.")}
    if set(language) != set(llm) or not all(torch.equal(language[name], llm[name]) for name in llm):
        raise ValueError("Alignment modified the frozen parent language weights")
    results = {name: {"keys": 0, "changed": [], "unchanged": []}
               for name in ("projector", "language_boundary", "frozen_language")}
    for name, tensor in current.items():
        before = alignment[name]
        if tensor.shape != before.shape or tensor.dtype != before.dtype:
            raise ValueError("State shape/dtype changed")
        label = group(name)
        results[label]["keys"] += 1
        results[label]["unchanged" if torch.equal(tensor, before) else "changed"].append(name)
    if results["frozen_language"]["changed"] or not audit["all_finite"]:
        raise ValueError("Frozen weights changed or SFT state is nonfinite")
    if not results["projector"]["changed"] or not results["language_boundary"]["changed"]:
        raise ValueError("Expected evidence of updates to both permitted component groups")
    evidence.assert_sources_unchanged()
    print(json.dumps({"kind": "vlm-freezing-state-comparison", "run_id": run.name,
                      "audited_at_utc": datetime.now(UTC).isoformat(), "sft_snapshot": str(snapshot.relative_to(root)),
                      "parent_llm_checkpoint_sha256": llm_hash, "alignment_checkpoint_sha256": alignment_hash,
                      "alignment_language_keys_equal_parent": len(language), "sft_resume_audit": audit,
                      "groups": results, "tool_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                      "limits": "Snapshot equality only, not an actual restart, gradient-path trace, "
                      "external vision-weight comparison, future-state guarantee or model capability evaluation."}, indent=2))


if __name__ == "__main__":
    main()
