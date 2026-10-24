"""CPU-only exact final-weight/resume comparisons and video freezing audit.

No model construction, GPU initialization, optimizer step, or checkpoint writes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

from audit_stage_lineage import STAGES, audit_stage, events


def video_group(name):
    if name.startswith("temporal_adapter."):
        return "temporal_adapter"
    if name.startswith("video_projector."):
        return "video_projector"
    if name.startswith(("language_model.layers.0.", "language_model.layers.7.")):
        return "language_boundary"
    if name.startswith("language_model."):
        return "frozen_language"
    raise ValueError("Unexpected saved video state key")


def compare_states(before, after, equal):
    if set(before) != set(after):
        raise ValueError("Model state key sets differ")
    changed = []
    for name, value in after.items():
        original = before[name]
        if value.shape != original.shape or value.dtype != original.dtype:
            raise ValueError("Model state shape/dtype differs")
        if not equal(value, original):
            changed.append(name)
    return changed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    run = args.run_dir.resolve()
    root = run.parents[2]
    sys.path.insert(0, str(root / "src"))
    import torch
    from minimind_lab.training.evidence import RunEvidence

    evidence = RunEvidence(root, run.name)
    if evidence.directory != run:
        raise ValueError("Not a canonical run directory")
    evidence.assert_sources_unchanged()
    ledger = events(run)

    def load(stage, filename):
        entries = [a for e in ledger if e.get("kind") == "artifacts_archived" and e.get("stage") == stage
                   for a in e["artifacts"] if Path(a["archive"]).name == filename]
        if len(entries) != 1:
            raise ValueError("Ambiguous archived weight file")
        entry = entries[0]
        path = (root / entry["archive"]).resolve()
        if not path.is_relative_to((run / "stages" / stage).resolve()):
            raise ValueError("Weight file escapes stage archive")
        with path.open("rb") as handle:
            digest = hashlib.file_digest(handle, "sha256").hexdigest()
            if digest != entry["sha256"]:
                raise ValueError("Weight archive hash mismatch")
            handle.seek(0)
            weights = torch.load(handle, map_location="cpu", weights_only=False)["model"]
        if not weights or not all(isinstance(t, torch.Tensor) and bool(torch.isfinite(t).all()) for t in weights.values()):
            raise ValueError("Missing, non-tensor or non-finite saved model state")
        return weights, digest

    stages = []
    for stage, (name, _, _) in STAGES.items():
        lineage = audit_stage(run, ledger, stage)
        if lineage["status"] != "archive-verified":
            raise ValueError("Completed stage required")
        model, model_hash = load(stage, name + ".pt")
        resume, resume_hash = load(stage, name + ".resume.pt")
        if compare_states(model, resume, torch.equal):
            raise ValueError("Model-only and resume weights differ")
        stages.append({"stage": stage, "model_only_sha256": model_hash, "resume_sha256": resume_hash,
                       "state_keys_compared": len(model), "all_keys_exactly_equal": True})
        del model, resume
        print("Model/resume exact match:", stage, file=sys.stderr, flush=True)
    llm, llm_hash = load("llm-sft", "llm-64m-sft-cuda.pt")
    alignment, alignment_hash = load("video-omni-alignment", "video-omni-alignment-cuda.pt")
    final, final_hash = load("video-omni-sft", "video-omni-sft-cuda.pt")
    language = {k.removeprefix("language_model."): t for k, t in alignment.items() if k.startswith("language_model.")}
    if compare_states(llm, language, torch.equal):
        raise ValueError("Video alignment changed frozen LLM SFT weights")
    changed = set(compare_states(alignment, final, torch.equal))
    groups = {k: {"changed": [], "unchanged": []} for k in
              ("temporal_adapter", "video_projector", "language_boundary", "frozen_language")}
    for name in final:
        groups[video_group(name)]["changed" if name in changed else "unchanged"].append(name)
    if groups["frozen_language"]["changed"]:
        raise ValueError("Video SFT changed frozen language weights")
    for name in ("temporal_adapter", "video_projector", "language_boundary"):
        if not groups[name]["changed"]:
            raise ValueError("No observed update in a permitted video SFT component")
    evidence.assert_sources_unchanged()
    print(json.dumps({"run_id": run.name, "audited_at_utc": datetime.now(UTC).isoformat(),
                      "auditor_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                      "model_resume_comparisons": stages,
                      "video_freezing": {"llm_sft_sha256": llm_hash, "video_alignment_sha256": alignment_hash,
                                         "video_sft_sha256": final_hash, "alignment_language_keys_equal_llm": len(language),
                                         "groups": groups},
                      "limits": "Exact equality of final saved tensor values only; not optimizer equivalence between stages, gradient trace, external SigLIP tensor comparison, actual restart, capability, or final document delivery."}, indent=2))


if __name__ == "__main__":
    main()
