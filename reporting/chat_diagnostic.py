"""Frozen supplemental chat diagnostic; does not alter training or baseline eval.

Scoring is standard-library only. Inference uses the existing server environment.
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import platform
import re
import shutil
import sys
import time
from datetime import UTC, datetime
from pathlib import Path


def sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def score_case(case: dict, prediction: str) -> dict:
    prediction = prediction.strip()
    if case["scorer"] == "exact":
        passed = prediction in case["references"]
        return {"score": int(passed), "reason": "reference_match" if passed else "unmatched_reference"}
    if case["scorer"] != "json_object":
        raise ValueError("Unknown scorer")

    def unique_object(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("Duplicate JSON key")
            result[key] = value
        return result

    def reject_constant(value):
        raise ValueError(f"Non-JSON constant {value}")

    try:
        value = json.loads(prediction, object_pairs_hook=unique_object, parse_constant=reject_constant)
    except (json.JSONDecodeError, ValueError):
        return {"score": 0, "reason": "invalid_or_nonunique_json"}
    passed = isinstance(value, dict) and value == case["reference_object"]
    return {"score": int(passed), "reason": "reference_match" if passed else "object_mismatch"}


def load_protocol(path: Path) -> dict:
    protocol = json.loads(path.read_text(encoding="utf-8"))
    if protocol["schema"] != 1 or protocol["protocol_id"] != "chat-diagnostic-v1":
        raise ValueError("Unsupported protocol; new protocols need an explicit new version")
    if len(protocol["cases"]) != 8 or len({c["id"] for c in protocol["cases"]}) != 8:
        raise ValueError("Protocol requires eight unique cases")
    for case in protocol["cases"]:
        reference = case.get("references", [None])[0]
        if case["scorer"] == "json_object":
            reference = json.dumps(case["reference_object"], ensure_ascii=False)
        if not reference or score_case(case, reference)["score"] != 1:
            raise ValueError("Invalid reference or scoring rule")
    expected = {"max_new_tokens": 80, "temperature": 0.0, "stop_at_tokenizer_eos": True,
                "precision": "float32", "kv_cache": False}
    if protocol["generation"] != expected:
        raise ValueError("Generation settings changed without a protocol version change")
    return protocol


def now() -> str:
    return datetime.now(UTC).isoformat()


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def evaluate(args) -> None:
    import torch
    import yaml
    from transformers import AutoTokenizer

    run_dir = args.run_dir.resolve()
    root = run_dir.parents[2]
    sys.path.insert(0, str(root / "src"))
    from minimind_lab.llm import MiniMindConfig, MiniMindForCausalLM
    from minimind_lab.training.evidence import RunEvidence

    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,79}", args.eval_id):
        raise ValueError("Invalid evaluation run ID")
    protocol_path = args.protocol.resolve()
    protocol = load_protocol(protocol_path)
    expected_protocol = run_dir / "supplemental-protocols/chat-diagnostic-v1.json"
    if not expected_protocol.is_file() or sha256(protocol_path) != sha256(expected_protocol):
        raise ValueError("Register and freeze this exact supplemental protocol before inference")
    evidence = RunEvidence(root, run_dir.name)
    evidence.assert_sources_unchanged()
    manifest = json.loads((run_dir / "manifest.json").read_text())
    checkpoint = args.checkpoint.resolve()
    # Read and fingerprint the same inode, even if a pathname is atomically replaced.
    with checkpoint.open("rb") as handle:
        checkpoint_hash = hashlib.file_digest(handle, "sha256").hexdigest()
        handle.seek(0)
        state = torch.load(handle, map_location="cpu", weights_only=False)
    events = [json.loads(line) for line in (run_dir / "events.jsonl").read_text().splitlines()]
    registered = {e["protocol_sha256"] for e in events if e.get("kind") == "evaluation_protocol_registered"
                  and e.get("protocol_id") == protocol["protocol_id"]}
    if registered != {sha256(protocol_path)}:
        raise ValueError("Protocol must match one unambiguous immutable registration event")
    correction_path = run_dir / "supplemental-protocols/chat-diagnostic-v1-provenance.json"
    correction = json.loads(correction_path.read_text())
    if correction["protocol_sha256"] != sha256(protocol_path):
        raise ValueError("Provenance correction is for another protocol fingerprint")
    correction_hashes = {e["correction_sha256"] for e in events
                         if e.get("kind") == "evaluation_protocol_provenance_corrected"
                         and e.get("protocol_id") == protocol["protocol_id"]}
    if correction_hashes != {sha256(correction_path)}:
        raise ValueError("Provenance correction must be frozen and registered")
    allowed_stages = {"llm-pretrain", "llm-sft", "vlm-sft", "video-omni-sft"}
    if args.stage not in allowed_stages:
        raise ValueError("Supplemental chat comparisons use the four designated language checkpoints")
    if args.probe:
        if not checkpoint.is_relative_to(run_dir / "diagnostics") or not args.snapshot_audit:
            raise ValueError("Probe needs a run-scoped diagnostic snapshot and its extraction audit")
        audit = json.loads(args.snapshot_audit.read_text())
        if audit["sha256"] != checkpoint_hash or Path(audit["target"]).resolve() != checkpoint:
            raise ValueError("Probe extraction audit does not identify this exact snapshot")
        expected_name = {"llm-pretrain": "llm-64m-pretrain-cuda.resume.pt",
                         "llm-sft": "llm-64m-sft-cuda.resume.pt",
                         "vlm-sft": "vlm-sft-cuda.resume.pt", "video-omni-sft": "video-omni-sft-cuda.resume.pt"}
        if Path(audit["source"]).name != expected_name[args.stage]:
            raise ValueError("Probe stage label does not match audited source")
        checkpoint_step = audit["source_step"]
    else:
        archives = [a for e in events if e.get("kind") == "artifacts_archived" and e.get("stage") == args.stage
                    for a in e["artifacts"]]
        bound = any(a["sha256"] == checkpoint_hash and (root / a["archive"]).resolve() == checkpoint for a in archives)
        complete = any(e.get("kind") == "stage_completed" and e.get("stage") == args.stage for e in events)
        if not bound or not complete:
            raise ValueError("Non-probe evaluation requires the completed stage's exact archived checkpoint")
        checkpoint_step = None
    language_config = yaml.safe_load((run_dir / "configs/llm/pretrain-cuda.yaml").read_text())
    config = MiniMindConfig(**language_config["model"])
    if state.get("config"):
        supplied = state["config"]["model"]
        fields = {f.name for f in dataclasses.fields(MiniMindConfig)}
        actual = MiniMindConfig(**{key: value for key, value in supplied.items() if key in fields})
        if actual != config:
            raise ValueError("Checkpoint language architecture differs from pinned language model")
    weights = state["model"]
    if args.stage in {"vlm-sft", "video-omni-sft"}:
        prefix = "language_model."
        weights = {key.removeprefix(prefix): value for key, value in weights.items() if key.startswith(prefix)}
    device = torch.device(args.device)
    if args.probe and device.type != "cpu":
        raise ValueError("Diagnostic probes use CPU to avoid competing with formal GPU training")
    model = MiniMindForCausalLM(config).to(device).eval()
    model.load_state_dict(weights, strict=True)
    del state, weights
    tokenizer_path = root / language_config["tokenizer"]["path"]
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
    if tokenizer.eos_token_id is None:
        raise ValueError("Protocol requires an EOS token")
    output = run_dir / "supplemental-evaluations" / args.eval_id
    output.mkdir(parents=True, exist_ok=False)
    shutil.copy2(protocol_path, output / "protocol.json")
    shutil.copy2(correction_path, output / "protocol-provenance.json")
    shutil.copy2(Path(__file__), output / "evaluator.py")
    provenance = {
        "schema": 1, "status": "running", "started_at_utc": now(), "eval_id": args.eval_id,
        "training_run_id": manifest["run_id"], "training_commit": manifest["code_commit"],
        "protocol_id": protocol["protocol_id"], "protocol_sha256": sha256(output / "protocol.json"),
        "evaluator_sha256": sha256(output / "evaluator.py"), "stage": args.stage,
        "checkpoint": str(checkpoint.relative_to(root)), "checkpoint_sha256": checkpoint_hash,
        "checkpoint_step": checkpoint_step, "probe_only": args.probe,
        "device": str(device), "python": platform.python_version(), "torch": torch.__version__,
        "tokenizer_files": [{"path": item["path"], "sha256": item["sha256"]} for item in manifest["data_files"]
                            if item["path"].startswith(language_config["tokenizer"]["path"] + "/")],
        "timing_scope": "Greedy single-request latency, explicit CUDA synchronization, includes EOS decision; no warmup or repetition, not a hardware benchmark",
        "generation": protocol["generation"], "data_role": correction["corrected_data_role"],
        "protocol_provenance_sha256": sha256(output / "protocol-provenance.json"), "limits": protocol["limits"],
    }
    write_json(output / "manifest.json", provenance)
    evidence.event("supplemental_evaluation_started", eval_id=args.eval_id, protocol_sha256=provenance["protocol_sha256"],
                   probe_only=args.probe, stage=args.stage, checkpoint_sha256=checkpoint_hash)
    predictions = []
    try:
        with torch.inference_mode():
            for case in protocol["cases"]:
                rendered = tokenizer.apply_chat_template([{"role": "user", "content": case["prompt"]}],
                                                         tokenize=False, add_generation_prompt=True)
                prompt_ids = tokenizer(rendered).input_ids
                inputs = torch.tensor([prompt_ids], dtype=torch.long, device=device)
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                started = time.perf_counter()
                completion_ids = []
                stopped_at_eos = False
                for _ in range(protocol["generation"]["max_new_tokens"]):
                    logits = model(inputs)["logits"][0, -1]
                    token = int(logits.argmax().item())
                    completion_ids.append(token)
                    if token == tokenizer.eos_token_id:
                        stopped_at_eos = True
                        break
                    inputs = torch.cat([inputs, torch.tensor([[token]], device=device)], dim=1)
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                seconds = time.perf_counter() - started
                answer = tokenizer.decode(completion_ids, skip_special_tokens=True)
                prediction = {
                    **case, "rendered_prompt": rendered, "prompt_token_ids": prompt_ids,
                    "completion": answer, "completion_token_ids": completion_ids,
                    "generated_tokens_including_eos": len(completion_ids), "stopped_at_eos": stopped_at_eos,
                    "seconds": seconds, "scoring": score_case(case, answer),
                }
                predictions.append(prediction)
                with (output / "predictions.jsonl").open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(prediction, ensure_ascii=False, allow_nan=False) + "\n")
                print(json.dumps({"case": case["id"], "score": prediction["scoring"]["score"],
                                  "tokens": len(completion_ids)}, ensure_ascii=False), flush=True)
        evidence.assert_sources_unchanged()
        passed = sum(p["scoring"]["score"] for p in predictions)
        summary = {"cases": len(predictions), "passed": passed, "reference_constrained_pass_rate": passed / len(predictions),
                   "per_case": [{"id": p["id"], "category": p["category"], **p["scoring"]} for p in predictions],
                   "probe_only": args.probe, "not_a_general_instruction_benchmark": True}
        write_json(output / "summary.json", summary)
        provenance.update(status="completed", completed_at_utc=now(), predictions_sha256=sha256(output / "predictions.jsonl"),
                          summary_sha256=sha256(output / "summary.json"))
        write_json(output / "manifest.json", provenance)
        evidence.event("supplemental_evaluation_completed", eval_id=args.eval_id, probe_only=args.probe,
                       manifest=str((output / "manifest.json").relative_to(root)),
                       manifest_sha256=sha256(output / "manifest.json"), cases=len(predictions))
        print(json.dumps({"directory": str(output), "status": "completed", **summary}, ensure_ascii=False))
    except Exception as error:
        provenance.update(status="failed", failed_at_utc=now(), error_type=type(error).__name__, error=str(error))
        write_json(output / "manifest.json", provenance)
        evidence.event("supplemental_evaluation_failed", eval_id=args.eval_id, error_type=type(error).__name__)
        raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--stage", required=True)
    parser.add_argument("--eval-id", required=True)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--probe", action="store_true")
    parser.add_argument("--snapshot-audit", type=Path)
    evaluate(parser.parse_args())


if __name__ == "__main__":
    main()
