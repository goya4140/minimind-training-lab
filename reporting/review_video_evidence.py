"""Verify and display the archived video evaluation, without new inference."""
import argparse
import hashlib
import json
import re
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load_result(run_id):
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,79}", run_id):
        raise ValueError("Invalid run ID")
    run = ROOT / "artifacts/runs" / run_id
    events = [json.loads(line) for line in (run / "events.jsonl").read_text().splitlines()]
    stage = "video-omni-sft"
    if not any(e.get("kind") == "stage_completed" and e.get("stage") == stage for e in events):
        raise ValueError("Final video stage is not completed")
    entries = [a for e in events if e.get("kind") == "artifacts_archived" and e.get("stage") == stage
               for a in e["artifacts"] if Path(a["archive"]).name == "video-omni-final.json"]
    if len(entries) != 1:
        raise ValueError("Ambiguous fixed video evaluation")
    path = (ROOT / entries[0]["archive"]).resolve()
    if not path.is_relative_to((run / "stages" / stage).resolve()):
        raise ValueError("Evaluation path escapes stage archive")
    data = path.read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    if digest != entries[0]["sha256"]:
        raise ValueError("Archived evaluation checksum mismatch")
    result = json.loads(data)
    if result["qivd_generation"]["generated_samples"] != 100 or result["controlled_temporal"]["generated_samples"] != 160:
        raise ValueError("Unexpected fixed generation counts")
    for name in ("qivd_generation", "controlled_temporal"):
        if not result[name]["qualitative"]:
            raise ValueError("Missing saved qualitative cases")
    return result, digest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    result, digest = load_result(args.run_id)
    print("MINIMIND | ARCHIVED FINAL VIDEO EVALUATION REVIEW")
    print("Review UTC:", datetime.now(UTC).isoformat())
    print("Run:", args.run_id)
    print("Evaluation SHA256:", digest)
    print("QIVD loss normal / reversed:", result["test_loss"], "/", result["reversed_frame_test_loss"])
    print("SET          N    EM-N / EM-R    F1-N / F1-R       CHANGED")
    for name, key in (("QIVD", "qivd_generation"), ("Controlled", "controlled_temporal")):
        v = result[key]
        print(f"{name:<12} {v['generated_samples']:>3}  {v['normalized_exact_match']:.3f} / "
              f"{v['reversed_frame_exact_match']:.3f}  {v['token_f1']:.6f} / "
              f"{v['reversed_frame_token_f1']:.6f}  {v['completion_change_rate_on_reversal']:.3f}")
    print("FIRST SAVED CASE PER SET (not selected for a high score):")
    for name, key in (("QIVD", "qivd_generation"), ("Controlled", "controlled_temporal")):
        row = result[key]["qualitative"][0]
        print(name, "ID:", row["id"], "| Question:", row["question"])
        print("Reference short answer:", row["short_answer"])
        print("Normal:", row["normal_completion"].replace("\x1b", "\\x1b"))
        print("Reversed:", row["reversed_completion"].replace("\x1b", "\\x1b"))
    print("Reverse scores use ORIGINAL references, not reversed-video truth.")
    print("Zero change here does not prove temporal understanding. Loss is a batch mean.")
    print("Read-only review of saved evaluation, not new inference or full goal completion.")


if __name__ == "__main__":
    main()
