"""Read and verify archived VLM evaluation for an actual Terminal screenshot."""
import argparse
import hashlib
import json
import re
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,79}", args.run_id):
        raise ValueError("Invalid run ID")
    run = ROOT / "artifacts/runs" / args.run_id
    events = [json.loads(line) for line in (run / "events.jsonl").read_text().splitlines()]
    if not any(e.get("kind") == "stage_completed" and e.get("stage") == "vlm-sft" for e in events):
        raise ValueError("Final VLM stage is not completed")
    entries = [a for e in events if e.get("kind") == "artifacts_archived" and e.get("stage") == "vlm-sft"
               for a in e["artifacts"] if Path(a["archive"]).name == "vlm-final.json"]
    if len(entries) != 1:
        raise ValueError("Ambiguous fixed evaluation")
    path = (ROOT / entries[0]["archive"]).resolve()
    if not path.is_relative_to((run / "stages/vlm-sft").resolve()):
        raise ValueError("Evaluation path escapes stage archive")
    data = path.read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    if digest != entries[0]["sha256"]:
        raise ValueError("Archived evaluation checksum mismatch")
    result = json.loads(data)
    cases = result["qualitative"]
    if len(cases) != 7:
        raise ValueError("Expected all seven fixed image cases")
    print("MINIMIND | ARCHIVED FINAL VLM EVALUATION REVIEW")
    print("Review UTC:", datetime.now(UTC).isoformat())
    print("Run:", args.run_id)
    print("Evaluation SHA256:", digest)
    print("Validation loss (128 holdout rows):", result["validation_loss"])
    v = result["visual_ablation"]
    print("Keyword recall normal / counterfactual:", v["correct_image_keyword_recall"], "/", v["counterfactual_keyword_recall"])
    print("ID  IMAGES  NORMAL  COUNTERFACTUAL  ANSWER-CHANGED")
    for index, row in enumerate(cases, 1):
        print(f"{index:02d}  {row['image_count']:>3}     {row['keyword_recall']:.2f}       "
              f"{row['counterfactual_keyword_recall']:.2f}         {row['completion_changed_on_counterfactual']}")
    print("Language PPL:", result["language_regression"]["corpus"]["validation_perplexity"])
    print("Case 02 prompt:", cases[1]["prompt"])
    print("Case 02 normal completion (entire saved text):")
    print(cases[1]["completion"].replace("\x1b", "\\x1b"))
    print("The actual fixed input is a rainbow umbrella; this answer hallucinates an animal.")
    print("Keyword recall / answer changes do not establish semantic or image-order accuracy.")
    print("Read-only review, not new inference. Video evaluation / full experiment still separate.")


if __name__ == "__main__":
    main()
