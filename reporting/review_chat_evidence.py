"""Read actual completed diagnostic files for a genuine Terminal screenshot."""
import argparse
import hashlib
import json
import re
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--eval-id", required=True)
    args = parser.parse_args()
    for value in (args.run_id, args.eval_id):
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,79}", value):
            raise ValueError("Invalid evidence ID")
    folder = ROOT / "artifacts/runs" / args.run_id / "supplemental-evaluations" / args.eval_id
    manifest = json.loads((folder / "manifest.json").read_text())
    if manifest["status"] != "completed":
        raise ValueError("Diagnostic is not completed")
    for name, key in (("predictions.jsonl", "predictions_sha256"), ("summary.json", "summary_sha256")):
        if hashlib.sha256((folder / name).read_bytes()).hexdigest() != manifest[key]:
            raise ValueError("Diagnostic result checksum mismatch")
    summary = json.loads((folder / "summary.json").read_text())
    predictions = [json.loads(line) for line in (folder / "predictions.jsonl").read_text().splitlines()]
    if len(predictions) != summary["cases"]:
        raise ValueError("Prediction count does not match summary")
    print("MINIMIND CHAT DIAGNOSTIC |", "EARLY PROBE, NOT FINAL" if manifest["probe_only"] else "SUPPLEMENTAL ONLY")
    print("Review UTC:", datetime.now(UTC).isoformat())
    print("Run:", args.run_id)
    print("Evaluation:", args.eval_id)
    print("Protocol:", manifest["protocol_id"], "| CPU/device:", manifest["device"])
    print("Source stage:", manifest["stage"], "| Snapshot step:", manifest["checkpoint_step"])
    print("Evaluation completed UTC:", manifest["completed_at_utc"])
    print("Strict reference matches:", summary["passed"], "/", summary["cases"])
    print("ID   CATEGORY                 MATCH   EOS-STOP   TOKENS")
    for row in predictions:
        print(f"{row['id']}  {row['category']:<24} {row['scoring']['score']:>3}     "
              f"{row['stopped_at_eos']!s:<7}    {len(row['completion_token_ids'])}")
    sample = next(row for row in predictions if row["id"] == "C07")
    print("C07 expected:", " / ".join(sample["references"]))
    print("C07 actual completion (entire saved text):")
    print(sample["completion"].replace("\x1b", "\\x1b"))
    print("Reason:", sample["scoring"]["reason"])
    print("A failed strict match can reflect repetition/format, not absence of relevant words.")
    print("Eight project-authored cases do not establish general capability.")


if __name__ == "__main__":
    main()
