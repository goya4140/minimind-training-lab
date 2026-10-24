"""Export four deterministic controlled-video input/output examples, CPU only.

No inference, data regeneration, training changes, or edits to baseline outputs.
Contact sheets are derived from decoded input frames, not generated model images.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import sys
import textwrap
from datetime import UTC, datetime
from pathlib import Path

FAMILIES = ("size-change", "motion-vertical", "motion-horizontal", "event-order")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def select_cases(rows):
    if len(rows) != 160 or len({r["id"] for r in rows}) != 160:
        raise ValueError("Expected 160 unique controlled cases")
    if {r["category"] for r in rows} != set(FAMILIES):
        raise ValueError("Unexpected controlled families")
    if any(sum(r["category"] == family for r in rows) != 40 for family in FAMILIES):
        raise ValueError("Expected 40 cases per family")
    # Evaluation order, not scores or generated content, determines selection.
    return [next(r for r in rows if r["category"] == family) for family in FAMILIES]


def contained_file(root, relative):
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()) or not path.is_file():
        raise ValueError("Missing or escaping evidence file")
    return path


def sheet(frames, timestamps, row, output, font_path):
    from PIL import Image, ImageDraw, ImageFont

    font = ImageFont.truetype(str(font_path), 22)
    small = ImageFont.truetype(str(font_path), 18)
    cell, pad = 164, 24
    lines = [f"Controlled video ID {row['id']} | {row['category']}",
             "Actual decoded INPUT frames, not generated outputs or a Terminal screenshot.",
             "Normal order (left to right):"]
    # The strip uses all eight sampled frames, not an illustrative reconstruction.
    canvas = Image.new("RGB", (cell * 8 + pad * 2, 850), "white")
    draw = ImageDraw.Draw(canvas)
    for index, line in enumerate(lines):
        draw.text((pad, 16 + index * 30), line, fill="#182838", font=font)
    for reverse, y in ((False, 118), (True, 345)):
        order = list(range(8))[::-1] if reverse else list(range(8))
        if reverse:
            draw.text((pad, y - 34), "Reversed order (same decoded inputs, reversed):", fill="#182838", font=font)
        for column, index in enumerate(order):
            image = frames[index].copy()
            image.thumbnail((cell - 8, cell - 8), Image.Resampling.LANCZOS)
            x = pad + cell * column
            canvas.paste(image, (x, y))
            draw.text((x, y + cell), f"f{index} t={timestamps[index]:.2f}s", fill="#182838", font=small)
    y = 550
    paragraphs = ["Question: " + row["question"], "Original reference: " + row["short_answer"],
                  "Actual normal answer: " + row["normal_completion"],
                  "Actual reversed answer: " + row["reversed_completion"],
                  "Reverse scores use ORIGINAL references, not reversed-video truth.",
                  "Selection: first evaluated case per family; full answers retained, no new inference."]
    for paragraph in paragraphs:
        for line in textwrap.wrap(paragraph, width=108) or [""]:
            draw.text((pad, y), line, fill="#182838", font=font)
            y += 29
    if y > canvas.height - 12:
        raise ValueError("Contact sheet text overflow; increase layout before export")
    canvas.save(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--eval-id", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--font", type=Path, default=Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"))
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,79}", args.eval_id):
        raise ValueError("Invalid evaluation ID")
    run = args.run_dir.resolve()
    root = run.parents[2]
    sys.path.insert(0, str(root / "src"))
    from minimind_lab.training.evidence import RunEvidence
    from minimind_lab.data.video import decode_uniform_video
    import av
    import pyarrow.parquet as pq

    evidence = RunEvidence(root, run.name)
    if evidence.directory != run:
        raise ValueError("Not a canonical run directory")
    evidence.assert_sources_unchanged()
    folder = run / "supplemental-evaluations" / args.eval_id
    manifest = json.loads((folder / "manifest.json").read_text())
    if manifest["status"] != "completed" or manifest["run_id"] != run.name:
        raise ValueError("Completed run-bound replay required")
    if manifest["kind"] != "fixed-video-evaluation-full-output-replay-v1":
        raise ValueError("Unexpected replay protocol")
    receipts = [r for r in manifest["predictions"] if r["dataset"] == "controlled_temporal"]
    if len(receipts) != 1:
        raise ValueError("Ambiguous controlled prediction receipt")
    receipt = receipts[0]
    predictions = contained_file(folder, receipt["file"])
    if digest(predictions) != receipt["sha256"]:
        raise ValueError("Prediction checksum mismatch")
    selected = select_cases([json.loads(line) for line in predictions.read_text().splitlines()])
    data_root = root / "data/eval/temporal-video"
    metadata_path = data_root / "metadata.parquet"
    metadata = pq.read_table(metadata_path).to_pylist()
    if len(metadata) != 160 or len({r["id"] for r in metadata}) != 160:
        raise ValueError("Unexpected controlled metadata")
    output = args.output.resolve()
    if not output.is_relative_to(run / "visual-evidence"):
        raise ValueError("Output must be in this run's visual-evidence directory")
    output.mkdir(parents=True, exist_ok=False)
    shutil.copy2(__file__, output / "exporter.py")
    records = []
    for row in selected:
        match = next(m for m in metadata if m["id"] == row["id"])
        if any(match[k] != row[k] for k in ("category", "question", "answer", "short_answer")):
            raise ValueError("Prediction/metadata identity mismatch")
        video = contained_file(data_root, match["video_file_name"])
        # These project-generated clips each contain exactly 8 frames at 4fps.
        with av.open(str(video)) as container:
            decoded = list(container.decode(video=0))
            if len(decoded) != 8 or any(f.pts is None or f.time_base is None for f in decoded):
                raise ValueError("Expected eight timestamped controlled video frames")
            times = [float(f.pts * f.time_base) for f in decoded]
            originals = [f.to_image().convert("RGB") for f in decoded]
        frames = decode_uniform_video(video, 8)
        if any(a.size != b.size or a.tobytes() != b.tobytes() for a, b in zip(frames, originals)):
            raise ValueError("Model decoder differs from independently decoded frame sequence")
        raw = output / f"raw-{row['id']}"
        raw.mkdir()
        frame_records = []
        for i, frame in enumerate(frames):
            path = raw / f"frame-{i}.png"
            frame.save(path)
            frame_records.append({"index": i, "pts_seconds": times[i], "rgb_sha256": hashlib.sha256(frame.tobytes()).hexdigest(),
                                  "file": str(path.relative_to(output)), "png_sha256": digest(path)})
        target = output / f"controlled-{row['id']}-{row['category']}.png"
        sheet(frames, times, row, target, args.font)
        records.append({"case": row, "video": str(data_root.relative_to(root) / match["video_file_name"]), "video_sha256": digest(video),
                        "frames": frame_records, "contact_sheet": target.name, "contact_sheet_sha256": digest(target)})
    result = {"schema": 1, "status": "completed", "created_at_utc": datetime.now(UTC).isoformat(),
              "run_id": run.name, "eval_id": args.eval_id, "replay_manifest_sha256": digest(folder / "manifest.json"),
              "predictions_sha256": receipt["sha256"], "metadata_sha256": digest(metadata_path),
              "exporter_sha256": digest(__file__), "font_sha256": digest(args.font),
              "selection": "First case per family in frozen replay evaluation order, not by score",
              "input_provenance_limit": "Controlled videos were generated by pinned evaluation code, not in original run data_files fingerprints; these are current input hashes, not claimed pre-run hashes.",
              "visual_type": "Contact sheets from actual MP4-decoded input frames plus saved replay text; no model-generated images, no screenshots, no new inference",
              "records": records}
    (output / "case-evidence.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    evidence.assert_sources_unchanged()
    evidence.event("controlled_video_case_evidence_exported", eval_id=args.eval_id, cases=4,
                   manifest=str((output / "case-evidence.json").relative_to(root)), manifest_sha256=digest(output / "case-evidence.json"))
    print(json.dumps({"output": str(output), "cases": 4, "raw_frames_server_only": 32}))


if __name__ == "__main__":
    main()
