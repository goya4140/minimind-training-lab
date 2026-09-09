#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPO = "Qualcomm-AI-Research/QIVD"
REVISION = "c5376ab0b9fd3643545a1503413aee64f26ba22a"
BASE_URL = f"https://huggingface.co/datasets/{REPO}/resolve/{REVISION}/"


def download_file(relative_path: str, destination: Path, retries: int, base_delay: float) -> None:
    if destination.is_file() and destination.stat().st_size > 0:
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".part")
    for attempt in range(retries):
        offset = partial.stat().st_size if partial.exists() else 0
        request = urllib.request.Request(BASE_URL + urllib.parse.quote(relative_path))
        if offset:
            request.add_header("Range", f"bytes={offset}-")
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                append = offset > 0 and response.status == 206
                mode = "ab" if append else "wb"
                with partial.open(mode) as handle:
                    while chunk := response.read(1024 * 1024):
                        handle.write(chunk)
            os.replace(partial, destination)
            return
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError) as error:
            if attempt + 1 == retries:
                raise RuntimeError(f"failed to download {relative_path} after {retries} attempts") from error
            retry_after = float(getattr(error, "headers", {}).get("Retry-After", 0) or 0)
            delay = min(60.0, max(retry_after, base_delay * (2**attempt)) + random.random())
            print(json.dumps({"file": relative_path, "attempt": attempt + 1, "error": str(error), "wait": delay}))
            time.sleep(delay)


def dataset_manifest(root: Path, files: list[str]) -> dict:
    entries = []
    for relative_path in sorted(files):
        path = root / relative_path
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        entries.append({"path": relative_path, "bytes": path.stat().st_size, "sha256": digest})
    aggregate = hashlib.sha256(
        "\n".join(f"{entry['path']} {entry['bytes']} {entry['sha256']}" for entry in entries).encode()
    ).hexdigest()
    return {
        "repository": REPO,
        "revision": REVISION,
        "video_count": len(entries),
        "total_video_bytes": sum(entry["bytes"] for entry in entries),
        "aggregate_sha256": aggregate,
        "files": entries,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Resumably fetch the pinned QIVD video corpus.")
    parser.add_argument("--output", default="data/raw/qivd")
    parser.add_argument("--retries", type=int, default=12)
    parser.add_argument("--base-delay", type=float, default=5.0)
    args = parser.parse_args()
    root = ROOT / args.output
    root.mkdir(parents=True, exist_ok=True)
    for relative_path in ("metadata.parquet", "LICENSE", "license.pdf"):
        download_file(relative_path, root / relative_path, args.retries, args.base_delay)

    import pyarrow.parquet as pq

    rows = pq.read_table(root / "metadata.parquet", columns=["video_file_name"]).to_pylist()
    video_files = sorted({row["video_file_name"] for row in rows})
    for index, relative_path in enumerate(video_files, start=1):
        download_file(relative_path, root / relative_path, args.retries, args.base_delay)
        if index == 1 or index % 50 == 0:
            completed = sum((root / path).is_file() for path in video_files)
            print(json.dumps({"completed": completed, "total": len(video_files)}), flush=True)

    manifest = dataset_manifest(root, video_files)
    manifest_path = ROOT / "data/manifests/qivd.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = manifest_path.with_suffix(".tmp")
    temporary.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, manifest_path)
    print(json.dumps({key: value for key, value in manifest.items() if key != "files"}, indent=2))


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
