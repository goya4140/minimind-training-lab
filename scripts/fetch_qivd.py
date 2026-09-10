#!/usr/bin/env python3
from __future__ import annotations

import argparse
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
sys.path.insert(0, str(ROOT / "src"))

from minimind_lab.data.integrity import (
    QIVD_REPOSITORY,
    QIVD_REVISION,
    QIVD_ROOT_FILES,
    file_matches,
    verified_dataset_manifest,
)

REPO = QIVD_REPOSITORY
REVISION = QIVD_REVISION
BASE_URL = f"https://huggingface.co/datasets/{REPO}/resolve/{REVISION}/"
ROOT_FILES = QIVD_ROOT_FILES


def download_file(
    relative_path: str,
    destination: Path,
    retries: int,
    base_delay: float,
    expected_size: int | None = None,
    expected_sha256: str | None = None,
) -> None:
    if file_matches(destination, expected_size, expected_sha256):
        return
    destination.unlink(missing_ok=True)
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
            if not file_matches(destination, expected_size, expected_sha256):
                destination.unlink(missing_ok=True)
                raise urllib.error.ContentTooShortError(
                    f"downloaded file failed size/SHA-256 verification: {relative_path}",
                    content=None,
                )
            return
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError) as error:
            if attempt + 1 == retries:
                raise RuntimeError(f"failed to download {relative_path} after {retries} attempts") from error
            retry_after = float(getattr(error, "headers", {}).get("Retry-After", 0) or 0)
            delay = min(60.0, max(retry_after, base_delay * (2**attempt)) + random.random())
            print(json.dumps({"file": relative_path, "attempt": attempt + 1, "error": str(error), "wait": delay}))
            time.sleep(delay)


def upstream_video_files() -> dict[str, dict[str, int | str]]:
    from huggingface_hub import HfApi

    entries = {}
    for item in HfApi().list_repo_tree(
        REPO, path_in_repo="videos", recursive=False, expand=True, revision=REVISION, repo_type="dataset"
    ):
        if getattr(item, "lfs", None) is not None and str(item.path).startswith("videos/"):
            entries[item.path] = {"bytes": item.lfs.size, "sha256": item.lfs.sha256}
    if not entries:
        raise RuntimeError("QIVD upstream tree returned no LFS videos")
    return entries


def main() -> None:
    parser = argparse.ArgumentParser(description="Resumably fetch the pinned QIVD video corpus.")
    parser.add_argument("--output", default="data/raw/qivd")
    parser.add_argument("--retries", type=int, default=12)
    parser.add_argument("--base-delay", type=float, default=5.0)
    args = parser.parse_args()
    root = ROOT / args.output
    root.mkdir(parents=True, exist_ok=True)
    for relative_path, upstream in ROOT_FILES.items():
        download_file(
            relative_path,
            root / relative_path,
            args.retries,
            args.base_delay,
            expected_size=int(upstream["bytes"]),
            expected_sha256=str(upstream["sha256"]),
        )

    import pyarrow.parquet as pq

    rows = pq.read_table(root / "metadata.parquet", columns=["video_file_name"]).to_pylist()
    video_files = sorted({row["video_file_name"] for row in rows})
    expected = upstream_video_files()
    if set(video_files) != set(expected):
        raise RuntimeError("QIVD metadata video paths do not match the pinned upstream tree")
    for index, relative_path in enumerate(video_files, start=1):
        upstream = expected[relative_path]
        download_file(
            relative_path,
            root / relative_path,
            args.retries,
            args.base_delay,
            expected_size=int(upstream["bytes"]),
            expected_sha256=str(upstream["sha256"]),
        )
        if index == 1 or index % 50 == 0:
            completed = sum((root / path).is_file() for path in video_files)
            print(json.dumps({"completed": completed, "total": len(video_files)}), flush=True)

    manifest = verified_dataset_manifest(
        root,
        video_files,
        expected,
        REPO,
        REVISION,
        root_files_expected=ROOT_FILES,
    )
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
