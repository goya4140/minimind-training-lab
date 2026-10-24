#!/usr/bin/env python3
"""Reverify every pinned QIVD byte offline before CUDA training."""
from __future__ import annotations

import json
from pathlib import Path

import pyarrow.parquet as pq

from minimind_lab.data.integrity import (
    QIVD_REPOSITORY,
    QIVD_REVISION,
    QIVD_ROOT_FILES,
    file_matches,
    verified_dataset_manifest,
)
from minimind_lab.reporting import validate_qivd_manifest

ROOT = Path(__file__).resolve().parents[1]
EXPECTED_AGGREGATE = "648b60445d59f1e445d0437ddd93f8ad97c09bb7e6ded0d46a996c98428ad7a3"


def main():
    root = ROOT / "data/raw/qivd"
    manifest = json.loads((ROOT / "data/manifests/qivd.json").read_text())
    validate_qivd_manifest(manifest, QIVD_REVISION, expected_root_files=QIVD_ROOT_FILES)
    if manifest.get("repository") != QIVD_REPOSITORY:
        raise ValueError("QIVD repository differs from pinned source")
    for name, expected in QIVD_ROOT_FILES.items():
        if not file_matches(root / name, int(expected["bytes"]), str(expected["sha256"])):
            raise ValueError(f"QIVD root file differs from pinned bytes: {name}")
    paths = sorted({row["video_file_name"] for row in pq.read_table(root / "metadata.parquet").to_pylist()})
    expected = {entry["path"]: entry for entry in manifest["files"]}
    if len(expected) != 2900 or set(paths) != set(expected):
        raise ValueError("QIVD metadata paths differ from manifest")
    verified = verified_dataset_manifest(root, paths, expected, QIVD_REPOSITORY, QIVD_REVISION, QIVD_ROOT_FILES)
    if verified["aggregate_sha256"] != EXPECTED_AGGREGATE:
        raise ValueError("QIVD aggregate differs from pinned original experiment")
    print(json.dumps({key: value for key, value in verified.items() if key != "files"}, indent=2))


if __name__ == "__main__":
    main()
