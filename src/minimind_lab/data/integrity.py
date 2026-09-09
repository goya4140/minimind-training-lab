from __future__ import annotations

import hashlib
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def file_matches(path: Path, expected_size: int | None = None, expected_sha256: str | None = None) -> bool:
    if not path.is_file() or path.stat().st_size == 0:
        return False
    if expected_size is not None and path.stat().st_size != expected_size:
        return False
    return expected_sha256 is None or sha256_file(path) == expected_sha256


def verified_dataset_manifest(
    root: Path, files: list[str], expected: dict[str, dict[str, int | str]], repository: str, revision: str
) -> dict:
    entries = []
    for relative_path in sorted(files):
        path = root / relative_path
        digest = sha256_file(path)
        upstream = expected[relative_path]
        if path.stat().st_size != upstream["bytes"] or digest != upstream["sha256"]:
            raise RuntimeError(f"dataset file differs from pinned upstream object: {relative_path}")
        entries.append({"path": relative_path, "bytes": path.stat().st_size, "sha256": digest})
    aggregate = hashlib.sha256(
        "\n".join(f"{entry['path']} {entry['bytes']} {entry['sha256']}" for entry in entries).encode()
    ).hexdigest()
    return {
        "repository": repository,
        "revision": revision,
        "video_count": len(entries),
        "total_video_bytes": sum(entry["bytes"] for entry in entries),
        "aggregate_sha256": aggregate,
        "upstream_lfs_verified": True,
        "files": entries,
    }
