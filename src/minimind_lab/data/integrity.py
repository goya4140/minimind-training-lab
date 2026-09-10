from __future__ import annotations

import hashlib
from pathlib import Path

QIVD_REPOSITORY = "Qualcomm-AI-Research/QIVD"
QIVD_REVISION = "c5376ab0b9fd3643545a1503413aee64f26ba22a"
QIVD_ROOT_FILES = {
    "LICENSE": {
        "bytes": 279,
        "sha256": "b059c35e7bc6ca46ee32784ad8a43c3d0b424e3a8a5f250974ef462f299747e1",
    },
    "license.pdf": {
        "bytes": 166_211,
        "sha256": "3759feaceba7b5c73336388b342622d8cb5c46283d5e8f41d0662a50be200128",
    },
    "metadata.parquet": {
        "bytes": 147_549,
        "sha256": "32ba00adfd99cbcf70fa882009882c0261bfd5d773a7dde4f1888ab255cfde3f",
    },
}


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
    root: Path,
    files: list[str],
    expected: dict[str, dict[str, int | str]],
    repository: str,
    revision: str,
    root_files_expected: dict[str, dict[str, int | str]] | None = None,
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
    root_files = []
    for relative_path, upstream in sorted((root_files_expected or {}).items()):
        path = root / relative_path
        digest = sha256_file(path)
        if path.stat().st_size != upstream["bytes"] or digest != upstream["sha256"]:
            raise RuntimeError(f"dataset root file differs from pinned upstream object: {relative_path}")
        root_files.append({"path": relative_path, "bytes": path.stat().st_size, "sha256": digest})
    return {
        "repository": repository,
        "revision": revision,
        "video_count": len(entries),
        "total_video_bytes": sum(entry["bytes"] for entry in entries),
        "aggregate_sha256": aggregate,
        "upstream_lfs_verified": True,
        "root_files": root_files,
        "files": entries,
    }
