"""Run-scoped, append-only evidence for reproducible learning reports."""

from __future__ import annotations

import importlib.metadata
import json
import os
import re
import shutil
import subprocess
import uuid
from datetime import UTC, datetime
from pathlib import Path

import torch

from .artifacts import file_sha256

SEGMENT_ID = uuid.uuid4().hex[:12]
VERSIONED_PACKAGES = ("torch", "transformers", "datasets", "numpy", "av", "pyarrow", "tokenizers", "huggingface-hub")


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def validate_run_id(value: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,79}", value):
        raise ValueError("run ID must be 1-80 letters, digits, underscores or hyphens")
    return value


def append_jsonl(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
        handle.flush()


def enrich_training_record(record, *, device, config, timer, start_step, total_steps):
    training = config["training"]
    accumulation = training.get("gradient_accumulation_steps", 1)
    step = record["step"]
    seconds = record["seconds_per_step"]
    record.update(
        recorded_at_utc=utc_now(),
        segment_id=SEGMENT_ID,
        pid=os.getpid(),
        resumed_from_step=start_step,
        optimizer_updates=step // accumulation + int(step == total_steps and step % accumulation != 0),
        training_seconds=timer.training_seconds,
        suspended_seconds=timer.suspended_seconds,
        padded_tokens_per_second=(
            training["batch_size"] * config["data"]["sequence_length"] / seconds if seconds else None
        ),
    )
    if device.type == "cuda":
        record.update(
            cuda_allocated_bytes=torch.cuda.memory_allocated(device),
            cuda_reserved_bytes=torch.cuda.memory_reserved(device),
            cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(device),
        )
    run_id = os.environ.get("MINIMIND_RUN_ID")
    if run_id:
        validate_run_id(run_id)
        root = Path(os.environ["MINIMIND_RUN_DIR"])
        record["run_id"] = run_id
        append_jsonl(root / "metrics" / f"{config['experiment']['name']}.jsonl", record)


class RunEvidence:
    def __init__(self, root: Path, run_id: str):
        self.root = root.resolve()
        self.run_id = validate_run_id(run_id)
        self.directory = self.root / "artifacts/runs" / self.run_id
        self.manifest_path = self.directory / "manifest.json"

    def event(self, kind: str, **fields):
        append_jsonl(
            self.directory / "events.jsonl", {"at_utc": utc_now(), "run_id": self.run_id, "kind": kind, **fields}
        )

    def create(self, configs: list[str], data_files: list[Path]):
        if self.manifest_path.exists():
            raise FileExistsError(f"run already exists; explicitly resume it: {self.run_id}")
        status = subprocess.check_output(["git", "status", "--porcelain"], cwd=self.root, text=True)
        if status.strip():
            raise RuntimeError("commit the experiment implementation before creating a formal run")
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=self.root, text=True).strip()
        sources = sorted(
            {
                *self.root.glob("src/**/*.py"),
                *self.root.glob("scripts/*.py"),
                *self.root.glob("scripts/*.sh"),
                *self.root.glob("configs/**/*.yaml"),
                *self.root.glob("data/eval/vlm/*"),
                self.root / "pyproject.toml",
                self.root / "uv.lock",
                self.root / "requirements-cuda.txt",
            }
        )
        source_hashes = {str(path.relative_to(self.root)): file_sha256(path) for path in sources if path.is_file()}
        self.directory.mkdir(parents=True)
        snapshots = self.directory / "configs"
        snapshots.mkdir()
        for config in configs:
            destination = snapshots / config.removeprefix("configs/")
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(self.root / config, destination)
        manifest = {
            "schema": 1,
            "run_id": self.run_id,
            "created_at_utc": utc_now(),
            "code_commit": commit,
            "source_hashes": source_hashes,
            "config_paths": configs,
            "data_files": [
                {
                    "path": str(path.relative_to(self.root)),
                    "bytes": path.stat().st_size,
                    "mtime_ns": path.stat().st_mtime_ns,
                    "sha256": file_sha256(path),
                }
                for path in data_files
            ],
            "runtime": {
                "torch": torch.__version__,
                "cuda": torch.version.cuda,
                "versions": {name: importlib.metadata.version(name) for name in VERSIONED_PACKAGES},
                "gpu": torch.cuda.get_device_name() if torch.cuda.is_available() else None,
                "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            },
            "retention": "Keep raw logs, evaluations, final weights and latest resume state on server; publish condensed evidence only.",
        }
        self.manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        subprocess.run(
            ["git", "archive", "--format=tar", "-o", str(self.directory / "source.tar"), commit],
            cwd=self.root,
            check=True,
        )
        self.event("run_created", code_commit=commit)
        return manifest

    def assert_sources_unchanged(self):
        manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        for name, expected in manifest.get("runtime", {}).get("versions", {}).items():
            if importlib.metadata.version(name) != expected:
                raise RuntimeError(f"pinned experiment dependency changed: {name}")
        for relative, expected in manifest["source_hashes"].items():
            path = self.root / relative
            if not path.is_file() or file_sha256(path) != expected:
                raise RuntimeError(
                    f"pinned experiment source changed: {relative}; start a new run for substantive changes"
                )
        for entry in manifest.get("data_files", []):
            path = self.root / entry["path"]
            if not path.is_file() or path.stat().st_size != entry["bytes"]:
                raise RuntimeError(f"pinned experiment data changed: {entry['path']}")
            if path.stat().st_mtime_ns != entry["mtime_ns"] and file_sha256(path) != entry["sha256"]:
                raise RuntimeError(f"pinned experiment data checksum changed: {entry['path']}")

    def archive(self, stage: str, paths: list[Path]):
        entries = []
        for path in paths:
            if not path.is_file():
                continue
            destination = self.directory / "stages" / stage / path.name
            destination.parent.mkdir(parents=True, exist_ok=True)
            digest = file_sha256(path)
            if destination.exists() and file_sha256(destination) != digest:
                raise RuntimeError(f"refusing to overwrite archived evidence: {destination}")
            if not destination.exists():
                shutil.copy2(path, destination)
            if file_sha256(destination) != digest:
                raise RuntimeError(f"archived evidence checksum mismatch: {destination}")
            entries.append(
                {
                    "source": str(path.relative_to(self.root)),
                    "archive": str(destination.relative_to(self.root)),
                    "bytes": destination.stat().st_size,
                    "sha256": digest,
                }
            )
        self.event("artifacts_archived", stage=stage, artifacts=entries)
