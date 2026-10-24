import json
import os
from pathlib import Path

import pytest
import torch

from minimind_lab.training import ActiveTrainingTimer, file_sha256
from minimind_lab.training.evidence import RunEvidence, enrich_training_record, validate_run_id


def test_run_id_cannot_escape_evidence_directory():
    for value in ("../outside", "/root", "two words", "", "a" * 81):
        with pytest.raises(ValueError):
            validate_run_id(value)


def test_pinned_source_guard_detects_changes(tmp_path: Path):
    source = tmp_path / "train.py"
    source.write_text("original")
    evidence = RunEvidence(tmp_path, "test-001")
    evidence.directory.mkdir(parents=True)
    evidence.manifest_path.write_text(json.dumps({"source_hashes": {"train.py": file_sha256(source)}}))
    evidence.assert_sources_unchanged()
    source.write_text("modified")
    with pytest.raises(RuntimeError, match="source changed"):
        evidence.assert_sources_unchanged()


def test_training_metrics_record_resume_and_padded_token_semantics(tmp_path, monkeypatch):
    monkeypatch.setenv("MINIMIND_RUN_ID", "test-001")
    monkeypatch.setenv("MINIMIND_RUN_DIR", str(tmp_path))
    record = {"step": 5, "loss": 2.0, "seconds_per_step": 0.5}
    config = {
        "experiment": {"name": "test"},
        "training": {"batch_size": 2, "gradient_accumulation_steps": 4},
        "data": {"sequence_length": 10},
    }
    timer = ActiveTrainingTimer(prior_training_seconds=4)
    enrich_training_record(record, device=torch.device("cpu"), config=config, timer=timer, start_step=4, total_steps=5)
    saved = json.loads((tmp_path / "metrics/test.jsonl").read_text())
    assert saved["run_id"] == "test-001"
    assert saved["optimizer_updates"] == 2
    assert saved["resumed_from_step"] == 4
    assert saved["padded_tokens_per_second"] == 40.0
    assert saved["training_seconds"] == 4


def test_archived_evidence_rejects_overwrite(tmp_path):
    source = tmp_path / "summary.json"
    source.write_text("first")
    evidence = RunEvidence(tmp_path, "test-001")
    evidence.archive("llm", [source])
    source.write_text("changed")
    with pytest.raises(RuntimeError, match="overwrite"):
        evidence.archive("llm", [source])


def test_pinned_data_guard_rejects_same_size_tampering(tmp_path):
    data = tmp_path / "data.bin"
    data.write_bytes(b"abc")
    original_time = data.stat().st_mtime_ns
    evidence = RunEvidence(tmp_path, "test-001")
    evidence.directory.mkdir(parents=True)
    evidence.manifest_path.write_text(
        json.dumps(
            {
                "source_hashes": {},
                "data_files": [
                    {"path": "data.bin", "bytes": 3, "mtime_ns": original_time, "sha256": file_sha256(data)}
                ],
            }
        )
    )
    data.write_bytes(b"xyz")
    os.utime(data, ns=(original_time + 1, original_time + 1))
    with pytest.raises(RuntimeError, match="data checksum changed"):
        evidence.assert_sources_unchanged()


def test_pinned_dependency_guard_rejects_version_drift(tmp_path, monkeypatch):
    evidence = RunEvidence(tmp_path, "test-001")
    evidence.directory.mkdir(parents=True)
    evidence.manifest_path.write_text(json.dumps({"source_hashes": {}, "runtime": {"versions": {"torch": "old"}}}))
    monkeypatch.setattr("minimind_lab.training.evidence.importlib.metadata.version", lambda name: "new")
    with pytest.raises(RuntimeError, match="dependency changed"):
        evidence.assert_sources_unchanged()
