import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
spec = importlib.util.spec_from_file_location("preserved_pipeline", ROOT / "scripts/run_mps_pipeline.py")
pipeline = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pipeline)


def fixture(tmp_path, monkeypatch):
    root = tmp_path
    stage = "vlm-alignment"
    directory = root / "artifacts/runs/unit"
    directory.mkdir(parents=True)
    config = root / "configs/test.yaml"
    config.parent.mkdir()
    config.write_text('experiment:\n  name: unit\ntraining:\n  checkpoint_path: artifacts/checkpoints/unit.pt\n')
    files = ["artifacts/checkpoints/unit.pt", "artifacts/checkpoints/unit.resume.pt",
             "artifacts/logs/unit.json", "artifacts/logs/unit.console.log"]
    entries = []
    for relative in files:
        source = root / relative
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_bytes(b"original evidence")
        archive = directory / "stages" / stage / source.name
        archive.parent.mkdir(parents=True, exist_ok=True)
        archive.write_bytes(source.read_bytes())
        entries.append({"source": relative, "archive": str(archive.relative_to(root)),
                        "sha256": hashlib.sha256(source.read_bytes()).hexdigest()})
    events = [{"kind": "stage_inherited", "stage": stage},
              {"kind": "artifacts_archived", "stage": stage, "artifacts": entries}]
    (directory / "events.jsonl").write_text("\n".join(json.dumps(e) for e in events) + "\n")
    recorded = []
    evidence = SimpleNamespace(directory=directory, event=lambda kind, **kw: recorded.append((kind, kw)))
    monkeypatch.setattr(pipeline, "ROOT", root)
    monkeypatch.setattr(pipeline, "active_pid", lambda path: None)
    monkeypatch.setattr(pipeline, "verify_checkpoint", lambda path: {"all_finite": True})
    monkeypatch.setattr(pipeline, "verify_resume_checkpoint", lambda *a, **kw: {"training_complete": True})
    return evidence, stage, recorded, entries


def test_preserved_stage_reuses_without_rewriting_reports(tmp_path, monkeypatch):
    evidence, stage, recorded, _ = fixture(tmp_path, monkeypatch)
    before = {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in tmp_path.rglob("*") if p.is_file()}
    assert pipeline.verify_preserved_stage(stage, "configs/test.yaml", evidence)
    assert before == {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in before}
    assert recorded[-1] == ("stage_completed", {"stage": stage, "preserved": True, "inherited": True})


def test_tampered_preserved_source_is_rejected(tmp_path, monkeypatch):
    evidence, stage, recorded, entries = fixture(tmp_path, monkeypatch)
    (tmp_path / entries[0]["source"]).write_text("changed")
    with pytest.raises(RuntimeError, match="Preserved artifact changed"):
        pipeline.verify_preserved_stage(stage, "configs/test.yaml", evidence)
    assert recorded == []


def test_missing_archived_evidence_is_rejected(tmp_path, monkeypatch):
    evidence, stage, recorded, entries = fixture(tmp_path, monkeypatch)
    (tmp_path / entries[0]["archive"]).unlink()
    with pytest.raises(RuntimeError, match="Preserved artifact changed"):
        pipeline.verify_preserved_stage(stage, "configs/test.yaml", evidence)
    assert recorded == []
