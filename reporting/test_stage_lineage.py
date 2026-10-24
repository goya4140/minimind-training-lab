"""Synthetic temporary fixtures only; never substitute for experiment evidence."""
import json
import tempfile
import unittest
from pathlib import Path

from audit_stage_lineage import audit_stage, digest, events, required


class StageLineageTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.run = self.root / "artifacts/runs/synthetic-child"
        self.run.mkdir(parents=True)
        self.stage = "llm-pretrain"
        self.entries = []
        for index, source in enumerate(sorted(required(self.stage))):
            p = self.run / "stages" / self.stage / f"fixture-{index}"
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text("synthetic fixture, not actual training\n")
            self.entries.append({"source": source, "archive": str(p.relative_to(self.root)), "sha256": digest(p)})
        self.rows = [{"kind": "stage_completed", "stage": self.stage},
                     {"kind": "artifacts_archived", "stage": self.stage, "artifacts": self.entries}]

    def test_complete_archive(self):
        result = audit_stage(self.run, self.rows, self.stage)
        self.assertEqual(result["archive_files_verified"], 5)
        self.assertFalse(result["inherited"])

    def test_incomplete_not_promoted_by_archive(self):
        result = audit_stage(self.run, self.rows[1:], self.stage)
        self.assertEqual(result["status"], "incomplete")

    def test_missing_or_duplicate_sources_rejected(self):
        for entries in (self.entries[:-1], self.entries + [self.entries[0]]):
            rows = [self.rows[0], dict(self.rows[1], artifacts=entries)]
            with self.assertRaises(ValueError):
                audit_stage(self.run, rows, self.stage)

    def test_corrupt_archive_rejected(self):
        (self.root / self.entries[0]["archive"]).write_text("changed")
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            audit_stage(self.run, self.rows, self.stage)

    def test_path_escape_rejected(self):
        self.entries[0]["archive"] = "outside-the-stage"
        with self.assertRaisesRegex(ValueError, "escapes"):
            audit_stage(self.run, self.rows, self.stage)

    def test_inheritance_without_parent_rejected(self):
        rows = self.rows + [{"kind": "stage_inherited", "stage": self.stage, "parent_run_id": "missing"}]
        with self.assertRaisesRegex(ValueError, "parent lineage"):
            audit_stage(self.run, rows, self.stage)

    def test_event_run_mismatch_rejected(self):
        (self.run / "events.jsonl").write_text(json.dumps({"run_id": "another"}) + "\n")
        with self.assertRaisesRegex(ValueError, "Mixed run"):
            events(self.run)

    def inherited_fixture(self):
        parent_root = self.root / "parent-workspace"
        parent = parent_root / "artifacts/runs/synthetic-parent"
        parent.mkdir(parents=True)
        (parent / "manifest.json").write_text(json.dumps({"run_id": parent.name, "code_commit": "fixture-commit"}))
        entries = []
        for index, entry in enumerate(self.entries):
            p = parent / "stages" / self.stage / f"fixture-{index}"
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes((self.root / entry["archive"]).read_bytes())
            entries.append(dict(entry, archive=str(p.relative_to(parent_root))))
        rows = [{"run_id": parent.name, "kind": "stage_completed", "stage": self.stage},
                {"run_id": parent.name, "kind": "artifacts_archived", "stage": self.stage, "artifacts": entries}]
        (parent / "events.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
        child_rows = self.rows + [
            {"kind": "stage_inherited", "stage": self.stage, "parent_run_id": parent.name},
            {"kind": "repair_lineage_registered", "parent_run_id": parent.name,
             "parent_root": str(parent_root), "parent_code_commit": "fixture-commit"}]
        return child_rows, parent_root, parent, entries

    def test_inherited_origin_preserved(self):
        rows, _, parent, _ = self.inherited_fixture()
        result = audit_stage(self.run, rows, self.stage)
        self.assertTrue(result["inherited"])
        self.assertEqual(result["original_training_run_id"], parent.name)

    def test_parent_content_change_rejected(self):
        rows, parent_root, _, entries = self.inherited_fixture()
        (parent_root / entries[0]["archive"]).write_text("corrupted parent")
        with self.assertRaisesRegex(ValueError, "differs from parent"):
            audit_stage(self.run, rows, self.stage)

    def test_parent_commit_mismatch_rejected(self):
        rows, _, _, _ = self.inherited_fixture()
        rows[-1]["parent_code_commit"] = "wrong"
        with self.assertRaisesRegex(ValueError, "manifest binding"):
            audit_stage(self.run, rows, self.stage)


if __name__ == "__main__":
    unittest.main()
