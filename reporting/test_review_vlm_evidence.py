"""Synthetic result-reader fixtures; not actual evaluation outputs."""
import contextlib
import hashlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import review_vlm_evidence as reader


class ReviewVLMTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.run = self.root / "artifacts/runs/synthetic-run"
        self.path = self.run / "stages/vlm-sft/vlm-final.json"
        self.path.parent.mkdir(parents=True)
        result = {"validation_loss": 1.0, "visual_ablation": {"correct_image_keyword_recall": 0,
                  "counterfactual_keyword_recall": 0}, "language_regression": {"corpus": {"validation_perplexity": 2}},
                  "qualitative": [{"image_count": 1, "keyword_recall": 0, "counterfactual_keyword_recall": 0,
                  "completion_changed_on_counterfactual": True, "prompt": "synthetic prompt",
                  "completion": "synthetic fixture text"} for _ in range(7)]}
        self.path.write_text(json.dumps(result))
        self.rows = [{"kind": "stage_completed", "stage": "vlm-sft"}, {"kind": "artifacts_archived", "stage": "vlm-sft",
                     "artifacts": [{"archive": str(self.path.relative_to(self.root)),
                                    "sha256": hashlib.sha256(self.path.read_bytes()).hexdigest()}]}]

    def invoke(self, run_id="synthetic-run"):
        (self.run / "events.jsonl").write_text("".join(json.dumps(r) + "\n" for r in self.rows))
        output = io.StringIO()
        with patch.object(reader, "ROOT", self.root), patch.object(sys, "argv", ["reader", "--run-id", run_id]), contextlib.redirect_stdout(output):
            reader.main()
        return output.getvalue()

    def test_verified_reader_shows_all_cases_and_saved_text(self):
        text = self.invoke()
        self.assertIn("07", text)
        self.assertIn("synthetic fixture text", text)
        self.assertIn("not new inference", text)

    def test_corruption_rejected(self):
        self.path.write_text("changed")
        with self.assertRaisesRegex(ValueError, "checksum"):
            self.invoke()

    def test_incomplete_rejected(self):
        self.rows = self.rows[1:]
        with self.assertRaisesRegex(ValueError, "not completed"):
            self.invoke()

    def test_unsafe_run_id_rejected(self):
        with self.assertRaisesRegex(ValueError, "Invalid run ID"):
            self.invoke("../other")

    def test_escaping_archive_rejected(self):
        self.rows[1]["artifacts"][0]["archive"] = "outside/vlm-final.json"
        with self.assertRaisesRegex(ValueError, "escapes"):
            self.invoke()


if __name__ == "__main__":
    unittest.main()
