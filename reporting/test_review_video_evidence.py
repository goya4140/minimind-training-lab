"""Synthetic reader fixtures only; not model performance evidence."""
import contextlib
import hashlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import review_video_evidence as reader


class ReviewVideoTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.run = self.root / "artifacts/runs/synthetic-video"
        self.path = self.run / "stages/video-omni-sft/video-omni-final.json"
        self.path.parent.mkdir(parents=True)
        row = {"id": 1, "question": "Synthetic question?", "short_answer": "reference",
               "normal_completion": "fixture output", "reversed_completion": "fixture output"}
        metric = {"normalized_exact_match": 0, "reversed_frame_exact_match": 0,
                  "token_f1": 0, "reversed_frame_token_f1": 0,
                  "completion_change_rate_on_reversal": 0, "qualitative": [row]}
        self.result = {"test_loss": 1, "reversed_frame_test_loss": 1,
                       "qivd_generation": {**metric, "generated_samples": 100},
                       "controlled_temporal": {**metric, "generated_samples": 160}}
        self.rows = [{"kind": "stage_completed", "stage": "video-omni-sft"},
                     {"kind": "artifacts_archived", "stage": "video-omni-sft", "artifacts": []}]
        self.save()

    def save(self):
        self.path.write_text(json.dumps(self.result))
        self.rows[-1]["artifacts"] = [{"archive": str(self.path.relative_to(self.root)),
                                       "sha256": hashlib.sha256(self.path.read_bytes()).hexdigest()}]

    def invoke(self, run_id="synthetic-video"):
        (self.run / "events.jsonl").write_text("".join(json.dumps(x) + "\n" for x in self.rows))
        output = io.StringIO()
        with patch.object(reader, "ROOT", self.root), patch.object(sys, "argv", ["reader", "--run-id", run_id]), contextlib.redirect_stdout(output):
            reader.main()
        return output.getvalue()

    def test_verified_saved_outputs(self):
        output = self.invoke()
        self.assertIn("fixture output", output)
        self.assertIn("ORIGINAL references", output)
        self.assertIn("not new inference", output)

    def test_corruption(self):
        self.path.write_text("changed")
        with self.assertRaisesRegex(ValueError, "checksum"):
            self.invoke()

    def test_incomplete(self):
        self.rows = self.rows[1:]
        with self.assertRaisesRegex(ValueError, "not completed"):
            self.invoke()

    def test_path_escape(self):
        self.rows[-1]["artifacts"][0]["archive"] = "outside/video-omni-final.json"
        with self.assertRaisesRegex(ValueError, "escapes"):
            self.invoke()

    def test_invalid_run(self):
        with self.assertRaisesRegex(ValueError, "Invalid run ID"):
            self.invoke("../escape")

    def test_wrong_sample_counts(self):
        self.result["qivd_generation"]["generated_samples"] = 99
        self.save()
        with self.assertRaisesRegex(ValueError, "generation counts"):
            self.invoke()


if __name__ == "__main__":
    unittest.main()
