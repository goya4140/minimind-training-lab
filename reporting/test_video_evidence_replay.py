import importlib.util
import json
from pathlib import Path
import tempfile
import types
import unittest

spec = importlib.util.spec_from_file_location("video_replay", Path(__file__).with_name("video_evidence_replay.py"))
replay = importlib.util.module_from_spec(spec)
spec.loader.exec_module(replay)


class RetentionTests(unittest.TestCase):
    def test_gpu_must_match_formal_run(self):
        manifest = {"runtime": {"cuda_visible_devices": "1"}}
        self.assertEqual(replay.validate_gpu_selection(manifest, "1"), "1")
        for selected in (None, "", "0", "0,1", "1,0", " 1", "1 "):
            with self.subTest(selected=selected), self.assertRaises(ValueError):
                replay.validate_gpu_selection(manifest, selected)

    def test_missing_run_gpu_binding_is_not_permission_to_use_any_gpu(self):
        for manifest in ({}, {"runtime": {}}, {"runtime": {"cuda_visible_devices": ""}},
                         {"runtime": {"cuda_visible_devices": None}}):
            with self.subTest(manifest=manifest), self.assertRaises(ValueError):
                replay.validate_gpu_selection(manifest, "1")

    def test_keeps_every_case_without_changing_summary(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            summaries = []
            def original(rows):
                result = {"generated_samples": len(rows), "qualitative": rows[:12]}
                summaries.append(result)
                return result
            module = types.SimpleNamespace(generation_summary=original)
            receipts = replay.retain_cases(module, output)
            for name, count in [("qivd", 100), ("controlled_temporal", 160)]:
                cases = [{"id": index, "normal_completion": "normal", "reversed_completion": "reverse",
                          "answer": "reference", "normal_token_f1": 0.5} for index in range(count)]
                actual = module.generation_summary(cases)
                self.assertIs(actual, summaries[-1])
                self.assertEqual(len(actual["qualitative"]), 12)
                path = output / f"{name}-predictions.jsonl"
                self.assertEqual([json.loads(line) for line in path.read_text().splitlines()], cases)
            self.assertEqual([r["cases"] for r in receipts], [100, 160])
            self.assertEqual(receipts[1]["sha256"], replay.digest(output / receipts[1]["file"]))
            with self.assertRaises(ValueError):
                module.generation_summary([])

    def test_rejects_shortened_and_duplicate_cases(self):
        for rows in [[{"id": i} for i in range(99)], [{"id": 0}] * 100]:
            with tempfile.TemporaryDirectory() as temporary:
                module = types.SimpleNamespace(generation_summary=lambda cases: {})
                receipts = replay.retain_cases(module, Path(temporary))
                with self.assertRaises(ValueError):
                    module.generation_summary(rows)
                self.assertEqual(receipts, [])
                self.assertEqual(list(Path(temporary).iterdir()), [])

    def test_does_not_overwrite_existing_predictions(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            path = output / "qivd-predictions.jsonl"
            path.write_text("existing evidence\n")
            module = types.SimpleNamespace(generation_summary=lambda cases: {})
            replay.retain_cases(module, output)
            with self.assertRaises(FileExistsError):
                module.generation_summary([{"id": i} for i in range(100)])
            self.assertEqual(path.read_text(), "existing evidence\n")


if __name__ == "__main__":
    unittest.main()
