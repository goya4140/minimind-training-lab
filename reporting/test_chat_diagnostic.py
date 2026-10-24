"""Scorer tests use reference fixtures, never claim measured model capability."""
import copy
import json
import tempfile
import unittest
from pathlib import Path

from chat_diagnostic import load_protocol, score_case


class ChatDiagnosticTests(unittest.TestCase):
    def setUp(self):
        self.protocol = load_protocol(Path(__file__).parent / "protocols/chat-diagnostic-v1.json")
        self.json_case = next(c for c in self.protocol["cases"] if c["scorer"] == "json_object")

    def test_all_references_pass(self):
        for case in self.protocol["cases"]:
            answers = case.get("references", [json.dumps(case.get("reference_object"), ensure_ascii=False)])
            for answer in answers:
                with self.subTest(case=case["id"], answer=answer):
                    self.assertEqual(score_case(case, answer)["score"], 1)

    def test_incorrect_and_empty_outputs_fail(self):
        for case in self.protocol["cases"]:
            for answer in ("", "完全不相关的输出"):
                with self.subTest(case=case["id"], answer=answer):
                    self.assertEqual(score_case(case, answer)["score"], 0)

    def test_strip_only_outer_whitespace(self):
        case = self.protocol["cases"][2]
        self.assertEqual(score_case(case, "\n蓝色河岸 Q7  ")["score"], 1)
        self.assertEqual(score_case(case, "蓝色河岸Q7")["score"], 0)
        self.assertEqual(score_case(case, "蓝色河岸 Q7。")["score"], 0)

    def test_extra_explanation_fails_strict_short_answer(self):
        case = self.protocol["cases"][0]
        self.assertEqual(score_case(case, "中国的首都是北京。")["score"], 0)

    def test_json_key_order_and_whitespace_are_immaterial(self):
        self.assertEqual(score_case(self.json_case, '{ "country": "中国", "city": "北京" }')["score"], 1)

    def test_json_invalid_fenced_extra_missing_wrong_and_duplicate(self):
        predictions = [
            '```json\n{"city":"北京","country":"中国"}\n```',
            '{"city":"北京"}',
            '{"city":"北京","country":"中国","extra":1}',
            '{"city":"上海","country":"中国"}',
            '{"city":"北京","city":"北京","country":"中国"}',
            '{"city":NaN,"country":"中国"}',
            '["北京","中国"]',
        ]
        for answer in predictions:
            with self.subTest(answer=answer):
                self.assertEqual(score_case(self.json_case, answer)["score"], 0)

    def test_changed_generation_or_case_identity_rejected(self):
        for invalid in ("generation", "cases"):
            protocol = copy.deepcopy(self.protocol)
            if invalid == "generation":
                protocol["generation"]["max_new_tokens"] = 10
            else:
                protocol["cases"][0]["id"] = protocol["cases"][1]["id"]
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "bad.json"
                path.write_text(json.dumps(protocol))
                with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                    load_protocol(path)

    def test_unknown_scorer_rejected(self):
        with self.assertRaises(ValueError):
            score_case({"scorer": "unknown"}, "answer")


if __name__ == "__main__":
    unittest.main()
