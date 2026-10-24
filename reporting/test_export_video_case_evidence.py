"""Synthetic selection/containment tests; not actual video evaluation."""
import tempfile
import unittest
from pathlib import Path

from export_video_case_evidence import FAMILIES, contained_file, select_cases


class VideoCaseSelectionTests(unittest.TestCase):
    def setUp(self):
        self.rows = [{"id": i, "category": FAMILIES[i % 4], "normal_token_f1": i / 160} for i in range(160)]

    def test_first_in_evaluation_order_not_best_score(self):
        self.assertEqual([r["id"] for r in select_cases(self.rows)], [0, 1, 2, 3])
        self.assertEqual([r["id"] for r in select_cases(self.rows[::-1])], [156, 157, 158, 159])

    def test_missing_or_duplicate_case(self):
        for rows in (self.rows[:-1], self.rows[:-1] + [self.rows[0]]):
            with self.assertRaisesRegex(ValueError, "160 unique"):
                select_cases(rows)

    def test_unbalanced_or_unknown_category(self):
        for category in (FAMILIES[1], "unknown"):
            rows = [dict(r) for r in self.rows]
            rows[0]["category"] = category
            with self.assertRaises(ValueError):
                select_cases(rows)

    def test_missing_and_escaping_paths_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "root"
            root.mkdir()
            outside = root.parent / "outside"
            outside.write_text("synthetic")
            (root / "link").symlink_to(outside)
            (root / "inside").write_text("synthetic")
            self.assertEqual(contained_file(root, "inside"), (root / "inside").resolve())
            for name in ("missing", "../outside", "link"):
                with self.assertRaisesRegex(ValueError, "escaping"):
                    contained_file(root, name)


if __name__ == "__main__":
    unittest.main()
