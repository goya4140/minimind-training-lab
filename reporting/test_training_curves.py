"""Synthetic fixtures test transformation rules, never supply experiment results."""
import copy
import unittest

from training_curves import aggregate


class CurveAggregationTests(unittest.TestCase):
    def setUp(self):
        base = {
            "run_id": "synthetic-unit-only", "loss": 3.0, "learning_rate": 0.01,
            "grad_norm": 1.0, "padded_tokens_per_second": 50.0,
            "cuda_allocated_bytes": 10, "cuda_reserved_bytes": 20,
            "cuda_peak_allocated_bytes": 15, "training_seconds": 1.0,
        }
        self.rows = [dict(base, step=1, grad_norm=None), dict(base, step=20, loss=1),
                     dict(base, step=80, loss=2)]

    def aggregate(self, rows=None, width=20):
        return aggregate(self.rows if rows is None else rows, run_id="synthetic-unit-only",
                         checkpoint_step=80, bin_width=width)

    def test_mean_median_count_and_gaps(self):
        rows = self.aggregate()
        self.assertEqual(rows[0]["loss_mean"], 2)
        self.assertEqual(rows[0]["grad_norm_median"], 1)
        self.assertEqual(rows[0]["logged_microbatches"], 2)
        self.assertEqual([r["bin_start"] for r in rows], [1, 61])

    def test_null_gradient_is_not_a_fabricated_zero(self):
        result = self.aggregate([self.rows[0]])
        self.assertIsNone(result[0]["grad_norm_median"])

    def test_duplicate_or_out_of_order_steps_rejected(self):
        for rows in (self.rows + self.rows, self.rows[::-1]):
            with self.subTest(rows=rows), self.assertRaises(ValueError):
                self.aggregate(rows)

    def test_other_run_rejected(self):
        rows = copy.deepcopy(self.rows)
        rows[0]["run_id"] = "another-run"
        with self.assertRaises(ValueError):
            self.aggregate(rows)

    def test_future_step_rejected(self):
        rows = copy.deepcopy(self.rows)
        rows[-1]["step"] = 81
        with self.assertRaises(ValueError):
            self.aggregate(rows)

    def test_nonfinite_missing_or_negative_metrics_rejected(self):
        for value in (float("nan"), float("inf"), -1, None):
            rows = copy.deepcopy(self.rows)
            rows[0]["loss"] = value
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.aggregate(rows)

    def test_impossible_peak_rejected(self):
        rows = copy.deepcopy(self.rows)
        rows[0]["cuda_peak_allocated_bytes"] = 1
        with self.assertRaises(ValueError):
            self.aggregate(rows)

    def test_empty_history_and_invalid_width_rejected(self):
        for rows, width in (([], 20), (self.rows, 0), (self.rows, -1)):
            with self.subTest(width=width), self.assertRaises(ValueError):
                self.aggregate(rows, width)


if __name__ == "__main__":
    unittest.main()
