"""Synthetic arithmetic/denominator tests, not experimental results."""
import math
import unittest
from model_results_comparison import LANGUAGE_STAGES, validate_language


class ComparisonTests(unittest.TestCase):
    def rows(self):
        return [{"stage": stage, "nll": 1., "ppl": math.e, "bpb": 670079/(math.log(2)*1906560),
                 "predicted_tokens": 670079, "utf8_bytes": 1906560} for stage in LANGUAGE_STAGES]

    def test_consistent_metrics(self):
        validate_language(self.rows())

    def test_different_denominator(self):
        rows = self.rows()
        rows[1]["predicted_tokens"] += 1
        with self.assertRaisesRegex(ValueError, "denominators"):
            validate_language(rows)

    def test_arithmetic_or_nonfinite_rejected(self):
        for key, value in (("ppl", 3), ("bpb", 2), ("nll", float("nan"))):
            rows = self.rows()
            rows[0][key] = value
            with self.assertRaises(ValueError):
                validate_language(rows)

    def test_missing_or_wrong_order(self):
        for rows in (self.rows()[:-1], self.rows()[::-1]):
            with self.assertRaisesRegex(ValueError, "four designated"):
                validate_language(rows)


if __name__ == "__main__":
    unittest.main()
