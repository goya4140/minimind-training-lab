"""Synthetic audit counter/RNG structures, not model execution evidence."""
import unittest
from audit_final_training import validate_history, validate_rng


class ByteTensorFixture:
    dtype = "uint8"
    ndim = 1
    def numel(self):
        return 16


class FinalTrainingAuditTests(unittest.TestCase):
    def test_unlogged_final_tail_is_explicit(self):
        r = validate_history([{"run_id": "original", "step": 8, "optimizer_updates": 2}], "original", 10, 4)
        self.assertEqual(r["unlogged_tail_microsteps"], 2)
        self.assertEqual(r["configured_final_optimizer_updates"], 3)

    def test_partial_final_update(self):
        validate_history([{"run_id": "original", "step": 10, "optimizer_updates": 3}], "original", 10, 4)
        with self.assertRaisesRegex(ValueError, "counter"):
            validate_history([{"run_id": "original", "step": 10, "optimizer_updates": 2}], "original", 10, 4)

    def test_retagging_inherited_history_rejected(self):
        with self.assertRaisesRegex(ValueError, "identity"):
            validate_history([{"run_id": "new-child", "step": 4, "optimizer_updates": 1}], "original", 8, 4)

    def test_duplicate_history_rejected(self):
        row = {"run_id": "original", "step": 4, "optimizer_updates": 1}
        with self.assertRaisesRegex(ValueError, "duplicated"):
            validate_history([row, row], "original", 8, 4)

    def test_rng_byte_structure(self):
        t = ByteTensorFixture()
        self.assertEqual(validate_rng(t, [t], ByteTensorFixture, "uint8")["cuda_rng_devices_saved"], 1)
        for cuda in (None, [], [t, t], [object()]):
            with self.assertRaisesRegex(ValueError, "CUDA RNG"):
                validate_rng(t, cuda, ByteTensorFixture, "uint8")
        with self.assertRaisesRegex(ValueError, "CPU RNG"):
            validate_rng(object(), [t], ByteTensorFixture, "uint8")


if __name__ == "__main__":
    unittest.main()
