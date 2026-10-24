"""Synthetic state fixtures, not actual training results."""
import unittest
from audit_final_weight_relations import compare_states, video_group


class TensorFixture:
    shape = (1,)
    dtype = "float32"
    def __init__(self, value):
        self.value = value


class WeightRelationTests(unittest.TestCase):
    def test_component_classification(self):
        names = {"temporal_adapter.queries": "temporal_adapter", "video_projector.0.weight": "video_projector",
                 "language_model.layers.0.norm.weight": "language_boundary",
                 "language_model.layers.7.norm.weight": "language_boundary",
                 "language_model.layers.70.norm.weight": "frozen_language",
                 "language_model.embed_tokens.weight": "frozen_language"}
        for name, group in names.items():
            self.assertEqual(video_group(name), group)
        with self.assertRaisesRegex(ValueError, "Unexpected"):
            video_group("vision_encoder.weight")

    def test_equal_and_changed_values(self):
        equal = lambda a, b: a.value == b.value
        self.assertEqual(compare_states({"x": TensorFixture(1)}, {"x": TensorFixture(1)}, equal), [])
        self.assertEqual(compare_states({"x": TensorFixture(1)}, {"x": TensorFixture(2)}, equal), ["x"])

    def test_key_shape_and_dtype_mismatch(self):
        with self.assertRaisesRegex(ValueError, "key sets"):
            compare_states({"x": TensorFixture(1)}, {"y": TensorFixture(1)}, lambda a, b: True)
        for attr, value in (("shape", (2,)), ("dtype", "float16")):
            a, b = TensorFixture(1), TensorFixture(1)
            setattr(b, attr, value)
            with self.assertRaisesRegex(ValueError, "shape/dtype"):
                compare_states({"x": a}, {"x": b}, lambda a, b: True)


if __name__ == "__main__":
    unittest.main()
