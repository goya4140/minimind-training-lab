"""State-key classification tests, not actual model results."""
import unittest
from audit_vlm_freezing import group


class FreezePolicyTests(unittest.TestCase):
    def test_projector(self):
        self.assertEqual(group("vision_projector.1.weight"), "projector")

    def test_boundary_layers(self):
        for index in (0, 7):
            self.assertEqual(group(f"language_model.layers.{index}.attention.q_proj.weight"), "language_boundary")

    def test_middle_layers_and_embeddings_frozen(self):
        for name in ("language_model.layers.1.attention.q_proj.weight", "language_model.layers.6.ffn.w1.weight",
                     "language_model.embed_tokens.weight", "language_model.lm_head.weight", "language_model.norm.weight"):
            self.assertEqual(group(name), "frozen_language")

    def test_prefix_boundaries_not_substring(self):
        for name in ("language_model.layers.70.weight", "prefix.vision_projector.weight", "language_model.layers.00.weight"):
            self.assertEqual(group(name), "frozen_language")


if __name__ == "__main__":
    unittest.main()
