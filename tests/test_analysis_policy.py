import unittest

from chiplet_tuner.agents.analysis_policy import resolve_analysis_policy


class AnalysisPolicyTests(unittest.TestCase):
    def test_standard_is_unchanged(self):
        self.assertEqual(resolve_analysis_policy("model_level", 15, "standard"), (15, ""))

    def test_focused_preserves_smaller_existing_limit(self):
        self.assertEqual(resolve_analysis_policy("layer_level", 1, "focused")[0], 1)
        self.assertEqual(resolve_analysis_policy("solution_generation", 15, "focused")[0], 2)
        self.assertEqual(resolve_analysis_policy("model_level", 15, "focused")[0], 5)

    def test_unknown_mode_rejected(self):
        with self.assertRaises(ValueError):
            resolve_analysis_policy("model_level", 15, "typo")

    def test_focused_includes_model_semantics_for_each_role(self):
        for role in ("model_level", "layer_level", "solution_generation"):
            _, guidance = resolve_analysis_policy(role, 15, "focused")
            self.assertIn("Neither expansion is cost-neutral", guidance)
            self.assertIn("access volume", guidance)

    def test_unknown_role_unchanged(self):
        self.assertEqual(resolve_analysis_policy("other", 15, "focused"), (15, ""))


if __name__ == "__main__":
    unittest.main()
