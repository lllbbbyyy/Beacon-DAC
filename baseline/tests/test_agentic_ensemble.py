import unittest

import numpy as np

from baseline.agentic_ensemble import ScalarEnsemble


class ScalarEnsembleTests(unittest.TestCase):
    def setUp(self):
        self.x = np.random.default_rng(7).random((12, 3))
        self.y = 2 + self.x[:, 0] ** 2 + self.x[:, 1]

    def test_selection_weights_and_acquisition(self):
        model = ScalarEnsemble(seed=3, folds=3, trees=5).fit(self.x, self.y)
        report = model.report
        self.assertEqual(len(report["selected"]), 3)
        self.assertEqual(report["selected"], [row["name"] for row in report["leaderboard"][:3]])
        self.assertAlmostEqual(sum(report["weights"]), 1)
        self.assertTrue(all(w > 0 for w in report["weights"]))
        ei = model.expected_improvement(self.x, min(self.y))
        self.assertTrue(np.isfinite(ei).all())
        self.assertTrue((ei >= 0).all())

    def test_physical_units_do_not_change_selection(self):
        first = ScalarEnsemble(seed=3, folds=3, trees=5).fit(self.x, self.y)
        # A binary scale avoids introducing rounding-dependent tree split ties.
        unit_scale = float(2 ** 70)
        scaled = ScalarEnsemble(seed=3, folds=3, trees=5).fit(self.x, self.y * unit_scale)
        self.assertEqual(first.report["selected"], scaled.report["selected"])
        np.testing.assert_allclose(first.weights, scaled.weights)
        np.testing.assert_allclose(first.expected_improvement(self.x, min(self.y)),
                                   scaled.expected_improvement(self.x, min(self.y) * unit_scale))

    def test_zero_uncertainty_has_finite_improvement(self):
        model = ScalarEnsemble(folds=3, trees=5).fit(self.x, np.ones(12))
        np.testing.assert_allclose(model.expected_improvement(self.x, 1), 0)
        np.testing.assert_allclose(model.expected_improvement(self.x, 2), 1)

    def test_invalid_training_data(self):
        for y in (np.zeros(12), np.full(12, np.nan), self.y[:-1]):
            with self.assertRaises(ValueError):
                ScalarEnsemble(folds=3).fit(self.x, y)
        with self.assertRaises(ValueError):
            ScalarEnsemble(folds=5).fit(self.x[:3], self.y[:3])


if __name__ == "__main__":
    unittest.main()
