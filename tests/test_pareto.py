from __future__ import annotations

import unittest

from chiplet_tuner.core.pareto import hypervolume_2d, nondominated_indices


class ParetoTest(unittest.TestCase):
    def test_front_preserves_first_duplicate_and_rejects_dominated_points(self):
        self.assertEqual(nondominated_indices([(1, 3), (2, 2), (3, 1), (2, 3), (1, 3)]), [0, 1, 2])

    def test_hypervolume_union_does_not_double_count_rectangles(self):
        self.assertEqual(hypervolume_2d([(1, 3), (2, 2), (3, 1)], (4, 4)), 6.)
        self.assertEqual(hypervolume_2d([(1, 3), (2, 2), (1, 3), (3, 3)], (4, 4)), 5.)

    def test_empty_and_outside_points_have_zero_volume(self):
        self.assertEqual(hypervolume_2d([], (1, 1)), 0.)
        self.assertEqual(hypervolume_2d([(2, 0), (0, 2), (1, 1)], (1, 1)), 0.)

    def test_scaling_and_monotonicity(self):
        before = hypervolume_2d([(1, 3), (2, 2)], (4, 4))
        self.assertEqual(hypervolume_2d([(2, 9), (4, 6)], (8, 12)), 6 * before)
        self.assertGreaterEqual(hypervolume_2d([(1, 3), (2, 2), (3, 1)], (4, 4)), before)
        with self.assertRaises(ValueError):
            hypervolume_2d([(float("nan"), 1)], (2, 2))


if __name__ == "__main__":
    unittest.main()
