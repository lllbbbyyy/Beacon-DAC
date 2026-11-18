import unittest

import numpy as np

from baseline.agentic_region import PhaseRegion
from baseline.config_encoding import FullDesignCodec
from chiplet_tuner.core.search_space import make_hardware_search_space
from chiplet_tuner.simulators.compass_config import build_compass_initial_hardware


class PhaseRegionTests(unittest.TestCase):
    def setUp(self):
        space = make_hardware_search_space("mixed", "64", "gpt3_merged", batch_size=128)
        self.codec = FullDesignCodec(build_compass_initial_hardware("mixed", "64"), space)
        self.region = PhaseRegion(self.codec)

    def test_narrow_shift_expand_and_sampling(self):
        key = "chiplets.0.compute_units"
        domain = self.region.domains[key]
        self.assertEqual(self.region.update({key: domain[:1]})[key]["change"], "narrow")
        self.assertEqual(self.region.update({key: domain[-1:]})[key]["change"], "shift")
        hardware = self.region.decode_unit(np.zeros_like(self.codec.nvec, dtype=float))
        self.assertEqual(hardware["chiplets"][0]["compute_units"], domain[-1])
        self.assertTrue(self.region.contains(hardware))
        self.assertEqual(self.region.update({key: domain})[key]["change"], "expand")

    def test_atomic_validation(self):
        before = self.region.snapshot()
        for invalid in ({"unknown": [1]}, {"num_chiplets": []},
                        {"num_chiplets": [True]}, {"num_chiplets": [2, 2]},
                        {"num_chiplets": self.codec.counts[:1], "unknown": [1]}):
            with self.assertRaises(ValueError):
                self.region.update(invalid)
            self.assertEqual(self.region.snapshot(), before)

    def test_inactive_slots_do_not_filter_history(self):
        hardware = self.region.decode_unit(np.zeros_like(self.codec.nvec, dtype=float))
        inactive = hardware["num_chiplets"]
        self.assertLess(inactive, self.codec.max_count)
        key = f"chiplets.{inactive}.compute_units"
        self.region.update({key: self.codec.compute[-1:]})
        self.assertTrue(self.region.contains(hardware))

    def test_order_matches_codec_and_all_samples_legal(self):
        self.assertEqual(len(self.region.domains), len(self.codec.nvec))
        rng = np.random.default_rng(42)
        for _ in range(12):
            hardware = self.region.decode_unit(rng.random(len(self.codec.nvec)))
            self.assertTrue(self.region.contains(hardware))
            self.assertEqual(self.codec.decode(self.codec.encode(hardware)), hardware)
        for value in (-0.1, 1.0, float("nan")):
            with self.assertRaises(ValueError):
                self.region.decode_unit(np.full(self.codec.nvec.shape, value))

    def test_small_enumeration_ignores_inactive_slots(self):
        self.assertIsNone(self.region.enumerate_small(limit=4))
        selected = {key: values[:1] for key, values in self.region.domains.items()}
        self.region.update(selected)
        count = self.region.choices["num_chiplets"][0]
        for index in range(count, self.codec.max_count):
            key = f"chiplets.{index}.compute_units"
            self.region.update({key: self.region.domains[key]})
        designs = self.region.enumerate_small(limit=1)
        self.assertEqual(len(designs), 1)
        self.assertTrue(self.region.contains(designs[0]))


if __name__ == "__main__":
    unittest.main()
