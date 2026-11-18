import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

from baseline.search import BaselineSearchRunner
from baseline.tests.test_baselines import _FakeSimulator
from chiplet_tuner.core.search_space import make_hardware_search_space
from chiplet_tuner.simulators.compass_config import build_compass_initial_hardware


@unittest.skipUnless(importlib.util.find_spec("botorch"), "Install baseline/requirements-bo.txt")
class HascoBOTests(unittest.TestCase):
    def test_sobol_then_fitted_gp_acquisition(self):
        import torch
        from baseline.hasco_bo import HascoScalarBOStrategy

        space = make_hardware_search_space("mixed", "64", "gpt3_merged", batch_size=128)
        hardware = build_compass_initial_hardware("mixed", "64")
        threads = torch.get_num_threads()
        torch.set_num_threads(1)
        try:
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                strategy = HascoScalarBOStrategy(space, seed=7, warmup=2, candidate_pool_size=8, fit_maxiter=5)
                runner = BaselineSearchRunner(
                    _FakeSimulator(root / "evals", space), strategy, hardware, space, root, 3,
                )
                summary = runner.run()
                self.assertEqual(summary["total_evaluations_including_initial"], 4)
                selections = [r.action["metadata"]["selection"] for r in runner.history[1:]]
                self.assertEqual(selections, ["sobol_warmup", "sobol_warmup", "gp_log_ei"])
                self.assertEqual(len({r.fingerprint for r in runner.history}), 4)
                metadata = runner.history[-1].action["metadata"]
                self.assertEqual(metadata["training_designs"], 3)
                self.assertGreater(metadata["predicted_log_std"], 0)
                config = json.loads((root / "baseline_config.json").read_text())
                self.assertEqual(config["strategy_config"]["acquisition"], "LogExpectedImprovement")
        finally:
            torch.set_num_threads(threads)


if __name__ == "__main__":
    unittest.main()
