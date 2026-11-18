import json
from pathlib import Path
import tempfile
import unittest

from baseline.actions import UnifiedActionSpace
from baseline.random_search import RandomSearchStrategy
from baseline.search import BaselineSearchRunner
from baseline.tests.test_baselines import _FakeSimulator
from chiplet_tuner.core.search_space import make_hardware_search_space
from chiplet_tuner.simulators.base import MappingEvaluationUnavailable
from chiplet_tuner.simulators.compass_config import build_compass_initial_hardware


class MappingFailureTests(unittest.TestCase):
    def test_failed_candidate_is_charged_but_not_regressed_or_retried(self):
        class Simulator(_FakeSimulator):
            def evaluate(self, hardware, iteration):
                if iteration == 1:
                    raise MappingEvaluationUnavailable(self.output_root / "iter_001")
                return super().evaluate(hardware, iteration)

        space = make_hardware_search_space("mixed", "64", "gpt3_merged", batch_size=128)
        hardware = build_compass_initial_hardware("mixed", "64")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            strategy = RandomSearchStrategy(UnifiedActionSpace(space), seed=42)
            runner = BaselineSearchRunner(Simulator(root / "evals", space), strategy,
                                          hardware, space, root, evaluations=3)
            summary = runner.run()
            self.assertEqual(summary["attempted_evaluations_including_initial"], 4)
            self.assertEqual(summary["total_evaluations_including_initial"], 3)
            self.assertEqual(summary["failed_evaluations"], 1)
            self.assertEqual([r.iteration for r in runner.history], [0, 2, 3])
            failure = json.loads((root / "failed_evaluations.json").read_text())[0]
            self.assertNotIn("metrics", failure)
            self.assertGreaterEqual(failure["completed_elapsed_seconds"], 0)
            self.assertNotIn(failure["fingerprint"], {r.fingerprint for r in runner.history})


if __name__ == "__main__":
    unittest.main()
