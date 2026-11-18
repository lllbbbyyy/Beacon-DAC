from pathlib import Path
import tempfile
import unittest

from baseline.agentic_search import AgenticPhaseStrategy
from baseline.search import BaselineSearchRunner
from baseline.tests.test_baselines import _FakeSimulator
from chiplet_tuner.core.search_space import make_hardware_search_space
from chiplet_tuner.simulators.compass_config import build_compass_initial_hardware


class ScriptedController:
    def __init__(self):
        self.calls = []

    def config(self):
        return {"kind": "scripted_test_only"}

    def select_region(self, phase, domains, current, history, reason="scheduled"):
        self.calls.append(phase)
        if phase == 0:
            return {"num_chiplets": [2]}
        return {"num_chiplets": domains["num_chiplets"]}


class AgenticSearchTests(unittest.TestCase):
    def test_singleton_region_replans_without_phase_rollback(self):
        class Controller(ScriptedController):
            def select_region(self, phase, domains, current, history, reason="scheduled"):
                self.calls.append((phase, reason))
                return {key: values[:1] if phase == 0 else values for key, values in domains.items()}

        space = make_hardware_search_space("mixed", "64", "gpt3_merged", batch_size=128)
        hardware = build_compass_initial_hardware("mixed", "64")
        controller = Controller()
        strategy = AgenticPhaseStrategy(space, controller, phase_evaluations=60,
                                        warmup=5, folds=2, trees=3, candidate_pool=4)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            runner = BaselineSearchRunner(_FakeSimulator(root / "evals", space),
                                          strategy, hardware, space, root, 3)
            runner.run()
            self.assertEqual(controller.calls, [(0, "scheduled"),
                (1, "no_unevaluated_candidate_found_in_current_region")])
            self.assertEqual([r.action["metadata"]["phase"] for r in runner.history[1:]], [0, 1, 1])
            self.assertEqual(len({r.fingerprint for r in runner.history}), 4)

    def test_phases_and_real_surrogate_in_shared_runner(self):
        space = make_hardware_search_space("mixed", "64", "gpt3_merged", batch_size=128)
        hardware = build_compass_initial_hardware("mixed", "64")
        controller = ScriptedController()
        strategy = AgenticPhaseStrategy(space, controller, seed=7, phase_evaluations=4,
                                        warmup=2, folds=2, trees=3, candidate_pool=4)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            runner = BaselineSearchRunner(_FakeSimulator(root / "evals", space),
                                          strategy, hardware, space, root, 5)
            summary = runner.run()
            self.assertEqual(summary["total_evaluations_including_initial"], 6)
            self.assertEqual(controller.calls, [0, 1])
            self.assertTrue(all(r.hardware["num_chiplets"] == 2 for r in runner.history[1:5]))
            metadata = [r.action["metadata"] for r in runner.history[1:]]
            self.assertIn("ensemble_ei", [row["selection"] for row in metadata])
            self.assertEqual(metadata[-1]["phase_changes"]["num_chiplets"]["change"], "expand")
            self.assertEqual(len({r.fingerprint for r in runner.history}), 6)


if __name__ == "__main__":
    unittest.main()
