import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np

from baseline.config_encoding import FullDesignCodec
from baseline.search import BaselineSearchRunner
from baseline.tests.test_baselines import _FakeSimulator
from chiplet_tuner.core.search_space import make_hardware_search_space
from chiplet_tuner.simulators.compass_config import build_compass_initial_hardware


class FullDesignCodecTests(unittest.TestCase):
    def setUp(self):
        self.space = make_hardware_search_space("mixed", "64", "gpt3_merged", batch_size=128)
        self.hardware = build_compass_initial_hardware("mixed", "64")
        self.codec = FullDesignCodec(self.hardware, self.space)

    def test_initial_round_trip(self):
        self.assertEqual(self.codec.decode(self.codec.encode(self.hardware)), self.hardware)

    def test_random_full_configurations_round_trip(self):
        rng = np.random.default_rng(9)
        for _ in range(20):
            action = rng.integers(self.codec.nvec)
            hardware = self.codec.decode(action)
            self.assertEqual(self.codec.decode(self.codec.encode(hardware)), hardware)
            self.assertEqual(len(hardware["chiplets"]), hardware["num_chiplets"])
            self.assertEqual(hardware["chip_y"] * hardware["chip_x"], hardware["num_chiplets"])

    def test_inactive_slots_do_not_change_design(self):
        action = self.codec.encode(self.hardware)
        changed = action.copy()
        start = 1 + 3 * self.hardware["num_chiplets"]
        end = 1 + 3 * self.codec.max_count
        changed[start:end] = self.codec.nvec[start:end] - 1
        self.assertEqual(self.codec.decode(action), self.codec.decode(changed))

    def test_invalid_action_rejected(self):
        for action in [self.codec.nvec, -np.ones_like(self.codec.nvec),
                       np.zeros(3, dtype=int), np.zeros_like(self.codec.nvec, dtype=float)]:
            with self.assertRaises(ValueError):
                self.codec.decode(action)


@unittest.skipUnless(importlib.util.find_spec("stable_baselines3"), "Install baseline/requirements-ppo.txt")
class PPOTrainingTests(FullDesignCodecTests):
    def test_reward_orders_valid_designs_above_mapping_failure(self):
        from baseline.chiplet_gym_ppo import objective_reward, PPOSettings
        rewards = [objective_reward(1, value) for value in (1e-300, 0.5, 1, 2, 1e300)]
        self.assertEqual(rewards, sorted(rewards, reverse=True))
        self.assertTrue(all(-1 < value < 1 for value in rewards))
        self.assertEqual(rewards[2], 0)
        with self.assertRaises(ValueError):
            PPOSettings(mapping_failure_reward=-0.5).validate()
        for value in (0, -1, float("inf"), float("nan")):
            with self.assertRaises(ValueError):
                objective_reward(1, value)

    def test_mapping_failure_keeps_measurement_and_consumes_attempt(self):
        import time
        from baseline.chiplet_gym_ppo import HardwareEnv, PPOSettings, _SearchStopped
        from chiplet_tuner.simulators.base import MappingEvaluationUnavailable

        class Simulator(_FakeSimulator):
            def evaluate(self, hardware, iteration):
                if iteration == 1:
                    raise MappingEvaluationUnavailable(self.output_root / "iter_001")
                return super().evaluate(hardware, iteration)

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            runner = BaselineSearchRunner(Simulator(root / "evals", self.space),
                SimpleNamespace(name="test"), self.hardware, self.space, root, 2,
                started_at=time.perf_counter())
            runner.history.append(runner._evaluate(self.hardware, 0, None, None))
            env = HardwareEnv(runner, PPOSettings())
            before, _ = env.reset()
            observation, reward, terminated, _, info = env.step(self.codec.encode(self.hardware))
            np.testing.assert_array_equal(observation, before)
            self.assertEqual(reward, -1)
            self.assertTrue(terminated and info["mapping_unavailable"])
            self.assertEqual(len(runner.history), 1)
            self.assertEqual(len(runner.failed_evaluations), 1)
            self.assertNotIn("metrics", runner.failed_evaluations[0])
            env.reset()
            env.step(self.codec.encode(self.hardware))
            self.assertEqual(runner.history[-1].iteration, 2)
            with self.assertRaises(_SearchStopped):
                env.step(self.codec.encode(self.hardware))

    def test_real_ppo_updates_and_respects_evaluation_cap(self):
        import torch
        from stable_baselines3 import PPO
        from baseline.chiplet_gym_ppo import PPOSettings, run_ppo

        threads = torch.get_num_threads()
        torch.set_num_threads(1)
        try:
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                runner = BaselineSearchRunner(
                    _FakeSimulator(root / "evals", self.space),
                    SimpleNamespace(name="chiplet_gym_derived_ppo"),
                    self.hardware, self.space, root, 9,
                )
                result = run_ppo(runner, PPOSettings(rollout_steps=4, batch_size=2, epochs=2))
                self.assertEqual(result["total_evaluations_including_initial"], 10)
                self.assertEqual(result["ppo_transitions_collected"], 9)
                self.assertEqual(result["ppo_optimizer_epochs_completed"], 4)
                self.assertTrue((root / "ppo_policy.zip").exists())
                model = PPO.load(root / "ppo_policy.zip", device="cpu")
                self.assertEqual(model.n_epochs, 2)
                logs = (root / "training/progress.json").read_text().splitlines()
                self.assertTrue(any("train/policy_gradient_loss" in json.loads(line) for line in logs))
                self.assertTrue(all(r.within_time_budget for r in runner.history))
                self.assertEqual(len(runner.history), len((root / "history.jsonl").read_text().splitlines()))
        finally:
            torch.set_num_threads(threads)


if __name__ == "__main__":
    unittest.main()
