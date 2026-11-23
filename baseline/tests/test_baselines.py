from __future__ import annotations

import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path
from typing import Any, Dict

from baseline.actions import ACTION_CATEGORIES, UnifiedActionSpace
from baseline.bayesian_optimization import BayesianOptimizationStrategy
from baseline.online_rl import OnlineRLStrategy
from baseline.random_search import RandomSearchStrategy
from baseline.search import BaselineSearchRunner
from chiplet_tuner.core.objective import hardware_compute_scale
from chiplet_tuner.core.schemas import EvaluationResult
from chiplet_tuner.core.search_space import (
    compute_unit_candidates,
    macs_for_compute_units,
    make_hardware_search_space,
    micro_batch_candidates,
    num_chiplet_candidates,
    system_param_candidates,
)
from chiplet_tuner.simulators.base import SimulatorAdapter
from chiplet_tuner.simulators.compass_config import (
    build_compass_initial_hardware,
    build_compass_search_config,
)


class _FakeSimulator(SimulatorAdapter):
    def __init__(self, output_root: Path, search_space: Dict[str, Any]) -> None:
        self.output_root = output_root
        self.search_space = search_space

    def evaluate(self, hardware: Dict[str, Any], iteration: int) -> EvaluationResult:
        run_dir = self.output_root / f"iter_{iteration:03d}"
        run_dir.mkdir(parents=True, exist_ok=True)
        chiplets = hardware["chiplets"]
        total_compute = sum(float(item["compute_units"]) for item in chiplets)
        total_buffer = sum(float(item["buffer_size"]) for item in chiplets)
        dataflows = {str(item["type"]) for item in chiplets}
        hetero_factor = 0.88 if len(dataflows) > 1 else 1.0
        metrics = {
            "latency": max(1.0, 2.0e7 / total_compute) * hetero_factor,
            "energy": 1.0 + total_compute / 1.0e5 + total_buffer / 1.0e6,
            "mc": 1.0 + len(chiplets) * 0.05 + total_buffer / 1.0e6,
        }
        return EvaluationResult(
            run_dir=run_dir,
            hardware=hardware,
            metrics=metrics,
            detail_files={},
        )

    def schema(self) -> Dict[str, Any]:
        return {"adapter": "fake", "search_space": self.search_space}


class BaselineTest(unittest.TestCase):
    def setUp(self) -> None:
        self.search_space = make_hardware_search_space(
            task_type="mixed",
            compute_scale="64",
            model_type="gpt3_merged",
            batch_size=128,
        )
        self.hardware = build_compass_initial_hardware(
            task_type="mixed",
            compute_scale="64",
        )
        self.action_space = UnifiedActionSpace(self.search_space)

    def test_tensor_parallel_is_a_tunable_system_parameter(self) -> None:
        self.assertEqual(
            system_param_candidates(self.search_space, "tensor_parall"),
            [1, 2, 4, 8, 16, 32, 64],
        )
        self.assertIn(self.hardware["tensor_parall"], [1, 2, 4, 8, 16, 32, 64])
        actions = self.action_space.candidates(self.hardware, parent_iteration=0)
        self.assertTrue(any(action.action_id.startswith("system:tensor_parall:") for action in actions))

        initialized = build_compass_initial_hardware(
            task_type="mixed",
            compute_scale="64",
            tensor_parall=16,
        )
        self.assertEqual(initialized["tensor_parall"], 16)

    def test_qkv_projection_mode_is_written_to_model_info(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            fake_compass = Path(tmp) / "Compass"
            (fake_compass / "config").mkdir(parents=True)
            for name in (
                "fixed_govreport_input_token_lens.json",
                "govreport_output_token_lens.json",
                "fixed_sharegpt_input_token_lens.json",
                "sharegpt_output_token_lens.json",
            ):
                (fake_compass / "config" / name).write_text("[2048, 128]\n")

            config = build_compass_search_config(
                compass_root=fake_compass,
                workload="chunked_prefill",
                dataset="govreport",
                compute_scale="64",
            )
            self.assertEqual(config["model_info"]["qkv_projection_mode"], "fused_tp")
            self.assertEqual(config["chunked_prefill_size"], 2048)

            separate_config = build_compass_search_config(
                compass_root=fake_compass,
                workload="mixed",
                dataset="sharegpt",
                compute_scale="64",
                qkv_projection_mode="separate",
            )
            self.assertEqual(separate_config["model_info"]["qkv_projection_mode"], "separate")

    def test_micro_batch_is_a_tunable_legal_divisor(self) -> None:
        self.assertEqual(
            system_param_candidates(self.search_space, "micro_batch"),
            [1, 2, 4, 8, 16, 32, 64, 128],
        )
        self.assertEqual(micro_batch_candidates(12), [1, 2, 3, 4, 6, 12])
        self.assertEqual(self.hardware["micro_batch"], 1)
        actions = self.action_space.candidates(self.hardware, parent_iteration=0)
        self.assertTrue(any(action.action_id == "system:micro_batch:+1" for action in actions))

        initialized = build_compass_initial_hardware(
            task_type="mixed",
            compute_scale="64",
            micro_batch=32,
            batch_size=128,
        )
        self.assertEqual(initialized["micro_batch"], 32)

    def test_expanded_compute_candidates_have_valid_mac_layouts(self) -> None:
        candidates = compute_unit_candidates(self.search_space)
        self.assertEqual(candidates[-2:], [262144, 524288])
        for compute_units in candidates[-2:]:
            macs = macs_for_compute_units(compute_units, "ws", self.search_space)
            dimensions = [int(value) for value in str(macs).strip("[]").split(",")]
            product = 1
            for dimension in dimensions:
                product *= dimension
            self.assertEqual(product, compute_units)

    def test_unified_actions_are_legal_and_unique(self) -> None:
        actions = self.action_space.candidates(self.hardware, parent_iteration=0)
        self.assertGreater(len(actions), 10)
        self.assertEqual(len(actions), len({action.fingerprint for action in actions}))
        categories = {action.category for action in actions}
        self.assertTrue(categories.issubset(set(ACTION_CATEGORIES)))
        for required in [
            "chiplet_count",
            "compute_one",
            "compute_all",
            "buffer_one",
            "buffer_all",
            "dataflow_one",
            "system",
        ]:
            self.assertIn(required, categories)
        legal_counts = set(num_chiplet_candidates(self.search_space))
        for action in actions:
            candidate = action.hardware
            self.assertIn(candidate["num_chiplets"], legal_counts)
            self.assertEqual(candidate["num_chiplets"], len(candidate["chiplets"]))
            self.assertEqual(candidate["num_chiplets"], candidate["chip_x"] * candidate["chip_y"])
        increased = next(action for action in actions if action.action_id == "chiplet_count:+1")
        self.assertEqual([item["type"] for item in increased.hardware["chiplets"]], ["ws", "os", "ws", "os"])
        self.assertEqual(hardware_compute_scale(self.hardware, self.search_space), 64.0)

    def test_all_strategies_run_with_the_common_runner(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            strategies = [
                RandomSearchStrategy(self.action_space, seed=7),
                BayesianOptimizationStrategy(
                    self.action_space,
                    self.search_space,
                    seed=7,
                    warmup=1,
                    candidate_pool_size=64,
                    max_parents=4,
                ),
                OnlineRLStrategy(
                    self.action_space,
                    self.search_space,
                    seed=7,
                    update_interval=2,
                    update_epochs=2,
                ),
            ]
            for strategy in strategies:
                output_dir = root / strategy.name
                runner = BaselineSearchRunner(
                    simulator=_FakeSimulator(output_dir / "evaluations", self.search_space),
                    strategy=strategy,
                    initial_hardware=self.hardware,
                    search_space=self.search_space,
                    output_dir=output_dir,
                    evaluations=4,
                )
                summary = runner.run()
                self.assertEqual(summary["total_evaluations_including_initial"], 5)
                self.assertTrue((output_dir / "history.jsonl").exists())
                self.assertTrue((output_dir / "best_hardware.json").exists())
                self.assertEqual(len({record.fingerprint for record in runner.history}), 5)
                action_metadata = [
                    record.action["metadata"]
                    for record in runner.history[1:]
                    if record.action is not None
                ]
                if strategy.name == "random":
                    self.assertTrue(all(item["selection"] == "uniform_random_action" for item in action_metadata))
                elif strategy.name == "bayesian_optimization":
                    self.assertTrue(any(item["selection"] == "expected_improvement" for item in action_metadata))
                else:
                    self.assertTrue(all("reward_log_objective_improvement" in item for item in action_metadata))

    def test_time_budget_excludes_evaluation_overrun(self) -> None:
        clock = [0.0]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            simulator = _FakeSimulator(root, self.search_space)
            evaluate = simulator.evaluate

            def timed_evaluate(hardware, iteration):
                clock[0] += 3.0
                return evaluate(hardware, iteration)

            simulator.evaluate = timed_evaluate
            strategy = RandomSearchStrategy(self.action_space, seed=7)
            runner = BaselineSearchRunner(
                simulator, strategy, self.hardware, self.search_space, root, 10,
                time_budget_seconds=5.0,
            )
            with patch("baseline.search.time.perf_counter", side_effect=lambda: clock[0]):
                summary = runner.run()
            self.assertEqual(summary["stop_reason"], "time_budget")
            self.assertEqual(summary["total_evaluations_including_initial"], 2)
            self.assertEqual(summary["eligible_evaluations_including_initial"], 1)
            self.assertEqual(summary["best_iteration"], 0)
            self.assertEqual([r.completed_elapsed_seconds for r in runner.history], [3.0, 6.0])
            self.assertFalse(runner.history[-1].within_time_budget)

    def test_time_budget_includes_proposal_and_setup(self) -> None:
        clock = [2.0]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            strategy = RandomSearchStrategy(self.action_space, seed=7)
            propose = strategy.propose

            def timed_propose(history, forbidden):
                clock[0] += 4.0
                return propose(history, forbidden)

            strategy.propose = timed_propose
            runner = BaselineSearchRunner(
                _FakeSimulator(root, self.search_space), strategy,
                self.hardware, self.search_space, root, 10,
                time_budget_seconds=5.0, started_at=0.0,
            )
            with patch("baseline.search.time.perf_counter", side_effect=lambda: clock[0]):
                summary = runner.run()
            self.assertEqual(summary["total_evaluations_including_initial"], 1)
            self.assertEqual(summary["total_elapsed_seconds"], 6.0)
            self.assertEqual(summary["stop_reason"], "time_budget")

    def test_no_eligible_initial_has_no_best_result(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            runner = BaselineSearchRunner(
                _FakeSimulator(root, self.search_space),
                RandomSearchStrategy(self.action_space, seed=7),
                self.hardware, self.search_space, root, 10,
                time_budget_seconds=1.0, started_at=0.0,
            )
            with patch("baseline.search.time.perf_counter", return_value=2.0):
                summary = runner.run()
            self.assertIsNone(summary["best_objective"])
            self.assertFalse((root / "best_hardware.json").exists())

    def test_invalid_time_budgets_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for budget in [0.0, -1.0, float("inf"), float("nan")]:
                with self.assertRaises(ValueError):
                    BaselineSearchRunner(
                        _FakeSimulator(root, self.search_space),
                        RandomSearchStrategy(self.action_space, seed=7),
                        self.hardware, self.search_space, root, 10,
                        time_budget_seconds=budget,
                    )


if __name__ == "__main__":
    unittest.main()
