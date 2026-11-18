import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

from chiplet_tuner.core.search_budget import SearchBudget
from chiplet_tuner.pipeline.langgraph_tuner import LangGraphTuner


class SearchBudgetTests(unittest.TestCase):
    def test_validation_and_boundary(self):
        for seconds in (0, -1, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                SearchBudget(seconds, 0)
        budget = SearchBudget(10, 2, clock=lambda: 12)
        self.assertTrue(budget.expired())
        self.assertTrue(budget.eligible(10))
        self.assertFalse(budget.eligible(10.01))

    def test_resume_preserves_deadline(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "clock.json"
            SearchBudget.open(path, 30, 100, False)
            restored = SearchBudget.open(path, None, 110, True)
            self.assertEqual(restored.started_at, 100)
            self.assertEqual(restored.seconds, 30)
            with self.assertRaises(ValueError):
                SearchBudget.open(path, 60, 110, True)
            with self.assertRaises(FileExistsError):
                SearchBudget.open(path, 30, 110, False)

    def test_legacy_resume_cannot_gain_new_budget(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "clock.json"
            with self.assertRaises(ValueError):
                SearchBudget.open(path, 30, 100, True)
            self.assertIsNone(SearchBudget.open(path, None, 100, True).seconds)

    def test_late_baseline_is_not_best(self):
        now = [0]
        with tempfile.TemporaryDirectory() as folder:
            tuner = LangGraphTuner.__new__(LangGraphTuner)
            tuner.search_budget = SearchBudget(10, 0, clock=lambda: now[0])
            tuner.output_root = Path(folder)
            tuner.progress = MagicMock()
            tuner._record_timing = MagicMock()
            tuner._write_graph_state_marker = MagicMock()
            tuner._write_tuning_metrics_outputs = MagicMock()
            tuner._initial_change_summary = lambda: {}
            tuner._objective = lambda *args: 1.0
            tuner._metrics_with_objective = lambda *args: {"objective": 1.0}
            tuner._evaluation_record = lambda result: {}

            def evaluate(*args):
                now[0] = 11
                return SimpleNamespace(metrics={}, hardware={})

            tuner.simulator = SimpleNamespace(evaluate=evaluate)
            state = tuner._node_baseline({"current_hardware": {}})
            self.assertEqual(state["best"], {})
            self.assertFalse(state["iteration_metrics"][0]["within_time_budget"])
            self.assertFalse((Path(folder) / "best_hardware.json").exists())
            self.assertEqual(tuner._route_after_prepare({**state, "iterations": 100}), "finish")

    def test_graph_stops_after_agent_exhausts_budget(self):
        now = [0]
        tuner = LangGraphTuner.__new__(LangGraphTuner)
        tuner.agent_mode = "multi"
        tuner.search_budget = SearchBudget(10, 0, clock=lambda: now[0])
        called = []
        for name in ("baseline", "prepare_iteration", "layer_level_agent",
                     "solution_generation_agent", "single_agent", "write_trace",
                     "repair_duplicate", "build_transition", "evaluate", "commit", "finalize"):
            def node(state, name=name):
                called.append(name)
                return {}
            setattr(tuner, "_node_" + name, node)

        def model(state):
            called.append("model_level_agent")
            now[0] = 11
            return {}

        tuner._node_model_level_agent = model
        graph = tuner._build_langgraph().compile()
        graph.invoke({"iterations": 100, "next_iteration": 1})
        self.assertEqual(called, ["baseline", "prepare_iteration", "model_level_agent", "finalize"])

    def test_guard_prevents_expired_resume_node(self):
        tuner = LangGraphTuner.__new__(LangGraphTuner)
        tuner.search_budget = SearchBudget(10, 0, clock=lambda: 11)
        node = MagicMock()
        self.assertEqual(tuner._budget_guard(node)({}), {"budget_stopped": True})
        node.assert_not_called()

    def test_late_improvement_does_not_replace_best(self):
        with tempfile.TemporaryDirectory() as folder:
            tuner = LangGraphTuner.__new__(LangGraphTuner)
            tuner.search_budget = SearchBudget(10, 0, clock=lambda: 11)
            tuner.output_root = Path(folder)
            tuner.agent_mode = "multi"
            tuner.progress = MagicMock()
            for name in ("_record_timing", "_commit_evaluated_transition",
                         "_write_tuning_metrics_outputs", "_write_graph_state_marker"):
                setattr(tuner, name, MagicMock())
            tuner._evaluation_from_record = lambda x: SimpleNamespace(metrics={}, hardware={})
            tuner._proposal_from_record = lambda x: SimpleNamespace(updated_hardware={}, actions=[])
            tuner._bottleneck_state_from_record = lambda x: SimpleNamespace(primary_impact="latency", dominant_root_cause="compute")
            tuner._evaluation_record = lambda x: {}
            tuner._change_summary_from_transition = lambda x: {}
            tuner._objective = lambda *args: 1.0
            tuner._metrics_with_objective = lambda *args: {"objective": 1.0}
            state = dict(next_iteration=1, iterations=100, current_evaluation={}, proposal={},
                         bottleneck_state={}, pending_transition={}, evaluations_by_iteration={},
                         evaluated_fingerprints=[], next_fingerprint="test",
                         applied_changes_by_iteration={}, best={"score": 2.0},
                         iteration_metrics=[], phase="guided", initial_objective=2.0,
                         completed_elapsed_seconds=11.0)
            result = tuner._node_commit(state)
            self.assertEqual(result["best"], {"score": 2.0})
            self.assertFalse(result["iteration_metrics"][-1]["is_new_best"])
            self.assertFalse((Path(folder) / "best_hardware.json").exists())

            # Eligibility is fixed at evaluation completion, not after history writes.
            state["completed_elapsed_seconds"] = 9.0
            result = tuner._node_commit(state)
            self.assertEqual(result["best"]["score"], 1.0)
            self.assertTrue((Path(folder) / "best_hardware.json").exists())


if __name__ == "__main__":
    unittest.main()
