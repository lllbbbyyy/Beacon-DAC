from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import MagicMock

from chiplet_tuner.core.search_budget import SearchBudget
from chiplet_tuner.pipeline.langgraph_tuner import LangGraphTuner
from chiplet_tuner.simulators.base import MappingEvaluationUnavailable


class MappingFailureStateTests(unittest.TestCase):
    def test_failure_advances_without_committing_measurement(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            tuner = LangGraphTuner.__new__(LangGraphTuner)
            tuner.output_root = root
            tuner.search_budget = SearchBudget(100, 0, clock=lambda: 2)
            tuner.agent_mode = "multi"
            tuner.tuning_stages = 7
            tuner.progress = MagicMock()
            tuner._record_timing = MagicMock()
            tuner._write_graph_state_marker = MagicMock()
            tuner._proposal_from_record = lambda _: SimpleNamespace(updated_hardware={"test": 1})
            tuner.simulator = SimpleNamespace(evaluate=MagicMock(
                side_effect=MappingEvaluationUnavailable(root / "iter_001")))
            tuner._commit_evaluated_transition = MagicMock()
            state = {"next_iteration": 1, "iterations": 10, "proposal": {},
                     "next_fingerprint": "failed", "best": {"score": 1}}
            update = tuner._node_evaluate(state)
            self.assertNotIn("current_evaluation", update)
            committed = tuner._node_commit({**state, **update})
            self.assertEqual(committed["next_iteration"], 2)
            self.assertEqual(len(committed["failed_evaluations"]), 1)
            self.assertNotIn("metrics", committed["failed_evaluations"][0])
            self.assertNotIn("best", committed)
            self.assertNotIn("current_hardware", committed)
            tuner._commit_evaluated_transition.assert_not_called()
            self.assertTrue((root / "failed_evaluations.json").is_file())
            tuner._record_timing.assert_called_once()


if __name__ == "__main__":
    unittest.main()
