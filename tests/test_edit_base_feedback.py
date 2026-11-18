import unittest
from pathlib import Path
from unittest.mock import patch

from chiplet_tuner.pipeline.tuner import MultiAgentTuner
from chiplet_tuner.core.schemas import EvaluationResult
from chiplet_tuner.tools.analysis_tools import AnalysisToolbox


class EditBaseFeedbackTests(unittest.TestCase):
    def test_effect_uses_actual_base_not_previous_failed_candidate(self):
        tuner = MultiAgentTuner.__new__(MultiAgentTuner)
        evaluations = {
            i: EvaluationResult(run_dir=Path(f'/tmp/iter_{i:03}'), hardware={}, detail_files={},
                metrics={'latency': value, 'energy': 1., 'mc': 1.})
            for i, value in [(9, 100.), (16, 10000.), (17, 75.)]
        }
        change = tuner._change_summary_from_transition({
            'source_iteration': 17, 'analysis_base': {'iteration': 9},
            'before_metrics': {'latency': 100., 'energy': 1., 'mc': 1., 'objective': 100.},
            'solution': {'strategy': 'bandwidth decrease', 'actions': ['nop_bw: 128->64']},
        })
        with patch.object(tuner, '_search_space', return_value={}):
            state, _ = tuner._build_search_state(17, evaluations, change)
        observation = state['search_observation']
        self.assertAlmostEqual(observation['objective_change_vs_previous'], -.9925)
        effect = observation['change_vs_edit_base']
        self.assertEqual(effect['base']['iteration'], 9)
        self.assertAlmostEqual(effect['relative_change']['objective'], -.25)
        compact = AnalysisToolbox()._compact_search_state_payload(state, {})
        self.assertEqual(compact['search_observation']['change_vs_edit_base'], effect)

    def test_legacy_transition_does_not_invent_base_metrics(self):
        tuner = MultiAgentTuner.__new__(MultiAgentTuner)
        change = tuner._change_summary_from_transition({'source_iteration': 2, 'solution': {}})
        self.assertNotIn('edit_base_metrics', change)


if __name__ == '__main__':
    unittest.main()
