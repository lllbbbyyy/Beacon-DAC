import unittest
import os
from pathlib import Path
from unittest.mock import patch

from chiplet_tuner.core.search_coverage import summarize_parameter_coverage
from chiplet_tuner.agents.analysis_policy import resolve_analysis_policy
from chiplet_tuner.core.schemas import EvaluationResult
from chiplet_tuner.tools.analysis_tools import AnalysisToolbox, ToolContext
from chiplet_tuner.tools.evidence_policy import ScalarEvidenceToolbox


class SearchCoverageTests(unittest.TestCase):
    def test_joint_change_is_not_single_parameter_evidence(self):
        base = {"dram_bw": 64, "micro_batch": 1}
        rows = [dict(iteration=0, hardware=base, objective=100),
                dict(iteration=1, hardware={"dram_bw": 32, "micro_batch": 32}, objective=20),
                dict(iteration=2, hardware={"dram_bw": 128, "micro_batch": 1}, objective=110)]
        result = summarize_parameter_coverage(rows, {"dram_bw": [16, 32, 64, 128]}, base)
        values = result["parameters"]["dram_bw"]["values"]
        self.assertEqual(values[0]["evaluations"], 0)
        self.assertIsNone(values[1]["same_other_parameters"])
        self.assertEqual(values[1]["best_observed"]["objective"], 20)
        self.assertEqual(values[3]["same_other_parameters"]["objective"], 110)
        self.assertEqual(base, {"dram_bw": 64, "micro_batch": 1})

    def test_invalid_objective_rejected(self):
        for objective in [0, -1, float("nan"), float("inf")]:
            with self.assertRaises(ValueError):
                summarize_parameter_coverage([dict(hardware={}, objective=objective)], {}, {})

    def test_opt_in_preserves_tool_budget(self):
        count, prompt = resolve_analysis_policy("model_level", 15, "objective_guided")
        self.assertEqual(count, 15)
        self.assertIn("increase from decrease", prompt)
        self.assertEqual(resolve_analysis_policy("model_level", 15, "standard"), (15, ""))

    def test_full_and_scalar_receive_same_coverage_without_report_access(self):
        evaluation = EvaluationResult(run_dir=Path('/test/iter_000'),
            hardware={"dram_bw": 64}, metrics={"latency": 2, "energy": 3, "mc": 4},
            detail_files={"latency": Path('/test/private_report.json')})
        context = ToolContext(output_dir=Path('/test'), evaluation=evaluation,
            current_hardware=evaluation.hardware,
            evaluation_bases={"current": evaluation, "best": evaluation},
            simulator_schema={"search_space": {"system_params": {"dram_bw": [32, 64]}}})
        with patch.dict(os.environ, {"CHIPLET_TUNER_ANALYSIS_POLICY": "objective_guided"}), \
                patch.object(Path, 'read_text', side_effect=AssertionError('report access')):
            full = AnalysisToolbox().compare_search_states(context).payload['parameter_coverage']
            scalar = ScalarEvidenceToolbox().compare_search_states(context).payload['parameter_coverage']
        self.assertEqual(full, scalar)
        self.assertEqual(full['parameters']['dram_bw']['values'][1]['evaluations'], 1)
        with patch.dict(os.environ, {"CHIPLET_TUNER_ANALYSIS_POLICY": "standard"}):
            self.assertNotIn('parameter_coverage', AnalysisToolbox().compare_search_states(context).payload)


if __name__ == "__main__":
    unittest.main()
