import unittest
import json
import tempfile
from pathlib import Path
from unittest.mock import patch

from chiplet_tuner.agents.analysis_policy import analysis_output_prompt, resolve_analysis_policy
from chiplet_tuner.agents.model_level import ModelLevelAgent
from chiplet_tuner.agents.layer_level import LayerLevelAgent
from chiplet_tuner.agents.tool_use import _retry_response_schema
from chiplet_tuner.core.schemas import EvaluationResult
from chiplet_tuner.llm.clients import MockLLMClient
from chiplet_tuner.tools.analysis_tools import AnalysisToolbox
from chiplet_tuner.tools.evidence_policy import ScalarEvidenceToolbox


class SystemAnalysisTest(unittest.TestCase):
    def setUp(self):
        self.env = patch.dict("os.environ", {"CHIPLET_TUNER_ANALYSIS_POLICY": "decision_first"})
        self.env.start()
        self.addCleanup(self.env.stop)
        self.evaluation = EvaluationResult(
            run_dir=Path("iter_000"), hardware={},
            metrics={"latency": 2., "energy": 3., "mc": 4.}, detail_files={},
        )

    def test_model_accepts_system_tradeoff_without_inventing_layers(self):
        agent = ModelLevelAgent(MockLLMClient(), AnalysisToolbox())
        agent._validate_llm_result({"summary": "Test a cost-latency tradeoff.", "candidate_layers": []}, {}, self.evaluation)
        with self.assertRaises(ValueError):
            agent._normalize_candidate_layers([None], {}, self.evaluation)
        with patch.dict("os.environ", {"CHIPLET_TUNER_ANALYSIS_POLICY": "standard"}):
            with self.assertRaises(ValueError):
                agent._normalize_candidate_layers([], {}, self.evaluation)

    def test_empty_diagnosis_does_not_claim_measured_root_cause(self):
        agent = LayerLevelAgent(MockLLMClient(), AnalysisToolbox())
        payload = dict(primary_impact="monetary_cost", dominant_root_cause="unknown",
                       layer_diagnoses=[], retrieval_description="Test lower cost with latency tradeoff.",
                       root_cause_summary=dict(compute=0., memory=0., communication=0., buffer=0.),
                       recommended_focus=["bandwidth"])
        self.assertEqual(agent._parse_state(payload).layer_diagnoses, [])
        with self.assertRaises(ValueError):
            agent._parse_state(dict(payload, dominant_root_cause="memory"))
        with self.assertRaises(ValueError):
            agent._parse_state(dict(payload, root_cause_summary={"memory": 1.}))

    def test_scalar_permissions_and_step_budget_unchanged(self):
        agent = ModelLevelAgent(MockLLMClient(), ScalarEvidenceToolbox())
        with self.assertRaises(ValueError):
            agent._normalize_candidate_layers([{"layer_id": 1}], {}, self.evaluation)
        for role in ("model_level", "layer_level", "solution_generation"):
            steps, guidance = resolve_analysis_policy(role, 15)
            self.assertEqual(steps, 15)
            self.assertIn("SYSTEM-LEVEL ANALYSIS", guidance)
        self.assertIn("candidate_layers may be []", analysis_output_prompt("schema"))
        for task, key in (("model_level_analysis", "candidate_layers"), ("layer_level_analysis", "layer_diagnoses")):
            self.assertEqual(_retry_response_schema(task, "")["properties"][key]["min_items"], 0)
        with patch.dict("os.environ", {"CHIPLET_TUNER_ANALYSIS_POLICY": "standard"}):
            self.assertEqual(analysis_output_prompt("schema"), "schema")

    def test_real_graph_and_final_handoff_without_layer_claims(self):
        class ScriptedClient(MockLLMClient):
            def __init__(self, responses):
                super().__init__()
                self.responses = iter(responses)
                self.calls = []

            def complete_json_messages(self, messages):
                self.calls.append(json.dumps(messages))
                return next(self.responses)

        metric_call = {"thought": "Read aggregate observations.", "action": "tool",
                       "tool": {"name": "summarize_metrics", "arguments": {}}}
        finish = {"thought": "The proposed system tradeoff needs evaluation, not a layer claim.",
                  "action": "finish"}
        model_final = {"summary": "Measured L=2, E=3, MC=4. Test a cost-latency tradeoff.",
                       "candidate_layers": [], "notes": {"global_findings": ["MC=4"]}}
        layer_final = dict(primary_impact="monetary_cost", dominant_root_cause="unknown",
                           layer_diagnoses=[], retrieval_description="Test system cost-latency tradeoff.",
                           root_cause_summary=dict(compute=0., memory=0., communication=0., buffer=0.),
                           recommended_focus=["dram_bw"])
        for toolbox in (AnalysisToolbox(), ScalarEvidenceToolbox()):
            with self.subTest(toolbox=type(toolbox).__name__), tempfile.TemporaryDirectory() as tmp:
                client = ScriptedClient([metric_call, finish, model_final, finish, layer_final])
                with patch.dict("os.environ", {"CHIPLET_TUNER_REACT_EXECUTION_MODE": "iterative"}):
                    model = ModelLevelAgent(client, toolbox).analyze(self.evaluation, Path(tmp))
                    state = LayerLevelAgent(client, toolbox).analyze(model, self.evaluation, Path(tmp))
                self.assertEqual(len(client.calls), 5)
                self.assertEqual(model.candidate_layers, [])
                self.assertEqual(state.layer_diagnoses, [])
                self.assertEqual(state.recommended_focus, ["dram_bw"])
                saved = json.loads((Path(tmp) / "analysis/model_bottlenecks.json").read_text())
                self.assertEqual(saved["candidate_layers"], [])
                self.assertIn("decision_first", client.calls[0])
                self.assertNotIn("identify layers that are plausible global bottleneck candidates", client.calls[0])
                self.assertNotIn("Produce the final model-level bottleneck candidate list", client.calls[2])
                self.assertNotIn("take the compact candidate list from the model-level agent", client.calls[3])
                self.assertNotIn("Produce the final layer-level bottleneck state", client.calls[4])
                self.assertIn("SYSTEM-LEVEL ANALYSIS IS VALID", client.calls[2])
                self.assertIn("SYSTEM-LEVEL ANALYSIS IS VALID", client.calls[4])


if __name__ == "__main__":
    unittest.main()
