from pathlib import Path
import tempfile
import unittest
import json
import os
from unittest.mock import Mock, patch

from baseline.agentic_controller import AgenticController
from chiplet_tuner.core.report_semantics import TIMING_FIELD_DEFINITIONS
from baseline.agentic_region import PhaseRegion
from baseline.config_encoding import FullDesignCodec
from baseline.search import SearchRecord
from chiplet_tuner.core.search_space import make_hardware_search_space
from chiplet_tuner.simulators.compass_config import build_compass_initial_hardware


class ScriptedClient:
    def __init__(self, replies):
        self.replies = iter(replies)
        self.payloads = []

    def complete_json(self, prompt, payload):
        self.payloads.append(payload)
        return next(self.replies)


class ControllerTests(unittest.TestCase):
    def test_objective_guided_coverage_is_available_to_all_roles(self):
        client = ScriptedClient([
            {"directive": "inspect"}, {"analysis": "test an unmeasured direction"},
            {"approved": True, "feedback": "proceed"},
            {"base": "incumbent", "selections": {"num_chiplets": [2]}},
        ])
        with tempfile.TemporaryDirectory() as tmp, patch.dict(
                os.environ, {"CHIPLET_TUNER_ANALYSIS_POLICY": "objective_guided"}):
            record = SearchRecord(0, None, None, self.hardware, {"objective": 1}, Path(tmp), 1)
            AgenticController(client, Path(tmp), "test").select_region(
                0, self.region.domains, self.region.snapshot(), [record])
        for payload in client.payloads:
            self.assertIn('parameter_coverage', payload)
            for key, value in TIMING_FIELD_DEFINITIONS.items():
                self.assertEqual(payload['report_field_definitions'][key], value)
            self.assertEqual(payload['parameter_coverage']['parameters']['num_chiplets']['base_value'],
                             self.hardware['num_chiplets'])

    def test_structured_analysis_is_reviewed_without_execution(self):
        evidence = {"evidence": ["measured cost"], "hypotheses": ["test bandwidth"]}
        client = ScriptedClient([
            {"directive": "inspect"}, {"analysis": evidence},
            {"approved": True, "feedback": "proceed"},
            {"base": "incumbent", "selections": {"num_chiplets": [2]}},
        ])
        with tempfile.TemporaryDirectory() as tmp:
            record = SearchRecord(0, None, None, self.hardware, {"objective": 1}, Path(tmp), 1)
            AgenticController(client, Path(tmp), "test").select_region(
                0, self.region.domains, self.region.snapshot(), [record])
            self.assertEqual(client.payloads[2]["analysis"], {"analysis": evidence})
            self.assertEqual(client.payloads[3]["analysis"], {"analysis": evidence})
        for invalid in (None, {}, [], " ", 1):
            self.assertFalse(AgenticController._has_analysis({"analysis": invalid}))

    def test_json_parse_retry_is_bounded_and_explicit(self):
        client = ScriptedClient([])
        error = json.JSONDecodeError("No JSON", "invalid", 0)
        client.complete_json = Mock(side_effect=[error, {"directive": "inspect"}])
        with tempfile.TemporaryDirectory() as tmp:
            controller = AgenticController(client, Path(tmp), "test")
            self.assertEqual(controller._call("Orchestrator", "Return JSON", {}),
                             {"directive": "inspect"})
            self.assertIn("json_correction", client.complete_json.call_args.args[1])
            client.complete_json = Mock(side_effect=error)
            with self.assertRaises(json.JSONDecodeError):
                controller._call("Orchestrator", "Return JSON", {})
            self.assertEqual(client.complete_json.call_count, 4)

    def setUp(self):
        self.space = make_hardware_search_space("mixed", "64", "gpt3_merged", batch_size=128)
        self.hardware = build_compass_initial_hardware("mixed", "64")
        self.region = PhaseRegion(FullDesignCodec(self.hardware, self.space))

    def test_three_roles_report_query_and_review(self):
        client = ScriptedClient([
            {"directive": "inspect memory energy"},
            {"query": {"iteration": 0, "report": "energy", "offset": 0}},
            {"analysis": "Compare nearby bandwidth candidates"},
            {"approved": True, "feedback": "proceed"},
            {"base": "incumbent", "selections": {"num_chiplets": [2]}},
        ])
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "search_energy_detail.json").write_text('{"dram": 42}')
            record = SearchRecord(0, None, None, self.hardware, {"objective": 1.0}, root, 1)
            controller = AgenticController(client, root / "agentic", "test-only")
            result = controller.select_region(0, self.region.domains, self.region.snapshot(), [record])
            self.region.update(result)
            self.assertEqual(self.region.choices["chiplets.0.compute_units"],
                             [self.hardware["chiplets"][0]["compute_units"]])
            self.assertEqual(self.region.choices["micro_batch"], [self.hardware["micro_batch"]])
            self.assertEqual([p["agent_name"] for p in client.payloads],
                             ["Orchestrator", "ArchitectureAnalyst", "ArchitectureAnalyst",
                              "Orchestrator", "OptimizationEngineer"])
            evidence = client.payloads[2]["observations"][0]["result"]
            self.assertIn('"dram": 42', evidence["text"])
            self.assertTrue((root / "agentic/phase_000.json").is_file())

    def test_rejected_analysis_is_not_executed(self):
        client = ScriptedClient([{"directive": "inspect"}, {"analysis": "unsupported"},
                                 {"approved": False, "feedback": "missing evidence"}])
        with tempfile.TemporaryDirectory() as tmp:
            record = SearchRecord(0, None, None, self.hardware, {"objective": 1}, Path(tmp), 1)
            controller = AgenticController(client, Path(tmp), "test", review_attempts=1)
            with self.assertRaisesRegex(ValueError, "rejected"):
                controller.select_region(0, self.region.domains, self.region.snapshot(), [record])
            self.assertNotIn("OptimizationEngineer", [p["agent_name"] for p in client.payloads])

    def test_invalid_engineer_field_is_returned_for_correction(self):
        value = self.hardware["chiplets"][0]["compute_units"]
        client = ScriptedClient([
            {"directive": "inspect"}, {"analysis": "test compute size"},
            {"approved": True, "feedback": "proceed"},
            {"base": "previous", "selections": {"compute_units": [value]}},
            {"base": "previous", "selections": {"chiplets.0.compute_units": [value]}},
        ])
        with tempfile.TemporaryDirectory() as tmp:
            record = SearchRecord(0, None, None, self.hardware, {"objective": 1}, Path(tmp), 1)
            result = AgenticController(client, Path(tmp), "test").select_region(
                0, self.region.domains, self.region.snapshot(), [record])
            self.assertEqual(result, {"chiplets.0.compute_units": [value]})
            feedback = client.payloads[-1]["validation_feedback"]
            self.assertIn("Unknown design field", feedback[0]["error"])
            self.assertEqual(self.region.snapshot(), self.region.domains)

    def test_query_rejects_arbitrary_paths_and_unknown_ids(self):
        record = SearchRecord(0, None, None, self.hardware, {"objective": 1}, Path("/tmp"), 1)
        for query in ({"iteration": 0, "report": "../../key.json"},
                      {"iteration": 1, "report": "hardware"},
                      {"iteration": False, "report": "hardware"},
                      {"iteration": 0, "report": "energy", "offset": -1}):
            with self.assertRaises(ValueError):
                AgenticController._query(query, [record])

    def test_analyst_missing_outer_key_is_corrected(self):
        client = ScriptedClient([
            {"directive": "inspect"},
            {"iteration": 0, "report": "hardware", "offset": 0},
            {"query": {"iteration": 0, "report": "hardware"}},
            {"analysis": "use observed hardware"},
            {"approved": True, "feedback": "proceed"},
            {"base": "incumbent", "selections": {"num_chiplets": [2]}},
        ])
        with tempfile.TemporaryDirectory() as tmp:
            record = SearchRecord(0, None, None, self.hardware, {"objective": 1}, Path(tmp), 1)
            AgenticController(client, Path(tmp), "test", max_queries=1).select_region(
                0, self.region.domains, self.region.snapshot(), [record])
            self.assertTrue(client.payloads[2]["validation_feedback"])
            self.assertEqual(client.payloads[2]["queries_remaining"], 1)
            self.assertEqual(client.payloads[3]["queries_remaining"], 0)

    def test_analyst_format_retry_is_bounded(self):
        client = ScriptedClient([{"directive": "inspect"}, {}, {}])
        with tempfile.TemporaryDirectory() as tmp:
            record = SearchRecord(0, None, None, self.hardware, {"objective": 1}, Path(tmp), 1)
            with self.assertRaisesRegex(ValueError, "bounded format correction"):
                AgenticController(client, Path(tmp), "test").select_region(
                    0, self.region.domains, self.region.snapshot(), [record])


if __name__ == "__main__":
    unittest.main()
