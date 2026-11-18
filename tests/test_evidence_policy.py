from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from chiplet_tuner.agents.layer_level import LayerLevelAgent
from chiplet_tuner.agents.model_level import ModelLevelAgent
from chiplet_tuner.core.schemas import EvaluationResult
from chiplet_tuner.llm.clients import MockLLMClient
from chiplet_tuner.tools.analysis_tools import AnalysisToolbox, ToolContext
from chiplet_tuner.tools.evidence_policy import (
    SCALAR_TOOLS, ScalarEvidenceToolbox, configure_evidence_policy,
)


class EvidencePolicyTest(unittest.TestCase):
    def evaluation(self, root):
        return EvaluationResult(
            run_dir=root / "iter_000", hardware={},
            metrics={"latency": 2., "energy": 3., "mc": 4.},
            detail_files={"latency": root / "forbidden_report.json"},
        )

    def test_report_calls_are_hidden_and_denied_before_file_access(self):
        toolbox = ScalarEvidenceToolbox()
        self.assertEqual({spec.name for spec in toolbox.specs()}, SCALAR_TOOLS)
        with tempfile.TemporaryDirectory() as temp:
            evaluation = self.evaluation(Path(temp))
            context = ToolContext(output_dir=Path(temp), evaluation=evaluation)
            with patch.object(Path, "read_text", side_effect=AssertionError("report read")):
                for name in ("inspect_layer_details", "build_execution_timeline", "summarize_monetary_cost",
                             "summarize_layer_rank_views", "aggregate_layer_loads", "summarize_operator_groups"):
                    with self.subTest(name=name), self.assertRaisesRegex(ValueError, "scalar evidence"):
                        toolbox.run(name, context)
                result = toolbox.run("summarize_metrics", context)
        self.assertEqual(result.payload["objective"], 24.)

    def test_scalar_allows_no_layer_findings_but_full_keeps_original_validation(self):
        with tempfile.TemporaryDirectory() as temp:
            evaluation = self.evaluation(Path(temp))
            agent = ModelLevelAgent(MockLLMClient(), ScalarEvidenceToolbox())
            self.assertEqual(agent._normalize_candidate_layers([], {}, evaluation), [])
            with self.assertRaisesRegex(ValueError, "candidate layers"):
                agent._normalize_candidate_layers([{"layer_id": 1}], {}, evaluation)
            original = ModelLevelAgent(MockLLMClient(), AnalysisToolbox())
            with self.assertRaisesRegex(ValueError, "no valid candidate_layers"):
                original._normalize_candidate_layers([], {}, evaluation)

    def test_scalar_layer_diagnosis_does_not_fabricate_a_root_cause(self):
        agent = LayerLevelAgent(MockLLMClient(), ScalarEvidenceToolbox())
        payload = {
            "primary_impact": "latency", "dominant_root_cause": "unknown",
            "layer_diagnoses": [], "retrieval_description": "aggregate latency regressed",
            "root_cause_summary": {"compute": 0., "memory": 0., "communication": 0., "buffer": 0.},
            "recommended_focus": ["micro_batch"],
        }
        self.assertEqual(agent._parse_state(payload).layer_diagnoses, [])
        with self.assertRaisesRegex(ValueError, "root_cause"):
            agent._parse_state(dict(payload, dominant_root_cause="memory"))
        with self.assertRaisesRegex(ValueError, "ratios"):
            agent._parse_state(dict(payload, root_cause_summary={"memory": 1.}))

    def test_history_provenance_and_resume_mode_cannot_be_changed(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            db = root / "history_vector_store.sqlite"
            configure_evidence_policy("scalar", root, db)
            db.touch()
            self.assertIsInstance(configure_evidence_policy("scalar", root, db), ScalarEvidenceToolbox)
            self.assertEqual(json.loads((root / "information_policy.json").read_text())["evidence_mode"], "scalar")
            with self.assertRaisesRegex(ValueError, "Cannot change"):
                configure_evidence_policy("full", root, db)
            with self.assertRaisesRegex(ValueError, "cannot import"):
                configure_evidence_policy("scalar", root, db, import_history="full.jsonl")
            with self.assertRaisesRegex(ValueError, "output-local"):
                configure_evidence_policy("scalar", root, root / "other.sqlite")
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            db = root / "history_vector_store.sqlite"
            db.touch()
            with self.assertRaisesRegex(ValueError, "no scalar evidence provenance"):
                configure_evidence_policy("scalar", root, db)


if __name__ == "__main__":
    unittest.main()
