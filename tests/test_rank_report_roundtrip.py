import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from chiplet_tuner.tools.analysis_tools import AnalysisToolbox, ToolContext
from chiplet_tuner.agents.tool_use import serialize_tool_results


class RankReportRoundtripTests(unittest.TestCase):
    def setUp(self):
        self.toolbox = AnalysisToolbox()
        self.layer = dict(layer_id=1, layer_name="ffn", group="ffn", latency_sum=100,
                          latency_max=60, critical_end=120, energy=200, occurrences=2,
                          cores=["core0"], batches=[0, 1], dominant_dimension="compute",
                          time_breakdown={"compute": .8}, energy_breakdown={"memory": .7})

    def test_compaction_is_idempotent_and_does_not_alias_input(self):
        once = self.toolbox._build_layer_rank_views([self.layer], 1)
        twice = self.toolbox._compact_rank_views(once)
        self.assertEqual(once, twice)
        self.assertEqual(twice, self.toolbox._compact_rank_views(twice))
        twice["energy"][0]["energy"]["total"] = 0
        self.assertEqual(once["energy"][0]["energy"]["total"], 200)

    def test_cold_cache_warm_cache_and_llm_have_same_numbers(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            evaluation = SimpleNamespace(metrics={}, hardware={})
            with patch.object(self.toolbox, "_require_evaluation", return_value=evaluation), \
                 patch.object(self.toolbox, "_search_space", return_value={}), \
                 patch.object(self.toolbox, "_analysis_dir", return_value=root), \
                 patch.object(self.toolbox, "_metrics_with_objective", return_value={}), \
                 patch.object(self.toolbox, "_record_artifact_reference"), \
                 patch.object(self.toolbox, "_summarize_breakdowns", return_value={}), \
                 patch.object(self.toolbox, "_layer_loads", return_value=[self.layer]) as loads:
                context = ToolContext(output_dir=root)
                cold = self.toolbox.summarize_layer_rank_views(context, top_layers=1)
                warm = self.toolbox.summarize_layer_rank_views(context, top_layers=1)
                self.assertEqual(loads.call_count, 1)
                self.assertEqual(cold.payload, warm.payload)
                sent = serialize_tool_results({"rank": warm})["rank"]["payload"]
                row = sent["layer_rank_views"]["latency_sum"][0]
                self.assertEqual(row["timing"]["latency_sum"], 100)
                self.assertEqual(row["energy"]["total"], 200)
                self.assertEqual(row["operator_group"], "ffn")
                self.assertEqual(row["placement"]["core_count"], 1)
                self.assertEqual(row["placement"]["batch_count"], 2)
                # Legacy caches must be rebuilt from raw evaluation records.
                path = root / "layer_rank_views_top1.json"
                legacy = json.loads(path.read_text())
                legacy.pop("report_schema_version")
                path.write_text(json.dumps(legacy))
                rebuilt = self.toolbox.summarize_layer_rank_views(context, top_layers=1)
                self.assertEqual(loads.call_count, 2)
                self.assertEqual(rebuilt.payload, cold.payload)


if __name__ == "__main__":
    unittest.main()
