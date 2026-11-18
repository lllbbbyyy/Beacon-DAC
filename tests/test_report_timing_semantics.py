"""Show what aggregate timing fields do and do not encode."""
import unittest
from pathlib import Path
from unittest.mock import patch

from chiplet_tuner.tools.analysis_tools import AnalysisToolbox, ToolContext
from chiplet_tuner.core.report_semantics import TIMING_FIELD_DEFINITIONS, TIMING_REPORT_TOOLS
from chiplet_tuner.core.schemas import ToolResult
from chiplet_tuner.agents.tool_use import serialize_tool_results


class TimingSemanticsTest(unittest.TestCase):
    def test_public_tool_results_keep_definitions_in_llm_handoff(self):
        toolbox = AnalysisToolbox()
        for name in TIMING_REPORT_TOOLS:
            original = {"cached_report": True}
            with patch.object(toolbox, "_run", return_value=ToolResult(name, original, {})):
                result = toolbox.run(name, ToolContext(output_dir=Path("unused")))
            self.assertEqual(result.payload["field_definitions"], TIMING_FIELD_DEFINITIONS)
            self.assertNotIn("field_definitions", original)
            handoff = serialize_tool_results({name: result})
            self.assertEqual(handoff[name]["payload"]["field_definitions"], TIMING_FIELD_DEFINITIONS)

    def test_sum_over_max_does_not_identify_serial_execution(self):
        toolbox = AnalysisToolbox()
        def entry(begin, end, batch):
            return dict(layerID=1, layerName="ffn", latencyBegin=begin,
                        latencyEnd=end, batchID=batch, calcTime=end - begin)
        serial = {"core0": [entry(0, 10, 0), entry(10, 20, 1)]}
        parallel = {"core0": [entry(0, 10, 0)], "core1": [entry(0, 10, 1)]}
        a = toolbox._aggregate_layers(serial, {})[(1, "ffn")]
        b = toolbox._aggregate_layers(parallel, {})[(1, "ffn")]
        self.assertEqual(a.latency_sum / a.latency_max, 2)
        self.assertEqual(b.latency_sum / b.latency_max, 2)
        self.assertEqual(a.critical_end, 20)
        self.assertEqual(b.critical_end, 10)

    def test_duration_over_makespan_is_not_a_partition_share(self):
        toolbox = AnalysisToolbox()
        record = toolbox._compact_layer_record(
            {"layer_id": 1, "layer_name": "ffn", "latency_sum": 20, "energy": 3},
            metrics={"latency": 10, "energy": 6}, include_fields={"shares"},
        )
        self.assertEqual(record["shares"]["latency_sum_share"], 2)
        self.assertEqual(record["shares"]["energy_share"], .5)


if __name__ == "__main__":
    unittest.main()
