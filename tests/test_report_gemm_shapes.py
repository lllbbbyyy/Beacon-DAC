import unittest

from chiplet_tuner.tools.analysis_tools import AnalysisToolbox
from chiplet_tuner.core.schemas import ToolResult
from chiplet_tuner.agents.tool_use import serialize_tool_results


class GemmReportTests(unittest.TestCase):
    def test_distinct_micro_batch_shapes_survive_compaction(self):
        entries = []
        for batch in range(10):
            shape = dict(layer_id=7, M=2079 if batch == 0 else 32, N=1792, K=5120,
                         rhs="parameter", scope="execution_gemm_before_buffer_tiling")
            entries.append(dict(layerID=7, layerName="ffn", batchID=batch,
                                latencyBegin=batch*10, latencyEnd=batch*10+10,
                                gemm_shapes=[shape]))
        toolbox = AnalysisToolbox()
        layer = toolbox._aggregate_layers({"core0": entries}, {})[(7, "ffn")]
        shapes = layer.features["gemm_shapes"]
        self.assertEqual([s["M"] for s in shapes], [2079, 32])
        self.assertEqual([s["occurrences"] for s in shapes], [1, 9])
        self.assertEqual(shapes[1]["example_batch_ids"], [1, 2, 3, 4])
        compact = toolbox._compact_layer_record(
            {"features": layer.features}, metrics={}, include_fields={"features"})
        self.assertEqual(compact["features"]["gemm_shapes"], shapes)
        result = ToolResult("inspect_layer_details", {"layers": [{"layer_id": 7, **compact}]}, {})
        handoff = serialize_tool_results({"details": result})
        self.assertEqual(handoff["details"]["payload"]["layers"][0]["features"]["gemm_shapes"], shapes)

    def test_legacy_reports_do_not_invent_shapes(self):
        layer = AnalysisToolbox()._aggregate_layers(
            {"core0": [dict(layerID=1, layerName="ffn", batchID=0)]}, {})[(1, "ffn")]
        self.assertNotIn("gemm_shapes", layer.features)


if __name__ == "__main__":
    unittest.main()
