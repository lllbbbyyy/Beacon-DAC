import unittest

from chiplet_tuner.tools.analysis_tools import AnalysisToolbox


class CostFieldMeaningsTest(unittest.TestCase):
    def test_component_meanings_survive_top_component_filter(self):
        toolbox = AnalysisToolbox()
        detail = {"cost_compute": 50, "cost_IO": 20, "cost_os_overall": 30, "cost_overall": 100}
        summary = toolbox._summarize_monetary_cost(detail, {"mc": 100})
        payload = toolbox._monetary_cost_payload_for_llm(summary, {"top_components": 2})
        components = {item["component"]: item for item in payload["top_components"]}
        self.assertEqual(payload["total"], 100)
        self.assertEqual(components["cost_compute"]["value"], 50)
        self.assertIn("SRAM/GLB", components["cost_compute"]["meaning"])
        self.assertIn("not output-stationary", components["cost_os_overall"]["meaning"])
        self.assertEqual(components["cost_os_overall"]["share"], 0.3)


if __name__ == "__main__":
    unittest.main()
