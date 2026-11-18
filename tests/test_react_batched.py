from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from typing import Any, Dict, List, Sequence

from chiplet_tuner.agents.react_graph import run_react_tool_graph
from chiplet_tuner.core.schemas import ToolResult, ToolSpec
from chiplet_tuner.llm.clients import LLMClient
from chiplet_tuner.llm.prompts import REACT_TOOL_USE_PROMPT
from chiplet_tuner.tools.analysis_tools import ToolContext


class PlannedLLM(LLMClient):
    def __init__(self, response: Dict[str, Any]) -> None:
        self.response = response
        self.calls: List[List[Dict[str, Any]]] = []

    def complete_json_messages(
        self,
        messages: Sequence[Dict[str, Any]],
    ) -> Dict[str, Any]:
        self.calls.append([dict(message) for message in messages])
        return self.response


class RecordingToolbox:
    def __init__(self, fail_tool: str | None = None) -> None:
        self.fail_tool = fail_tool
        self.calls: List[tuple[str, Dict[str, Any]]] = []

    def specs(self) -> List[ToolSpec]:
        return [
            ToolSpec(name="first", description="first tool", input_schema={}),
            ToolSpec(name="second", description="second tool", input_schema={}),
        ]

    def run(
        self,
        name: str,
        context: ToolContext,
        arguments: Dict[str, Any],
    ) -> ToolResult:
        self.calls.append((name, dict(arguments)))
        if name == self.fail_tool:
            raise ValueError(f"{name} failed")
        return ToolResult(name=name, payload={"arguments": arguments})


class BatchedReactExecutionTest(unittest.TestCase):
    def test_selects_once_and_executes_complete_plan_in_order(self) -> None:
        llm = PlannedLLM(
            {
                "thought": "Collect both observations.",
                "tools": [
                    {"name": "first", "arguments": {"value": 1}},
                    {"name": "second", "arguments": {"value": 2}},
                ],
            }
        )
        toolbox = RecordingToolbox()
        with tempfile.TemporaryDirectory() as tmp_dir:
            result = run_react_tool_graph(
                llm=llm,
                toolbox=toolbox,  # type: ignore[arg-type]
                system_prompt=f"test role\n\n{REACT_TOOL_USE_PROMPT}",
                task="react_tool_use",
                agent_name="test_agent",
                agent_goal="collect evidence",
                context_payload={"initial": True},
                context=ToolContext(output_dir=Path(tmp_dir)),
                allowed_tool_names=["first", "second"],
                execution_mode="batched",
                batch_max_tools=2,
            )

        self.assertEqual(len(llm.calls), 1)
        selection_system_prompt = llm.calls[0][0]["content"]
        self.assertIn("BATCHED TOOL-SELECTION ABLATION", selection_system_prompt)
        self.assertNotIn("request one useful tool observation at a time", selection_system_prompt)
        self.assertEqual(
            toolbox.calls,
            [("first", {"value": 1}), ("second", {"value": 2})],
        )
        self.assertEqual(list(result.tool_results), ["first", "second"])
        observation_payload = json.loads(result.messages[-1]["content"])
        self.assertEqual(observation_payload["task"], "batched_tool_observations")
        self.assertEqual(observation_payload["selected_tool_count"], 2)
        self.assertIn("Tool selection is closed", observation_payload["instruction"])

    def test_tool_error_becomes_observation_without_replanning(self) -> None:
        llm = PlannedLLM(
            {
                "thought": "Execute the fixed initial plan.",
                "tools": [
                    {"name": "first", "arguments": {}},
                    {"name": "second", "arguments": {}},
                ],
            }
        )
        toolbox = RecordingToolbox(fail_tool="first")
        with tempfile.TemporaryDirectory() as tmp_dir:
            result = run_react_tool_graph(
                llm=llm,
                toolbox=toolbox,  # type: ignore[arg-type]
                system_prompt="test role",
                task="react_tool_use",
                agent_name="test_agent",
                agent_goal="collect evidence",
                context_payload={},
                context=ToolContext(output_dir=Path(tmp_dir)),
                execution_mode="batched",
                batch_max_tools=2,
            )

        self.assertEqual(len(llm.calls), 1)
        self.assertEqual(len(toolbox.calls), 2)
        self.assertEqual(result.tool_results["first"].payload["status"], "error")
        self.assertEqual(result.transcript[0]["action"], "tool_error")
        self.assertEqual(result.transcript[1]["action"], "tool")


if __name__ == "__main__":
    unittest.main()
