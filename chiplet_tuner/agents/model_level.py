from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

from chiplet_tuner.agents.tool_use import (
    collect_generated_files,
    complete_json_continuing_messages,
)
from chiplet_tuner.agents.react_graph import run_react_tool_graph
from chiplet_tuner.agents.analysis_policy import (
    allows_system_analysis,
    analysis_output_prompt,
    analysis_role_prompt,
)
from chiplet_tuner.core.io import write_json
from chiplet_tuner.core.objective import metrics_with_objective
from chiplet_tuner.core.schemas import EvaluationResult, ModelAnalysisResult
from chiplet_tuner.llm.clients import LLMClient
from chiplet_tuner.llm.prompts import MODEL_LEVEL_ANALYSIS_PROMPT, MODEL_LEVEL_REACT_PROMPT, REACT_TOOL_USE_PROMPT
from chiplet_tuner.tools.analysis_tools import AnalysisToolbox, ToolContext


HARDWARE_EDIT_TOOLS = {"modify_hardware_parameter", "step_hardware_parameter"}


class ModelLevelAgent:
    """LLM-driven model-level agent with tool selection and execution."""

    def __init__(self, llm: LLMClient, toolbox: AnalysisToolbox) -> None:
        self.llm = llm
        self.toolbox = toolbox

    def analyze(
        self,
        evaluation: EvaluationResult,
        output_dir: Path,
        search_state: Dict[str, Any] | None = None,
        evaluation_bases: Dict[str, EvaluationResult] | None = None,
        simulator_schema: Dict[str, Any] | None = None,
    ) -> ModelAnalysisResult:
        active_base = self._initial_active_base(evaluation)
        context = ToolContext(
            output_dir=output_dir,
            evaluation=evaluation,
            current_hardware=evaluation.hardware,
            simulator_schema=simulator_schema,
            search_state=search_state,
            evaluation_bases=evaluation_bases or {"current": evaluation},
            active_base=active_base,
        )
        role_prompt = analysis_role_prompt("model_level", MODEL_LEVEL_REACT_PROMPT)
        react_result = run_react_tool_graph(
            llm=self.llm,
            toolbox=self.toolbox,
            system_prompt=f"{role_prompt}\n\n{REACT_TOOL_USE_PROMPT}",
            task="react_tool_use",
            agent_name="model_level",
            agent_goal=(
                "Identify an objective-relevant system tradeoff or layer bottleneck using measured evidence."
                if allows_system_analysis()
                else "Gather global evidence to identify model-level bottleneck layers."
            ),
            context_payload={
                "evaluation": self._evaluation_context(evaluation, simulator_schema),
                "search_state_available": search_state is not None,
                "hardware_context_available": simulator_schema is not None,
                "analysis_base_protocol": (
                    "Call compare_search_states and select_analysis_base before detailed evidence tools "
                    "when search_state_available is true. Compare current/previous/best first; call "
                    "compare_search_states again with include_history=true when trend evidence is needed. "
                    "The final selected base is used downstream."
                ),
            },
            context=context,
            max_steps=15,
            allowed_tool_names=self._analysis_tool_names(),
        )
        tool_results = react_result.tool_results
        react_trace = react_result.transcript
        analysis_evaluation = context.evaluation or evaluation
        analysis_base = context.active_base or active_base

        llm_result = complete_json_continuing_messages(
            self.llm,
            react_result.messages,
            {
                "task": "model_level_analysis",
                "final_decision_required": True,
                "instruction": (
                    "Tool use is complete. Continue from the previous messages and produce the "
                    "final model-level analysis. Use the real tool observations already present "
                    "in this conversation, not a separately summarized context."
                ),
                "output_prompt": analysis_output_prompt(MODEL_LEVEL_ANALYSIS_PROMPT),
                "active_analysis_base": analysis_base,
            },
            validate_response=lambda payload: self._validate_llm_result(payload, tool_results, analysis_evaluation),
        )
        candidates = self._normalize_candidate_layers(
            self._require_list(llm_result, "candidate_layers"),
            tool_results=tool_results,
            evaluation=analysis_evaluation,
        )
        summary = self._require_str(llm_result, "summary")
        analysis_summary = self._merge_analysis_summary(tool_results, analysis_evaluation, simulator_schema)
        analysis_summary["analysis_base"] = analysis_base
        generated_files = collect_generated_files(tool_results)
        artifact_refs_path = output_dir / "analysis" / "artifact_references.json"
        if artifact_refs_path.exists():
            generated_files["artifact_references_json"] = str(artifact_refs_path)
        analysis_summary_path = output_dir / "analysis" / "analysis_summary.json"
        write_json(analysis_summary_path, analysis_summary)
        generated_files["analysis_summary_json"] = str(analysis_summary_path)
        bottleneck_path = output_dir / "analysis" / "model_bottlenecks.json"
        write_json(
            bottleneck_path,
            {
                "selected_by": "model_level_agent",
                "analysis_base": analysis_base,
                "candidate_layers": candidates,
                "selection_notes": llm_result.get("notes", {}),
            },
        )
        generated_files["bottleneck_json"] = str(bottleneck_path)
        notes = llm_result.get("notes", {})
        notes = notes if isinstance(notes, dict) else {"raw_notes": notes}
        return ModelAnalysisResult(
            metrics=self._metrics_with_objective(analysis_evaluation.metrics, analysis_evaluation.hardware, simulator_schema),
            candidate_layers=candidates,
            generated_files=generated_files,
            summary=summary,
            global_findings=self._optional_string_list(notes.get("global_findings", [])),
            selected_views=self._optional_string_list(notes.get("selected_rank_views", notes.get("selected_views", []))),
            analysis_base=analysis_base,
            llm_notes={
                **notes,
                "analysis_base": analysis_base,
                "react_trace": react_trace,
                "react_backend": (
                    "langgraph" if react_result.execution_mode == "iterative" else "batched_tool_plan"
                ),
                "react_execution_mode": react_result.execution_mode,
                "react_batch_max_tools": react_result.max_selected_tools,
            },
        )

    def _initial_active_base(self, evaluation: EvaluationResult) -> Dict[str, Any]:
        iteration = None
        name = evaluation.run_dir.name
        if name.startswith("iter_"):
            try:
                iteration = int(name.split("_", 1)[1])
            except (IndexError, ValueError):
                iteration = None
        return {
            "source": "current",
            "iteration": iteration,
            "base_key": "current",
        }

    def _analysis_tool_names(self) -> List[str]:
        return sorted(spec.name for spec in self.toolbox.specs() if spec.name not in HARDWARE_EDIT_TOOLS)

    def _validate_llm_result(
        self,
        payload: Dict[str, Any],
        tool_results: Dict[str, Any],
        evaluation: EvaluationResult,
    ) -> None:
        self._require_str(payload, "summary")
        self._normalize_candidate_layers(
            self._require_list(payload, "candidate_layers"),
            tool_results=tool_results,
            evaluation=evaluation,
        )

    def _require_list(self, payload: Dict[str, Any], key: str) -> List[Any]:
        value = payload.get(key)
        if not isinstance(value, list):
            raise ValueError(f"Model-level LLM result must contain list field {key!r}")
        return value

    def _require_str(self, payload: Dict[str, Any], key: str) -> str:
        value = payload.get(key)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"Model-level LLM result must contain non-empty string field {key!r}")
        return value

    def _merge_analysis_summary(
        self,
        tool_results: Dict[str, ToolResult],
        evaluation: EvaluationResult,
        simulator_schema: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        summary: Dict[str, Any] = {
            "metrics": self._metrics_with_objective(evaluation.metrics, evaluation.hardware, simulator_schema)
        }
        payloads = [result.payload for result in tool_results.values()]
        for payload in payloads:
            if "timeline" in payload:
                summary["timeline"] = payload["timeline"]
            if "operator_groups" in payload:
                summary["operator_groups"] = payload["operator_groups"]
            if "aggregate_breakdown" in payload:
                summary["aggregate_breakdown"] = payload["aggregate_breakdown"]
            if "layer_rank_views" in payload:
                summary["layer_rank_views"] = payload["layer_rank_views"]
            if "rank_notes" in payload:
                summary["rank_notes"] = payload["rank_notes"]
            if "monetary_cost" in payload:
                summary["monetary_cost"] = payload["monetary_cost"]
            if "hardware" in payload:
                summary["hardware"] = payload["hardware"]
            if "objective" in payload:
                summary["objective"] = payload["objective"]
        return summary

    def _evaluation_context(
        self,
        evaluation: EvaluationResult,
        simulator_schema: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        return {
            "metrics": self._metrics_with_objective(evaluation.metrics, evaluation.hardware, simulator_schema),
        }

    def _metrics_with_objective(
        self,
        metrics: Dict[str, float],
        hardware: Dict[str, Any] | None = None,
        simulator_schema: Dict[str, Any] | None = None,
    ) -> Dict[str, float]:
        search_space = {}
        if isinstance(simulator_schema, dict) and isinstance(simulator_schema.get("search_space"), dict):
            search_space = simulator_schema["search_space"]
        return metrics_with_objective(metrics, hardware=hardware, search_space=search_space)

    def _optional_string_list(self, value: Any) -> List[str]:
        if not isinstance(value, list):
            return []
        return [str(item) for item in value if isinstance(item, str) and item.strip()]

    def _normalize_candidate_layers(
        self,
        raw_layers: List[Any],
        tool_results: Dict[str, Any],
        evaluation: EvaluationResult,
    ) -> List[Dict[str, Any]]:
        if getattr(self.toolbox, "evidence_mode", "full") == "scalar":
            if raw_layers:
                raise ValueError("Scalar evidence cannot support measured candidate layers; use candidate_layers=[].")
            return []
        metrics = self._metrics_with_objective(evaluation.metrics, evaluation.hardware)
        candidates = []
        for item in raw_layers:
            if not isinstance(item, dict):
                continue
            latency_sum = self._optional_float(item.get("latency_sum"))
            energy = self._optional_float(item.get("energy"))
            evidence = item.get("evidence", [])
            if not isinstance(evidence, list):
                evidence = [str(evidence)]
            if latency_sum is not None:
                evidence.append(f"latency_sum={latency_sum:.4g}")
                evidence.append(f"latency_share={latency_sum / max(float(metrics['latency']), 1.0):.4g}")
            if energy is not None:
                evidence.append(f"energy={energy:.4g}")
                evidence.append(f"energy_share={energy / max(float(metrics['energy']), 1.0):.4g}")
            candidates.append(
                {
                    "layer_id": item.get("layer_id", item.get("layerID")),
                    "layer_name": item.get("layer_name", item.get("layerName")),
                    "operator_group": item.get("operator_group", item.get("group", "unknown")),
                    "concern_types": self._normalize_concern_types(item),
                    "rank_metric": item.get("rank_metric"),
                    "rank_value": item.get("rank_value"),
                    "evidence": [str(entry) for entry in evidence if str(entry).strip()],
                    "confidence": str(item.get("confidence", "medium")),
                }
            )
        if not candidates and not (allows_system_analysis() and raw_layers == []):
            raise ValueError("Model-level LLM result produced no valid candidate_layers.")
        return candidates

    def _normalize_concern_types(self, item: Dict[str, Any]) -> List[str]:
        raw = item.get("concern_types", item.get("impact_types"))
        if isinstance(raw, str):
            raw = [raw]
        if isinstance(raw, list):
            values = [str(value) for value in raw if str(value) in {"latency", "energy", "monetary_cost", "mixed"}]
            if values:
                return values
        rank_metric = str(item.get("rank_metric", ""))
        if rank_metric in {"latency_sum", "critical_end", "compute", "memory", "communication", "buffer"}:
            return ["latency"]
        if rank_metric == "energy":
            return ["energy"]
        return ["mixed"]

    def _optional_float(self, value: Any) -> float | None:
        try:
            return float(value)
        except (TypeError, ValueError):
            return None
