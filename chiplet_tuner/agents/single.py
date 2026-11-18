from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from chiplet_tuner.agents.react_graph import run_react_tool_graph
from chiplet_tuner.agents.solution import SolutionGenerationAgent
from chiplet_tuner.agents.tool_use import (
    collect_generated_files,
    complete_json_continuing_messages,
    serialize_tool_results,
)
from chiplet_tuner.core.io import write_json
from chiplet_tuner.core.objective import metrics_with_objective
from chiplet_tuner.core.schemas import BottleneckState, EvaluationResult, ModelAnalysisResult, SolutionProposal
from chiplet_tuner.llm.clients import LLMClient
from chiplet_tuner.llm.prompts import (
    REACT_TOOL_USE_PROMPT,
    SINGLE_AGENT_ANALYSIS_PROMPT,
    SINGLE_AGENT_REACT_PROMPT,
    SINGLE_AGENT_SOLUTION_PROMPT,
)
from chiplet_tuner.rag.vector_store import HistoryVectorStore, hardware_fingerprint, summarize_cases
from chiplet_tuner.tools.analysis_tools import AnalysisToolbox, ToolContext


class SingleAgent(SolutionGenerationAgent):
    """Monolithic agent for ablation against the hierarchical multi-agent pipeline.

    The class intentionally reuses SolutionGenerationAgent's hardware parsing,
    normalization, duplicate repair, and diff logic so the ablation changes only
    the reasoning organization, not the candidate materialization path.
    """

    def __init__(
        self,
        llm: LLMClient,
        store: HistoryVectorStore,
        toolbox: AnalysisToolbox,
        simulator_schema: Dict[str, Any],
        top_k: int = 5,
    ) -> None:
        super().__init__(
            llm=llm,
            store=store,
            toolbox=toolbox,
            simulator_schema=simulator_schema,
            top_k=top_k,
        )

    def analyze_and_propose(
        self,
        evaluation: EvaluationResult,
        output_dir: Path,
        forbidden_hardware_fingerprints: Optional[Set[str]] = None,
        search_state: Optional[Dict[str, Any]] = None,
        evaluation_bases: Optional[Dict[str, EvaluationResult]] = None,
    ) -> Tuple[ModelAnalysisResult, BottleneckState, SolutionProposal, EvaluationResult]:
        forbidden_hardware_fingerprints = set(forbidden_hardware_fingerprints or set())
        evaluation_bases = dict(evaluation_bases or {})
        evaluation_bases.setdefault("current", evaluation)
        active_base = self._initial_active_base(evaluation)
        context = ToolContext(
            output_dir=output_dir,
            evaluation=evaluation,
            current_hardware=evaluation.hardware,
            simulator_schema=self.simulator_schema,
            search_state=search_state,
            evaluation_bases=evaluation_bases,
            active_base=active_base,
        )
        react_result = run_react_tool_graph(
            llm=self.llm,
            toolbox=self.toolbox,
            system_prompt=f"{SINGLE_AGENT_REACT_PROMPT}\n\n{REACT_TOOL_USE_PROMPT}",
            task="react_tool_use",
            agent_name="single_agent",
            agent_goal=(
                "Gather enough global, layer, and hardware evidence to diagnose the current design. "
                "Do not modify hardware during tool use; the final solution JSON will provide "
                "the hardware-edit tool calls that the framework applies."
            ),
            context_payload={
                "evaluation": self._evaluation_context(evaluation),
                "current_hardware": evaluation.hardware,
                "search_space": self._search_space(),
                "forbidden_hardware_fingerprints": sorted(forbidden_hardware_fingerprints),
                "search_state_available": search_state is not None,
                "analysis_base_protocol": (
                    "Call compare_search_states and select_analysis_base before detailed evidence tools "
                    "when search_state_available is true. The final selected base defines the proposal context."
                ),
            },
            context=context,
            max_steps=20,
            allowed_tool_names=self._solution_analysis_tool_names(),
        )
        tool_results = dict(react_result.tool_results)
        react_trace = list(react_result.transcript)
        react_messages = list(react_result.messages)

        analysis_result = complete_json_continuing_messages(
            self.llm,
            react_messages,
            {
                "task": "single_agent_analysis",
                "final_decision_required": True,
                "agent_name": "single_agent",
                "instruction": (
                    "Tool use is complete. Produce the single-agent diagnosis. Use the real tool "
                    "observations already present in this conversation."
                ),
                "output_prompt": SINGLE_AGENT_ANALYSIS_PROMPT,
                "active_analysis_base": context.active_base or active_base,
            },
            validate_response=self._validate_single_analysis,
        )
        model_result = self._model_result_from_analysis(
            analysis_result,
            evaluation=evaluation,
            tool_results=tool_results,
            output_dir=output_dir,
            fallback_base=context.active_base or active_base,
            react_trace=react_trace,
        )
        self._record_react_execution(model_result.llm_notes, react_result)
        state = self._bottleneck_state_from_analysis(
            analysis_result,
            fallback_base=model_result.analysis_base,
            tool_results=tool_results,
            react_trace=react_trace,
        )
        self._record_react_execution(state.llm_notes, react_result)
        analysis_evaluation = self._evaluation_for_base_selection(
            state.llm_notes.get("analysis_base", model_result.analysis_base),
            evaluation_bases,
            default=evaluation,
        )

        retrieved_cases = summarize_cases(
            self.store.search(
                state.retrieval_description,
                analysis_evaluation.hardware,
                top_k=self.top_k,
                bottleneck_state=self._state_message(state),
            )
        )
        solution_result = complete_json_continuing_messages(
            self.llm,
            react_messages,
            {
                "task": "single_agent_solution_generation",
                "final_decision_required": True,
                "agent_name": "single_agent",
                "instruction": (
                    "Use the single-agent diagnosis, retrieved cases, search space, proposal base hardware, "
                    "forbidden fingerprints, and prior tool observations to produce one legal hardware-edit "
                    "tool_calls list. The framework will execute these calls sequentially on "
                    "proposal_base_hardware after this response."
                ),
                "output_prompt": SINGLE_AGENT_SOLUTION_PROMPT,
                "single_agent_diagnosis": {
                    "model_analysis": self._model_message(model_result),
                    "bottleneck_state": self._state_message(state),
                },
                "current_hardware": analysis_evaluation.hardware,
                "proposal_base": state.llm_notes.get("analysis_base", model_result.analysis_base),
                "proposal_base_hardware_summary": self._proposal_base_summary(analysis_evaluation.hardware),
                "allowed_final_tools": sorted({"modify_hardware_parameter", "step_hardware_parameter"}),
                "retrieved_cases": retrieved_cases,
                "search_space": self._search_space(),
                "forbidden_hardware_fingerprints": sorted(forbidden_hardware_fingerprints),
            },
            validate_response=lambda payload: self._validate_solution_result(
                payload,
                context=context,
                base_hardware=analysis_evaluation.hardware,
            ),
        )
        proposal = self._proposal_from_solution_result(
            solution_result,
            tool_results=tool_results,
            context=context,
            state=state,
            hardware=analysis_evaluation.hardware,
            retrieved_cases=retrieved_cases,
            forbidden_hardware_fingerprints=forbidden_hardware_fingerprints,
            proposal_base=model_result.analysis_base,
            react_trace=react_trace,
        )
        self._record_react_execution(proposal.llm_notes, react_result)
        return model_result, state, proposal, analysis_evaluation

    @staticmethod
    def _record_react_execution(notes: Dict[str, Any], react_result: Any) -> None:
        notes["react_backend"] = (
            "langgraph" if react_result.execution_mode == "iterative" else "batched_tool_plan"
        )
        notes["react_execution_mode"] = react_result.execution_mode
        notes["react_batch_max_tools"] = react_result.max_selected_tools

    def _validate_single_analysis(self, payload: Dict[str, Any]) -> None:
        model = self._require_nested_dict(payload, "model_analysis")
        bottleneck = self._require_nested_dict(payload, "bottleneck_state")
        self._require_str(model, "summary")
        self._require_nonempty_list(model, "candidate_layers")
        self._parse_bottleneck_state(bottleneck, fallback_base={})

    def _model_result_from_analysis(
        self,
        payload: Dict[str, Any],
        evaluation: EvaluationResult,
        tool_results: Dict[str, Any],
        output_dir: Path,
        fallback_base: Dict[str, Any],
        react_trace: List[Dict[str, Any]],
    ) -> ModelAnalysisResult:
        model = self._require_nested_dict(payload, "model_analysis")
        analysis_base = model.get("analysis_base")
        if not isinstance(analysis_base, dict):
            analysis_base = fallback_base
        candidates = [self._normalize_candidate_layer(item) for item in self._require_nonempty_list(model, "candidate_layers")]
        notes = model.get("notes", {})
        notes = notes if isinstance(notes, dict) else {"raw_notes": notes}
        generated_files = collect_generated_files(tool_results)
        analysis_summary_path = output_dir / "analysis" / "single_agent_analysis_summary.json"
        write_json(
            analysis_summary_path,
            {
                "agent_mode": "single",
                "analysis_base": analysis_base,
                "metrics": self._metrics_with_objective(evaluation.metrics, evaluation.hardware),
                "tool_results": serialize_tool_results(tool_results),
            },
        )
        generated_files["single_agent_analysis_summary_json"] = str(analysis_summary_path)
        return ModelAnalysisResult(
            metrics=self._metrics_with_objective(evaluation.metrics, evaluation.hardware),
            candidate_layers=candidates,
            generated_files=generated_files,
            summary=self._require_str(model, "summary"),
            global_findings=self._optional_string_list(notes.get("global_findings", [])),
            selected_views=self._optional_string_list(notes.get("selected_rank_views", notes.get("selected_views", []))),
            analysis_base=analysis_base,
            llm_notes={
                **notes,
                "agent_mode": "single",
                "analysis_base": analysis_base,
                "react_trace": react_trace,
                "react_backend": "langgraph",
            },
        )

    def _bottleneck_state_from_analysis(
        self,
        payload: Dict[str, Any],
        fallback_base: Dict[str, Any],
        tool_results: Dict[str, Any],
        react_trace: List[Dict[str, Any]],
    ) -> BottleneckState:
        bottleneck = self._require_nested_dict(payload, "bottleneck_state")
        state = self._parse_bottleneck_state(bottleneck, fallback_base=fallback_base)
        notes = dict(state.llm_notes)
        notes["tool_results"] = serialize_tool_results(tool_results)
        notes["generated_files"] = collect_generated_files(tool_results)
        notes["react_trace"] = react_trace
        notes["react_backend"] = "langgraph"
        notes["agent_mode"] = "single"
        state.llm_notes = notes
        return state

    def _proposal_from_solution_result(
        self,
        result: Dict[str, Any],
        tool_results: Dict[str, Any],
        context: ToolContext,
        state: BottleneckState,
        hardware: Dict[str, Any],
        retrieved_cases: List[Dict[str, Any]],
        forbidden_hardware_fingerprints: Set[str],
        proposal_base: Dict[str, Any],
        react_trace: List[Dict[str, Any]],
    ) -> SolutionProposal:
        strategy = self._require_str(result, "strategy")
        final_tool_calls = self._require_final_tool_calls(result)
        update_results = self._apply_final_tool_calls(
            context=context,
            base_hardware=hardware,
            tool_calls=final_tool_calls,
        )
        for index, update_result in enumerate(update_results):
            tool_results[f"final_hardware_tool_call_{index:02d}_{update_result.name}"] = update_result
        proposed_hardware = copy.deepcopy(context.current_hardware or hardware)
        actions: List[str] = []
        rationale = self._require_str(result, "rationale", allow_empty=True)
        validation_notes: List[str] = []
        for update_result in update_results:
            validation_notes.extend(
                str(item) for item in update_result.payload.get("validation", []) if isinstance(item, str)
            )
        updated_hardware, validation_notes = self._normalize_hardware(
            proposed=proposed_hardware,
            current=hardware,
            validation_notes=validation_notes,
        )
        if hardware_fingerprint(updated_hardware) in forbidden_hardware_fingerprints:
            repaired, repair_actions = self.make_exploration_move(
                hardware=updated_hardware,
                state=state,
                forbidden_hardware_fingerprints=forbidden_hardware_fingerprints,
            )
            if hardware_fingerprint(repaired) != hardware_fingerprint(updated_hardware):
                updated_hardware = repaired
                validation_notes.append("proposal matched an evaluated hardware fingerprint; applied exploration repair")
                validation_notes.extend(repair_actions)
            else:
                validation_notes.append("proposal matched an evaluated hardware fingerprint and no legal repair was found")
        inferred_actions = self._diff_actions(hardware, updated_hardware)
        for action in inferred_actions:
            if action not in actions:
                actions.append(action)
        raw_notes = result.get("notes", {})
        notes = dict(raw_notes) if isinstance(raw_notes, dict) else {"raw_notes": raw_notes}
        notes["tool_calls"] = final_tool_calls
        notes["validation"] = validation_notes
        notes["updated_hardware_fingerprint"] = hardware_fingerprint(updated_hardware)
        notes["tool_results"] = serialize_tool_results(tool_results)
        notes["react_trace"] = react_trace
        notes["react_backend"] = "langgraph"
        notes["proposal_base"] = proposal_base
        notes["agent_mode"] = "single"
        return SolutionProposal(
            strategy=strategy,
            updated_hardware=updated_hardware,
            actions=actions,
            retrieved_cases=retrieved_cases,
            rationale=rationale,
            llm_notes=notes,
        )

    def _parse_bottleneck_state(self, payload: Dict[str, Any], fallback_base: Dict[str, Any]) -> BottleneckState:
        primary_impact = self._require_choice(
            payload,
            "primary_impact",
            {"latency", "energy", "monetary_cost", "mixed", "unknown"},
        )
        dominant_root_cause = self._require_choice(
            payload,
            "dominant_root_cause",
            {"compute", "memory", "communication", "buffer", "scheduling", "imbalance", "mixed", "unknown"},
        )
        diagnoses = [self._normalize_layer_diagnosis(item) for item in self._require_nonempty_list(payload, "layer_diagnoses")]
        retrieval_description = self._require_str(payload, "retrieval_description")
        root_cause_summary = self._require_ratios(payload, "root_cause_summary")
        recommended_focus = self._require_string_list(payload, "recommended_focus")
        raw_notes = payload.get("notes", {})
        notes = dict(raw_notes) if isinstance(raw_notes, dict) else {"raw_notes": raw_notes}
        notes.setdefault("analysis_base", fallback_base)
        return BottleneckState(
            primary_impact=primary_impact,
            dominant_root_cause=dominant_root_cause,
            layer_diagnoses=diagnoses,
            retrieval_description=retrieval_description,
            root_cause_summary=root_cause_summary,
            recommended_focus=recommended_focus,
            llm_notes=notes,
        )

    def _initial_active_base(self, evaluation: EvaluationResult) -> Dict[str, Any]:
        iteration = None
        name = evaluation.run_dir.name
        if name.startswith("iter_"):
            try:
                iteration = int(name.split("_", 1)[1])
            except (IndexError, ValueError):
                iteration = None
        return {"source": "current", "iteration": iteration, "base_key": "current"}

    def _evaluation_for_base_selection(
        self,
        base: Dict[str, Any],
        evaluation_bases: Dict[str, EvaluationResult],
        default: EvaluationResult,
    ) -> EvaluationResult:
        if not isinstance(base, dict):
            return default
        key = base.get("base_key")
        if isinstance(key, str) and key in evaluation_bases:
            return evaluation_bases[key]
        source = base.get("source")
        if isinstance(source, str) and source in evaluation_bases:
            return evaluation_bases[source]
        iteration = base.get("iteration")
        if iteration is not None:
            iter_key = f"iteration:{iteration}"
            if iter_key in evaluation_bases:
                return evaluation_bases[iter_key]
        return default

    def _evaluation_context(self, evaluation: EvaluationResult) -> Dict[str, Any]:
        return {
            "metrics": self._metrics_with_objective(evaluation.metrics, evaluation.hardware),
        }

    def _metrics_with_objective(self, metrics: Dict[str, float], hardware: Dict[str, Any]) -> Dict[str, float]:
        return metrics_with_objective(metrics, hardware=hardware, search_space=self._search_space())

    def _model_message(self, model_result: ModelAnalysisResult) -> Dict[str, Any]:
        return {
            "metrics": model_result.metrics,
            "summary": model_result.summary,
            "global_findings": model_result.global_findings,
            "selected_views": model_result.selected_views,
            "candidate_layers": model_result.candidate_layers,
            "analysis_base": model_result.analysis_base,
        }

    def _require_nested_dict(self, payload: Dict[str, Any], key: str) -> Dict[str, Any]:
        value = payload.get(key)
        if not isinstance(value, dict):
            raise ValueError(f"Single-agent result must contain object field {key!r}")
        return value

    def _require_nonempty_list(self, payload: Dict[str, Any], key: str) -> List[Any]:
        value = payload.get(key)
        if not isinstance(value, list) or not value:
            raise ValueError(f"Single-agent result must contain non-empty list field {key!r}")
        return value

    def _require_string_list(self, payload: Dict[str, Any], key: str) -> List[str]:
        value = self._require_nonempty_list(payload, key)
        if not all(isinstance(item, str) and item.strip() for item in value):
            raise ValueError(f"Single-agent result field {key!r} must be a list of non-empty strings")
        return value

    def _require_choice(self, payload: Dict[str, Any], key: str, choices: set[str]) -> str:
        value = self._require_str(payload, key)
        if value not in choices:
            raise ValueError(f"Invalid {key}: {value!r}; expected one of {sorted(choices)}")
        return value

    def _require_ratios(self, payload: Dict[str, Any], key: str) -> Dict[str, float]:
        value = payload.get(key)
        if not isinstance(value, dict):
            raise ValueError(f"Single-agent result must contain dict field {key!r}")
        missing = {"compute", "memory", "communication", "buffer"} - set(value)
        if missing:
            raise ValueError(f"Single-agent result field {key!r} missing dimensions: {sorted(missing)}")
        ratios = {}
        for dim in ["compute", "memory", "communication", "buffer"]:
            try:
                ratios[dim] = float(value[dim])
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Single-agent result field {key!r}.{dim} must be numeric") from exc
        return ratios

    def _normalize_candidate_layer(self, item: Any) -> Dict[str, Any]:
        if not isinstance(item, dict):
            raise ValueError(f"Candidate layer must be an object, got {item!r}")
        return {
            "layer_id": item.get("layer_id", item.get("layerID")),
            "layer_name": item.get("layer_name", item.get("layerName")),
            "operator_group": item.get("operator_group", item.get("group", "unknown")),
            "concern_types": self._optional_string_list(item.get("concern_types", item.get("impact_types", []))),
            "rank_metric": item.get("rank_metric"),
            "rank_value": item.get("rank_value"),
            "evidence": self._optional_string_list(item.get("evidence", [])),
            "confidence": str(item.get("confidence", "medium")),
        }

    def _normalize_layer_diagnosis(self, item: Any) -> Dict[str, Any]:
        if not isinstance(item, dict):
            raise ValueError(f"Layer diagnosis must be an object, got {item!r}")
        ratios = item.get("root_cause_ratios", item.get("ratios", {}))
        ratios = ratios if isinstance(ratios, dict) else {}
        return {
            "layerID": item.get("layerID", item.get("layer_id")),
            "layerName": item.get("layerName", item.get("layer_name")),
            "operator_group": item.get("operator_group", item.get("group", "unknown")),
            "impact_types": self._optional_string_list(item.get("impact_types", [])),
            "root_causes": self._optional_string_list(item.get("root_causes", [])),
            "dominant_root_cause": item.get("dominant_root_cause", item.get("bottleneck_type", "unknown")),
            "root_cause_ratios": {
                key: self._optional_float(ratios.get(key, 0.0))
                for key in ["compute", "memory", "communication", "buffer"]
            },
            "load_features": item.get("load_features", {}),
            "diagnosis": item.get("diagnosis", item.get("problem_location", "")),
        }

    def _optional_string_list(self, value: Any) -> List[str]:
        if isinstance(value, str):
            value = [value]
        if not isinstance(value, list):
            return []
        return [str(item) for item in value if isinstance(item, str) and item.strip()]

    def _optional_float(self, value: Any) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return 0.0
