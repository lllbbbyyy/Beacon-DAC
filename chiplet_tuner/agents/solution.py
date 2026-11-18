from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from chiplet_tuner.agents.react_graph import run_react_tool_graph
from chiplet_tuner.agents.analysis_policy import analysis_role_prompt
from chiplet_tuner.agents.tool_use import (
    complete_json_continuing_messages,
    serialize_tool_results,
)
from chiplet_tuner.core.schemas import BottleneckState, EvaluationResult, SolutionProposal, ToolResult
from chiplet_tuner.core.search_space import (
    DEFAULT_HARDWARE_SEARCH_SPACE,
    chip_type_candidates,
    materialize_hardware,
    normalize_hardware_to_design_space,
    step_array_shape,
    step_chiplet_resource,
    system_param_candidates,
    system_param_keys,
)
from chiplet_tuner.core.utils import clamp_to_candidates
from chiplet_tuner.llm.clients import LLMClient
from chiplet_tuner.llm.prompts import REACT_TOOL_USE_PROMPT, SOLUTION_GENERATION_PROMPT, SOLUTION_REACT_PROMPT
from chiplet_tuner.rag.vector_store import HistoryVectorStore, hardware_fingerprint, summarize_cases
from chiplet_tuner.tools.analysis_tools import AnalysisToolbox, ToolContext


HARDWARE_EDIT_TOOLS = {"modify_hardware_parameter", "step_hardware_parameter"}
FINAL_HARDWARE_TOOL_NAMES = {"modify_hardware_parameter", "step_hardware_parameter"}


class SolutionGenerationAgent:
    """LLM-driven solution generator with RAG historical context."""

    def __init__(
        self,
        llm: LLMClient,
        store: HistoryVectorStore,
        toolbox: AnalysisToolbox,
        simulator_schema: Dict[str, Any],
        top_k: int = 5,
    ) -> None:
        self.llm = llm
        self.store = store
        self.toolbox = toolbox
        self.simulator_schema = simulator_schema
        self.top_k = top_k

    def propose(
        self,
        state: BottleneckState,
        hardware: Dict[str, Any],
        output_dir: Path,
        evaluation: Optional[EvaluationResult] = None,
        forbidden_hardware_fingerprints: Optional[Set[str]] = None,
        search_state: Optional[Dict[str, Any]] = None,
        evaluation_bases: Optional[Dict[str, EvaluationResult]] = None,
    ) -> SolutionProposal:
        forbidden_hardware_fingerprints = set(forbidden_hardware_fingerprints or set())
        evaluation_bases = dict(evaluation_bases or {})
        if evaluation is not None:
            evaluation_bases.setdefault("current", evaluation)
        state_message = self._state_message(state)
        cases = self.store.search(
            state.retrieval_description,
            hardware,
            top_k=self.top_k,
            bottleneck_state=state_message,
        )
        summarized_cases = summarize_cases(cases)
        context = ToolContext(
            output_dir=output_dir,
            evaluation=evaluation,
            bottleneck_state=state_message,
            current_hardware=hardware,
            retrieved_cases=summarized_cases,
            simulator_schema=self.simulator_schema,
            active_base=state.llm_notes.get("analysis_base", {}),
            search_state=search_state,
            evaluation_bases=evaluation_bases,
        )
        role_prompt = analysis_role_prompt("solution_generation", SOLUTION_REACT_PROMPT)
        react_result = run_react_tool_graph(
            llm=self.llm,
            toolbox=self.toolbox,
            system_prompt=f"{role_prompt}\n\n{REACT_TOOL_USE_PROMPT}",
            task="react_tool_use",
            agent_name="solution_generation",
            agent_goal=(
                "Gather hardware evidence for the final update decision. Do not modify hardware during tool use; "
                "the final solution JSON will provide the hardware-edit tool calls that the framework applies."
            ),
            context_payload={
                "bottleneck_state": state_message,
                "analysis_base": state.llm_notes.get("analysis_base", {}),
                "current_hardware": hardware,
                "retrieved_cases": summarized_cases,
                "search_space": self.simulator_schema.get("search_space", {}),
                "forbidden_hardware_fingerprints": sorted(forbidden_hardware_fingerprints),
                "search_state_available": search_state is not None,
                "analysis_base_tools_available": bool(evaluation_bases),
            },
            context=context,
            max_steps=15,
            allowed_tool_names=self._solution_analysis_tool_names(),
        )
        tool_results = dict(react_result.tool_results)
        react_trace = list(react_result.transcript)
        react_messages = list(react_result.messages)
        proposal_base_hardware = copy.deepcopy(context.current_hardware or hardware)
        proposal_base_record = context.active_base or state.llm_notes.get("analysis_base", {})
        result = complete_json_continuing_messages(
            self.llm,
            react_messages,
            {
                "task": "solution_generation",
                "final_decision_required": True,
                "instruction": (
                    "Tool use is complete. Continue from the previous messages and produce the "
                    "final solution proposal. Decide the hardware changes now by returning a "
                    "canonical hardware-edit tool_calls list. The framework will execute those calls "
                    "sequentially on the proposal_base_hardware after this response. "
                    "Use the real tool observations already present in this conversation, not a "
                    "separately summarized context."
                ),
                "proposal_base": proposal_base_record,
                "proposal_base_hardware_summary": self._proposal_base_summary(proposal_base_hardware),
                "allowed_final_tools": sorted(FINAL_HARDWARE_TOOL_NAMES),
                "output_prompt": SOLUTION_GENERATION_PROMPT,
            },
            validate_response=lambda payload: self._validate_solution_result(
                payload,
                context=context,
                base_hardware=proposal_base_hardware,
            ),
        )
        strategy = self._require_str(result, "strategy")
        final_tool_calls = self._require_final_tool_calls(result)
        update_results = self._apply_final_tool_calls(
            context=context,
            base_hardware=proposal_base_hardware,
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
            current=proposal_base_hardware,
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
        inferred_actions = self._diff_actions(proposal_base_hardware, updated_hardware)
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
        notes["react_backend"] = (
            "langgraph" if react_result.execution_mode == "iterative" else "batched_tool_plan"
        )
        notes["react_execution_mode"] = react_result.execution_mode
        notes["react_batch_max_tools"] = react_result.max_selected_tools
        notes["proposal_base"] = proposal_base_record
        return SolutionProposal(
            strategy=strategy,
            updated_hardware=updated_hardware,
            actions=actions,
            retrieved_cases=summarized_cases,
            rationale=rationale,
            llm_notes=notes,
        )

    def _validate_solution_result(
        self,
        result: Dict[str, Any],
        context: ToolContext,
        base_hardware: Dict[str, Any],
    ) -> None:
        self._require_str(result, "strategy")
        final_tool_calls = self._require_final_tool_calls(result)
        validation_context = self._copy_context_for_hardware(context=context, hardware=base_hardware)
        self._execute_final_tool_calls(validation_context, final_tool_calls)
        self._require_str(result, "rationale", allow_empty=True)

    def _apply_final_tool_calls(
        self,
        context: ToolContext,
        base_hardware: Dict[str, Any],
        tool_calls: List[Dict[str, Any]],
    ) -> List[ToolResult]:
        context.current_hardware = copy.deepcopy(base_hardware)
        return self._execute_final_tool_calls(context, tool_calls)

    def _execute_final_tool_calls(
        self,
        context: ToolContext,
        tool_calls: List[Dict[str, Any]],
    ) -> List[ToolResult]:
        results: List[ToolResult] = []
        for call in tool_calls:
            name = str(call["name"])
            arguments = copy.deepcopy(call["arguments"])
            results.append(self.toolbox.run(name, context, arguments=arguments))
        return results

    def _copy_context_for_hardware(self, context: ToolContext, hardware: Dict[str, Any]) -> ToolContext:
        return ToolContext(
            output_dir=context.output_dir,
            evaluation=context.evaluation,
            bottleneck_state=context.bottleneck_state,
            current_hardware=copy.deepcopy(hardware),
            retrieved_cases=context.retrieved_cases,
            simulator_schema=context.simulator_schema,
            active_base=context.active_base,
            search_state=context.search_state,
            evaluation_bases=context.evaluation_bases,
        )

    def _solution_analysis_tool_names(self) -> List[str]:
        return sorted(spec.name for spec in self.toolbox.specs() if spec.name not in HARDWARE_EDIT_TOOLS)

    def _state_message(self, state: BottleneckState) -> Dict[str, Any]:
        return {
            "primary_impact": state.primary_impact,
            "dominant_root_cause": state.dominant_root_cause,
            "layer_diagnoses": state.layer_diagnoses,
            "retrieval_description": state.retrieval_description,
            "root_cause_summary": state.root_cause_summary,
            "recommended_focus": state.recommended_focus,
        }

    def make_exploration_move(
        self,
        hardware: Dict[str, Any],
        state: BottleneckState,
        forbidden_hardware_fingerprints: Optional[Set[str]] = None,
    ) -> Tuple[Dict[str, Any], List[str]]:
        forbidden = set(forbidden_hardware_fingerprints or set())
        focus_order = list(state.recommended_focus) or self._focus_for_bottleneck(state.dominant_root_cause)
        focus_order.extend(
            ["num_chiplets", "compute_units", "buffer_size", *system_param_keys(self._search_space()), "chiplet_type"]
        )
        directions = self._directions_for_bottleneck(state.dominant_root_cause)
        tried = set()
        for key in focus_order:
            if key in tried:
                continue
            tried.add(key)
            for direction in directions.get(key, [1, -1]):
                candidate = copy.deepcopy(hardware)
                actions: List[str] = []
                if key in {"array_shape", "num_chiplets", "shape", "chip_shape"}:
                    self._step_chiplet_count(candidate, direction, actions)
                elif key in {"compute_units", "buffer_size"}:
                    self._step_chiplet_resource(candidate, key, direction, actions)
                elif key in {"chiplet_type", "type"}:
                    self._step_chiplet_type(candidate, direction, actions)
                elif key in candidate:
                    self._step_top_level(candidate, key, direction, actions)
                else:
                    self._step_chip_field(candidate, key, direction, actions)
                if actions and hardware_fingerprint(candidate) not in forbidden:
                    return candidate, actions
        return hardware, []

    def _normalize_hardware(
        self,
        proposed: Dict[str, Any],
        current: Dict[str, Any],
        validation_notes: Optional[List[str]] = None,
    ) -> Tuple[Dict[str, Any], List[str]]:
        updated, notes = normalize_hardware_to_design_space(
            proposed=proposed,
            current=current,
            search_space=self._search_space(),
        )
        combined_notes = list(validation_notes or [])
        combined_notes.extend(notes)
        return updated, combined_notes

    def _require_dict(self, payload: Dict[str, Any], key: str) -> Dict[str, Any]:
        value = payload.get(key)
        if not isinstance(value, dict):
            raise ValueError(f"Solution LLM result must contain object field {key!r}")
        return value

    def _require_str(self, payload: Dict[str, Any], key: str, allow_empty: bool = False) -> str:
        value = payload.get(key)
        if not isinstance(value, str) or (not allow_empty and not value.strip()):
            raise ValueError(f"Solution LLM result must contain string field {key!r}")
        return value

    def _require_final_tool_calls(self, payload: Dict[str, Any]) -> List[Dict[str, Any]]:
        raw_calls = payload.get("tool_calls")
        if not isinstance(raw_calls, list) or not raw_calls:
            raise ValueError("Solution LLM result must contain non-empty tool_calls=[...].")
        calls: List[Dict[str, Any]] = []
        for item in raw_calls:
            if not isinstance(item, dict):
                raise ValueError(f"Each final tool call must be an object, got {item!r}.")
            unknown = sorted(set(item) - {"name", "arguments"})
            if unknown:
                raise ValueError(
                    "Each final tool call accepts only name and arguments; "
                    f"unexpected keys: {unknown}."
                )
            name = item.get("name")
            if name not in FINAL_HARDWARE_TOOL_NAMES:
                raise ValueError(
                    "Final solution may only call hardware-edit tools "
                    f"{sorted(FINAL_HARDWARE_TOOL_NAMES)}, got {name!r}."
                )
            arguments = item.get("arguments")
            if not isinstance(arguments, dict):
                raise ValueError(f"Final tool call {name!r} requires object arguments.")
            calls.append({"name": name, "arguments": copy.deepcopy(arguments)})
        return calls

    def _proposal_base_summary(self, hardware: Dict[str, Any]) -> Dict[str, Any]:
        chiplets = hardware.get("chiplets", []) if isinstance(hardware, dict) else []
        chiplet_summary = []
        for idx, chiplet in enumerate(chiplets):
            if not isinstance(chiplet, dict):
                continue
            chiplet_summary.append(
                {
                    "index": idx,
                    "type": chiplet.get("type"),
                    "compute_units": chiplet.get("compute_units"),
                    "buffer_size": chiplet.get("buffer_size"),
                }
            )
        return {
            "num_chiplets": hardware.get("num_chiplets"),
            "chip_y": hardware.get("chip_y"),
            "chip_x": hardware.get("chip_x"),
            "system_params": {
                key: hardware.get(key)
                for key in system_param_keys(self._search_space())
                if key in hardware
            },
            "chiplets": chiplet_summary,
            "fingerprint": hardware_fingerprint(hardware),
        }

    def _numeric_candidates(self, candidates: Sequence[Any]) -> List[float]:
        numeric: List[float] = []
        for item in candidates:
            if isinstance(item, bool):
                return []
            if not isinstance(item, (int, float)):
                return []
            numeric.append(float(item))
        return numeric

    def _step_top_level(self, hardware: Dict[str, Any], key: str, direction: int, actions: List[str]) -> None:
        search_space = self._search_space()
        candidates = system_param_candidates(search_space, key)
        if key not in hardware or not candidates or not self._numeric_candidates(candidates):
            return
        old = int(hardware[key])
        new = clamp_to_candidates(old, candidates, direction)
        if new != old:
            hardware[key] = new
            actions.append(f"{key}: {old}->{new}")

    def _step_chip_field(self, hardware: Dict[str, Any], key: str, direction: int, actions: List[str]) -> None:
        if key in {"compute_units", "buffer_size"}:
            self._step_chiplet_resource(hardware, key, direction, actions)
        elif key in {"type", "chiplet_type"}:
            self._step_chiplet_type(hardware, direction, actions)

    def _step_chiplet_count(self, hardware: Dict[str, Any], direction: int, actions: List[str]) -> None:
        search_space = self._search_space()
        try:
            shape = step_array_shape(hardware, search_space, direction)
        except ValueError:
            return
        old_shape = (hardware.get("chip_y"), hardware.get("chip_x"))
        new_shape = (shape.get("chip_y"), shape.get("chip_x"))
        if old_shape == new_shape:
            return
        old_num = hardware.get("num_chiplets")
        updated = materialize_hardware(
            template=hardware,
            num_chiplets=shape["num_chiplets"],
            search_space=search_space,
        )
        hardware.clear()
        hardware.update(updated)
        actions.append(
            f"num_chiplets {old_num}->{hardware.get('num_chiplets')}; "
            f"derived_shape {old_shape[0]}x{old_shape[1]}->{hardware.get('chip_y')}x{hardware.get('chip_x')}"
        )

    def _step_chiplet_resource(self, hardware: Dict[str, Any], key: str, direction: int, actions: List[str]) -> None:
        try:
            updated, resource_actions = step_chiplet_resource(
                hardware=hardware,
                search_space=self._search_space(),
                field=key,
                direction=direction,
                scope="indices",
                indices=[0],
            )
        except ValueError:
            return
        if not resource_actions:
            return
        hardware.clear()
        hardware.update(updated)
        actions.extend(resource_actions)

    def _step_chiplet_type(self, hardware: Dict[str, Any], direction: int, actions: List[str]) -> None:
        candidates = chip_type_candidates(self._search_space())
        if len(candidates) < 2:
            return
        chiplets = hardware.get("chiplets", [])
        if not isinstance(chiplets, list):
            return
        for idx, chip in enumerate(chiplets):
            if not isinstance(chip, dict):
                continue
            old = str(chip.get("type", candidates[0]))
            old_idx = candidates.index(old) if old in candidates else 0
            new_idx = max(0, min(len(candidates) - 1, old_idx + direction))
            if new_idx == old_idx:
                continue
            chip["type"] = candidates[new_idx]
            actions.append(f"chiplet_type[{idx}]: {old}->{chip['type']}")
            return

    def _diff_actions(self, old: Dict[str, Any], new: Dict[str, Any]) -> List[str]:
        actions: List[str] = []
        for key in system_param_keys(self._search_space()):
            if old.get(key) != new.get(key):
                actions.append(f"{key}: {old.get(key)}->{new.get(key)}")
        if old.get("num_chiplets") != new.get("num_chiplets"):
            actions.append(f"num_chiplets: {old.get('num_chiplets')}->{new.get('num_chiplets')}")
        old_shape = (old.get("chip_y"), old.get("chip_x"))
        new_shape = (new.get("chip_y"), new.get("chip_x"))
        if old_shape != new_shape:
            actions.append(f"chip_shape: {old_shape[0]}x{old_shape[1]}->{new_shape[0]}x{new_shape[1]}")
        old_chiplets = old.get("chiplets", [])
        new_chiplets = new.get("chiplets", [])
        changed_types = 0
        changed_compute = 0
        changed_buffer = 0
        for old_chip, new_chip in zip(old_chiplets, new_chiplets):
            if not isinstance(old_chip, dict) or not isinstance(new_chip, dict):
                continue
            if old_chip.get("type") != new_chip.get("type"):
                changed_types += 1
            if old_chip.get("compute_units") != new_chip.get("compute_units"):
                changed_compute += 1
            if old_chip.get("buffer_size") != new_chip.get("buffer_size"):
                changed_buffer += 1
        if changed_types:
            actions.append(f"chiplet_type: changed on {changed_types} chiplets")
        if changed_compute:
            actions.append(f"compute_units: changed on {changed_compute} chiplets")
        if changed_buffer:
            actions.append(f"buffer_size: changed on {changed_buffer} chiplets")
        return actions

    def _directions_for_bottleneck(self, bottleneck_type: str) -> Dict[str, List[int]]:
        has_tensor = bool(system_param_candidates(self._search_space(), "tensor_parall"))
        has_micro_batch = bool(system_param_candidates(self._search_space(), "micro_batch"))
        if bottleneck_type == "communication":
            directions = {"nop_bw": [1], "chiplet_type": [1, -1], "num_chiplets": [1, -1]}
            if has_tensor:
                directions["tensor_parall"] = [-1, 1]
            if has_micro_batch:
                directions["micro_batch"] = [-1, 1]
            return directions
        if bottleneck_type == "memory":
            directions = {"dram_bw": [1], "buffer_size": [1]}
            if has_micro_batch:
                directions["micro_batch"] = [-1, 1]
            return directions
        if bottleneck_type == "compute":
            directions = {"compute_units": [1]}
            if has_tensor:
                directions["tensor_parall"] = [1, -1]
            if has_micro_batch:
                directions["micro_batch"] = [1, -1]
            return directions
        if bottleneck_type == "buffer":
            return {"buffer_size": [1], "chiplet_type": [1, -1]}
        return {}

    def _focus_for_bottleneck(self, bottleneck_type: str) -> List[str]:
        has_tensor = bool(system_param_candidates(self._search_space(), "tensor_parall"))
        has_micro_batch = bool(system_param_candidates(self._search_space(), "micro_batch"))
        focus = {
            "compute": ["compute_units", "num_chiplets"],
            "memory": ["dram_bw", "buffer_size"],
            "communication": ["nop_bw", "chiplet_type", "num_chiplets"],
            "buffer": ["buffer_size", "chiplet_type"],
        }.get(bottleneck_type, ["dram_bw", "nop_bw", "compute_units", "buffer_size", "chiplet_type"])
        if has_tensor and bottleneck_type in {"compute", "communication"}:
            insert_at = 1 if bottleneck_type == "compute" else 1
            focus.insert(insert_at, "tensor_parall")
        if has_micro_batch and bottleneck_type in {"compute", "memory", "communication"}:
            focus.insert(1, "micro_batch")
        return focus

    def _search_space(self) -> Dict[str, Any]:
        return self.simulator_schema.get("search_space") or DEFAULT_HARDWARE_SEARCH_SPACE
