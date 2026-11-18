from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Sequence, TypedDict

from chiplet_tuner.agents.tool_use import (
    ReactLoopResult,
    _full_tool_result,
    _json_message,
    _react_system_prompt,
    _require_str,
    _tool_call_key,
    _unique_result_key,
    _validate_react_response,
    complete_json_with_retries,
    _record_retry_event,
    _retry_temperature_for_attempt,
    tool_specs_for_prompt,
)
from chiplet_tuner.agents.analysis_policy import resolve_analysis_policy
from chiplet_tuner.core.schemas import ToolResult
from chiplet_tuner.llm.clients import LLMClient
from chiplet_tuner.llm.prompts import REACT_TOOL_USE_PROMPT
from chiplet_tuner.tools.analysis_tools import AnalysisToolbox, ToolContext


REACT_EXECUTION_MODE_ENV = "CHIPLET_TUNER_REACT_EXECUTION_MODE"
REACT_BATCH_MAX_TOOLS_ENV = "CHIPLET_TUNER_REACT_BATCH_MAX_TOOLS"
DEFAULT_REACT_EXECUTION_MODE = "iterative"
DEFAULT_REACT_BATCH_MAX_TOOLS = 15


class ReactGraphState(TypedDict, total=False):
    messages: List[Dict[str, Any]]
    tool_results: Dict[str, Dict[str, Any]]
    transcript: List[Dict[str, Any]]
    seen_tool_calls: List[str]
    step: int
    max_steps: int
    agent_name: str
    pending_response: Dict[str, Any]
    pending_assistant_message: Dict[str, Any]
    pending_tool_name: str
    pending_tool_arguments: Dict[str, Any]
    pending_duplicate: bool
    tool_error_count: int
    tool_retry_temperature: Optional[float]
    finished: bool


def run_react_tool_graph(
    llm: LLMClient,
    toolbox: AnalysisToolbox,
    system_prompt: str,
    task: str,
    agent_name: str,
    agent_goal: str,
    context_payload: Dict[str, Any],
    context: ToolContext,
    max_steps: int = 8,
    allowed_tool_names: Optional[Sequence[str]] = None,
    execution_mode: Optional[str] = None,
    batch_max_tools: Optional[int] = None,
) -> ReactLoopResult:
    """Run iterative ReAct or the open-loop batched-tool ablation.

    ``iterative`` preserves the observation-conditioned LangGraph loop.
    ``batched`` asks for at most ``batch_max_tools`` from the initial context,
    executes the complete plan, and returns all observations at once. The next
    LLM call is therefore the agent's existing final structured-output call.
    """

    if max_steps <= 0:
        raise ValueError(f"max_steps must be positive, got {max_steps}")
    max_steps, analysis_guidance = resolve_analysis_policy(agent_name, max_steps)
    if analysis_guidance:
        system_prompt = f"{system_prompt}\n\n{analysis_guidance}"
        context_payload = dict(
            context_payload, analysis_policy=os.environ.get("CHIPLET_TUNER_ANALYSIS_POLICY", "standard")
        )
    evidence_prompt = getattr(toolbox, "evidence_prompt", "")
    if evidence_prompt:
        system_prompt = f"{system_prompt}\n\n{evidence_prompt}"
        context_payload = dict(context_payload, evidence_mode="scalar")
    resolved_mode = _resolve_execution_mode(execution_mode)

    if allowed_tool_names is None:
        prompt_tool_specs = tool_specs_for_prompt(toolbox)
        valid_tool_names = {spec.name for spec in toolbox.specs()}
    else:
        allowed = set(allowed_tool_names)
        prompt_tool_specs = tool_specs_for_prompt(toolbox, allowed_tool_names=sorted(allowed))
        valid_tool_names = {spec.name for spec in toolbox.specs() if spec.name in allowed}
    if resolved_mode == "batched":
        resolved_batch_max_tools = _resolve_batch_max_tools(batch_max_tools)
        return _run_batched_tool_plan(
            llm=llm,
            toolbox=toolbox,
            system_prompt=system_prompt,
            task=task,
            agent_name=agent_name,
            agent_goal=agent_goal,
            context_payload=context_payload,
            context=context,
            prompt_tool_specs=prompt_tool_specs,
            valid_tool_names=valid_tool_names,
            max_tools=resolved_batch_max_tools,
        )
    graph = _build_react_graph(
        llm=llm,
        toolbox=toolbox,
        context=context,
        task=task,
        valid_tool_names=valid_tool_names,
    )
    messages = [
        {
            "role": "system",
            "content": _react_system_prompt(
                system_prompt=system_prompt,
                available_tools=prompt_tool_specs,
            ),
        },
        {
            "role": "user",
            "content": _json_message(
                {
                    "task": task,
                    "agent_name": agent_name,
                    "agent_goal": agent_goal,
                    "context": context_payload,
                    "step": 0,
                    "max_steps": max_steps,
                    "instruction": "Choose exactly one next tool call or finish tool use.",
                }
            ),
        },
    ]
    initial_state: ReactGraphState = {
        "messages": messages,
        "tool_results": {},
        "transcript": [],
        "seen_tool_calls": [],
        "step": 0,
        "max_steps": max_steps,
        "agent_name": agent_name,
        "tool_error_count": 0,
        "tool_retry_temperature": None,
        "finished": False,
    }
    max_tool_retries = max(0, int(getattr(llm, "retry_attempts", 0) or 0))
    final_state = graph.compile().invoke(
        initial_state,
        config={"recursion_limit": max(10, max_steps * (4 + max_tool_retries * 3) + 10)},
    )
    return ReactLoopResult(
        tool_results=_tool_results_from_records(final_state.get("tool_results", {})),
        transcript=final_state.get("transcript", []),
        messages=final_state.get("messages", messages),
    )


def _resolve_execution_mode(explicit_mode: Optional[str]) -> str:
    mode = explicit_mode
    if mode is None:
        mode = os.environ.get(REACT_EXECUTION_MODE_ENV, DEFAULT_REACT_EXECUTION_MODE)
    normalized = str(mode).strip().lower().replace("-", "_")
    aliases = {
        "react": "iterative",
        "iterative": "iterative",
        "batched": "batched",
        "batch": "batched",
        "open_loop": "batched",
    }
    if normalized not in aliases:
        raise ValueError(
            f"Unsupported ReAct execution mode {mode!r}; expected iterative or batched."
        )
    return aliases[normalized]


def _resolve_batch_max_tools(explicit_limit: Optional[int]) -> int:
    raw_limit: Any = explicit_limit
    if raw_limit is None:
        raw_limit = os.environ.get(REACT_BATCH_MAX_TOOLS_ENV, DEFAULT_REACT_BATCH_MAX_TOOLS)
    try:
        limit = int(raw_limit)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"ReAct batched tool limit must be an integer, got {raw_limit!r}.") from exc
    if limit <= 0:
        raise ValueError(f"ReAct batched tool limit must be positive, got {limit}.")
    return limit


def _run_batched_tool_plan(
    llm: LLMClient,
    toolbox: AnalysisToolbox,
    system_prompt: str,
    task: str,
    agent_name: str,
    agent_goal: str,
    context_payload: Dict[str, Any],
    context: ToolContext,
    prompt_tool_specs: List[Dict[str, Any]],
    valid_tool_names: set[str],
    max_tools: int,
) -> ReactLoopResult:
    messages = [
        {
            "role": "system",
            "content": _react_system_prompt(
                system_prompt=(
                    system_prompt.replace(REACT_TOOL_USE_PROMPT, "").strip()
                    + "\n\n"
                    + _batched_tool_protocol(max_tools)
                ),
                available_tools=prompt_tool_specs,
            ),
        },
        {
            "role": "user",
            "content": _json_message(
                {
                    "task": "batched_tool_selection",
                    "source_task": task,
                    "agent_name": agent_name,
                    "agent_goal": agent_goal,
                    "context": context_payload,
                    "max_tools": max_tools,
                    "instruction": (
                        "Using only this initial context, choose zero or more analysis tools up to "
                        "max_tools. This is the only tool-selection turn."
                    ),
                }
            ),
        },
    ]
    response = complete_json_with_retries(
        llm,
        messages,
        validate_response=lambda payload: _validate_batched_tool_response(
            payload,
            valid_tool_names=valid_tool_names,
            max_tools=max_tools,
        ),
        retry_context={"task": "batched_tool_selection", "agent_name": agent_name},
    )
    assistant_message = llm.assistant_message_from_response(response, _json_message(response))
    thought = _require_str(response, "thought")
    tool_calls = response.get("tools", [])
    tool_results: Dict[str, ToolResult] = {}
    transcript: List[Dict[str, Any]] = []
    observations: List[Dict[str, Any]] = []

    for index, tool_call in enumerate(tool_calls):
        tool_name = _require_str(tool_call, "name")
        arguments = dict(tool_call.get("arguments", {}))
        result_key = _unique_result_key(tool_results, tool_name)
        try:
            result = toolbox.run(tool_name, context, arguments=arguments)
            observation = _full_tool_result(result)
            transcript_action = "tool"
        except Exception as exc:
            if not _is_retryable_tool_error(exc):
                raise
            result = ToolResult(
                name=tool_name,
                payload={
                    "status": "error",
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                },
                generated_files={},
            )
            observation = _full_tool_result(result)
            transcript_action = "tool_error"
        tool_results[result_key] = result
        observations.append(
            {
                "execution_index": index,
                "result_key": result_key,
                "tool_name": tool_name,
                "arguments": arguments,
                "observation": observation,
            }
        )
        transcript.append(
            {
                "step": index,
                "thought": thought,
                "action": transcript_action,
                "execution_mode": "batched",
                "tool_name": tool_name,
                "result_key": result_key,
                "arguments": arguments,
            }
        )

    messages.append(assistant_message)
    messages.append(
        {
            "role": "user",
            "content": _json_message(
                {
                    "task": "batched_tool_observations",
                    "agent_name": agent_name,
                    "execution_mode": "batched",
                    "selected_tool_count": len(tool_calls),
                    "max_tools": max_tools,
                    "observations": observations,
                    "instruction": (
                        "All selected tools have now been executed. Tool selection is closed. "
                        "Use these observations in the next final structured-output response; "
                        "do not request or select another tool."
                    ),
                }
            ),
        }
    )
    if not transcript:
        transcript.append(
            {
                "step": 0,
                "thought": thought,
                "action": "finish",
                "execution_mode": "batched",
                "finish_reason": "initial plan selected no analysis tools",
            }
        )
    return ReactLoopResult(
        tool_results=tool_results,
        transcript=transcript,
        messages=messages,
        execution_mode="batched",
        max_selected_tools=max_tools,
    )


def _batched_tool_protocol(max_tools: int) -> str:
    return f"""
BATCHED TOOL-SELECTION ABLATION:
- This protocol supersedes the iterative one-tool-at-a-time instructions above.
- Select every analysis tool from the initial context in one response.
- Select at most {max_tools} calls. An empty tools list is valid when no observation is needed.
- The framework executes the selected calls in list order and returns all observations together.
- You will not receive an observation before choosing the remaining calls.
- After the batch observations, produce the role's final structured output directly.

Return one strict JSON object:
{{
  "thought": "brief rationale for the complete evidence plan",
  "tools": [
    {{"name": "tool_name_from_AVAILABLE_TOOLS", "arguments": {{}}}}
  ]
}}
""".strip()


def _validate_batched_tool_response(
    response: Dict[str, Any],
    valid_tool_names: set[str],
    max_tools: int,
) -> None:
    _require_str(response, "thought")
    tools = response.get("tools")
    if not isinstance(tools, list):
        raise ValueError("Batched tool response must contain array field 'tools'.")
    if len(tools) > max_tools:
        raise ValueError(
            f"Batched tool response selected {len(tools)} tools; maximum is {max_tools}."
        )
    seen_calls: set[str] = set()
    for index, tool_call in enumerate(tools):
        if not isinstance(tool_call, dict):
            raise ValueError(f"Batched tool call at index {index} must be an object.")
        tool_name = _require_str(tool_call, "name")
        if tool_name not in valid_tool_names:
            raise ValueError(
                f"Unknown tool {tool_name!r}; expected one of {sorted(valid_tool_names)}"
            )
        arguments = tool_call.get("arguments", {})
        if not isinstance(arguments, dict):
            raise ValueError(
                f"Batched tool call arguments at index {index} must be an object."
            )
        call_key = _tool_call_key(tool_name, arguments)
        if call_key in seen_calls:
            raise ValueError(
                f"Batched tool response contains duplicate call {tool_name!r} at index {index}."
            )
        seen_calls.add(call_key)


def _tool_result_record(result: ToolResult) -> Dict[str, Any]:
    return {
        "name": result.name,
        "payload": result.payload,
        "generated_files": result.generated_files,
    }


def _tool_result_from_record(record: Dict[str, Any]) -> ToolResult:
    return ToolResult(
        name=str(record.get("name", "")),
        payload=dict(record.get("payload", {})),
        generated_files=dict(record.get("generated_files", {})),
    )


def _tool_results_from_records(records: Dict[str, Dict[str, Any]]) -> Dict[str, ToolResult]:
    return {key: _tool_result_from_record(record) for key, record in records.items()}


def _build_react_graph(
    llm: LLMClient,
    toolbox: AnalysisToolbox,
    context: ToolContext,
    task: str,
    valid_tool_names: set[str],
) -> Any:
    try:
        from langgraph.graph import END, StateGraph
    except ImportError as exc:
        raise ImportError(
            "LangGraph is required for agent ReAct tool execution. "
            "Install it in the active environment with: pip install langgraph"
        ) from exc

    def llm_decide(state: ReactGraphState) -> Dict[str, Any]:
        step = int(state.get("step", 0))
        max_steps = int(state.get("max_steps", 0))
        agent = str(state.get("agent_name", "unknown"))
        if step >= max_steps:
            transcript = list(state.get("transcript", []))
            transcript.append(
                {
                    "step": max_steps,
                    "thought": f"{agent} reached max_steps={max_steps}; continuing with gathered evidence.",
                    "action": "finish",
                    "finish_reason": "max_steps reached without explicit finish action",
                }
            )
            return {"finished": True, "transcript": transcript}

        messages = list(state.get("messages", []))
        previous_temperature = getattr(llm, "temperature_override", None)
        tool_retry_temperature = state.get("tool_retry_temperature")
        if tool_retry_temperature is not None:
            llm.temperature_override = float(tool_retry_temperature)
        try:
            response = complete_json_with_retries(
                llm,
                messages,
                validate_response=lambda payload: _validate_react_response(payload, valid_tool_names),
                retry_context={"task": task, "agent_name": agent},
            )
        finally:
            llm.temperature_override = previous_temperature
        assistant_message = llm.assistant_message_from_response(response, _json_message(response))
        action = _require_str(response, "action")
        thought = _require_str(response, "thought")
        if action == "finish":
            messages.append(assistant_message)
            transcript = list(state.get("transcript", []))
            transcript.append(
                {
                    "step": step,
                    "thought": thought,
                    "action": "finish",
                    "finish_reason": response.get("finish_reason", ""),
                }
            )
            return {"messages": messages, "transcript": transcript, "finished": True}

        tool_call = response.get("tool")
        if not isinstance(tool_call, dict):
            raise ValueError("ReAct tool action must include object field 'tool'.")
        tool_name = _require_str(tool_call, "name")
        arguments = tool_call.get("arguments", {})
        if not isinstance(arguments, dict):
            raise ValueError(f"Tool arguments for {tool_name} must be an object.")
        duplicate = _tool_call_key(tool_name, arguments) in set(state.get("seen_tool_calls", []))
        return {
            "pending_response": response,
            "pending_assistant_message": assistant_message,
            "pending_tool_name": tool_name,
            "pending_tool_arguments": arguments,
            "pending_duplicate": duplicate,
            "tool_retry_temperature": None,
            "finished": False,
        }

    def execute_tool(state: ReactGraphState) -> Dict[str, Any]:
        step = int(state.get("step", 0))
        max_steps = int(state.get("max_steps", 0))
        agent = str(state.get("agent_name", "unknown"))
        messages = list(state.get("messages", []))
        transcript = list(state.get("transcript", []))
        tool_results = dict(state.get("tool_results", {}))
        seen_tool_calls = set(state.get("seen_tool_calls", []))
        response = state["pending_response"]
        assistant_message = state["pending_assistant_message"]
        tool_name = state["pending_tool_name"]
        arguments = state.get("pending_tool_arguments", {})
        thought = response.get("thought")
        if not isinstance(thought, str) or not thought.strip():
            thought = f"Execute {tool_name} with the provided arguments."

        if state.get("pending_duplicate"):
            messages.append(assistant_message)
            transcript.append(
                {
                    "step": step,
                    "thought": thought,
                    "action": "finish",
                    "finish_reason": f"duplicate tool call {tool_name}; continuing with gathered evidence",
                }
            )
            return {"messages": messages, "transcript": transcript, "finished": True}

        try:
            result = toolbox.run(tool_name, context, arguments=arguments)
        except Exception as exc:
            if not _is_retryable_tool_error(exc):
                raise
            max_tool_retries = max(0, int(getattr(llm, "retry_attempts", 0) or 0))
            retry_attempt = int(state.get("tool_error_count", 0)) + 1
            if retry_attempt > max_tool_retries:
                _record_retry_event(
                    llm,
                    context={"task": f"{task}:tool_execution", "agent_name": agent},
                    event="retry_exhausted",
                    retry_attempt=retry_attempt - 1,
                    max_retries=max_tool_retries,
                    retry_temperature=None,
                    error=exc,
                )
                raise
            retry_temperature = _retry_temperature_for_attempt(llm, retry_attempt)
            _record_retry_event(
                llm,
                context={"task": f"{task}:tool_execution", "agent_name": agent},
                event="retry",
                retry_attempt=retry_attempt,
                max_retries=max_tool_retries,
                retry_temperature=retry_temperature,
                error=exc,
            )
            messages.append(assistant_message)
            messages.append(
                {
                    "role": "user",
                    "content": _json_message(
                        _tool_error_retry_payload(
                            agent=agent,
                            step=step,
                            max_steps=max_steps,
                            tool_name=tool_name,
                            arguments=arguments,
                            error=exc,
                            retry_attempt=retry_attempt,
                            max_retries=max_tool_retries,
                            retry_temperature=retry_temperature,
                        )
                    ),
                }
            )
            transcript.append(
                {
                    "step": step,
                    "thought": thought,
                    "action": "tool_error",
                    "tool_name": tool_name,
                    "arguments": arguments,
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                    "retry_attempt": retry_attempt,
                }
            )
            return {
                "messages": messages,
                "transcript": transcript,
                "tool_error_count": retry_attempt,
                "tool_retry_temperature": retry_temperature,
                "pending_response": {},
                "pending_assistant_message": {},
                "pending_tool_name": "",
                "pending_tool_arguments": {},
                "pending_duplicate": False,
            }

        seen_tool_calls.add(_tool_call_key(tool_name, arguments))
        result_key = _unique_result_key(tool_results, tool_name)
        tool_results[result_key] = _tool_result_record(result)
        observation = {
            "step": step,
            "tool_name": tool_name,
            "arguments": arguments,
            "observation": _full_tool_result(result),
        }
        messages.append(assistant_message)
        messages.append(
            {
                "role": "user",
                "content": _json_message(
                    {
                        "task": "tool_observation",
                        "agent_name": agent,
                        "observation_record": observation,
                        "next_step": step + 1,
                        "max_steps": max_steps,
                        "instruction": (
                            "Use this real tool observation with the previous messages. "
                            "Choose exactly one next tool call or finish tool use."
                        ),
                    }
                ),
            }
        )
        transcript.append(
            {
                "step": step,
                "thought": thought,
                "action": "tool",
                "tool_name": tool_name,
                "result_key": result_key,
                "arguments": arguments,
            }
        )
        return {
            "messages": messages,
            "tool_results": tool_results,
            "transcript": transcript,
            "seen_tool_calls": sorted(seen_tool_calls),
            "step": step + 1,
            "tool_error_count": 0,
            "tool_retry_temperature": None,
            "pending_response": {},
            "pending_assistant_message": {},
            "pending_tool_name": "",
            "pending_tool_arguments": {},
            "pending_duplicate": False,
        }

    def route_after_decide(state: ReactGraphState) -> str:
        if state.get("finished"):
            return "end"
        return "execute_tool"

    def route_after_tool(state: ReactGraphState) -> str:
        if state.get("finished"):
            return "end"
        return "llm_decide"

    graph = StateGraph(ReactGraphState)
    graph.add_node("llm_decide", llm_decide)
    graph.add_node("execute_tool", execute_tool)
    graph.set_entry_point("llm_decide")
    graph.add_conditional_edges(
        "llm_decide",
        route_after_decide,
        {
            "execute_tool": "execute_tool",
            "end": END,
        },
    )
    graph.add_conditional_edges(
        "execute_tool",
        route_after_tool,
        {
            "llm_decide": "llm_decide",
            "end": END,
        },
    )
    return graph


def _is_retryable_tool_error(exc: BaseException) -> bool:
    return isinstance(exc, (ValueError, KeyError, TypeError, IndexError))


def _tool_error_retry_payload(
    agent: str,
    step: int,
    max_steps: int,
    tool_name: str,
    arguments: Dict[str, Any],
    error: BaseException,
    retry_attempt: int,
    max_retries: int,
    retry_temperature: float,
) -> Dict[str, Any]:
    return {
        "task": "tool_retry_correction",
        "agent_name": agent,
        "step": step,
        "max_steps": max_steps,
        "failed_tool_call": {
            "name": tool_name,
            "arguments": arguments,
        },
        "retry_attempt": retry_attempt,
        "max_retry_attempts": max_retries,
        "retry_temperature": retry_temperature,
        "error": {
            "type": type(error).__name__,
            "message": str(error),
        },
        "instruction": (
            "The previous tool call failed before producing a usable observation. "
            "Treat this as a recoverable tool-call formatting or argument-selection error. "
            "Use the AVAILABLE_TOOLS schema in the system message to correct the tool name, "
            "argument names, or argument values. Do not repeat the same failing arguments. "
            "Return exactly one ReAct JSON object: either a corrected tool call or finish "
            "if the existing observations are sufficient."
        ),
    }
