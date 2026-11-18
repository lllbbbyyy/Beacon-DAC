"""Opt-in analysis budget and role contracts for development pilots; ordinary runs are unchanged."""
import os

from chiplet_tuner.llm.prompts import (
    LAYER_LEVEL_ANALYSIS_PROMPT,
    MODEL_LEVEL_ANALYSIS_PROMPT,
)


POLICY_ENV = "CHIPLET_TUNER_ANALYSIS_POLICY"
FOCUSED_STEPS = {"model_level": 5, "layer_level": 3, "solution_generation": 2}
SYSTEM_TRADEOFF_GUIDANCE = (
    "SYSTEM-LEVEL ANALYSIS IS VALID: A measured latency hotspot is not a prerequisite "
    "for exploring a latency-energy-cost tradeoff. Use aggregate metrics and parameter "
    "coverage to identify alternatives, then query detailed evidence when it can "
    "distinguish those alternatives. Do not manufacture a bottleneck layer to justify "
    "a system parameter change. In the final model output, candidate_layers may be [] "
    "when no layer-specific claim is made; explain the observed system-level tradeoff "
    "and proposed test in summary. In the final layer output, layer_diagnoses may be []; "
    "in that case use dominant_root_cause='unknown' and zero root_cause_summary ratios, "
    "and describe the tradeoff and unresolved question in retrieval_description and "
    "recommended_focus. This replaces the requirement for a nonempty layer list, "
    "not the requirement for measured evidence supporting factual claims."
)

DECISION_FIRST_ROLE_PROMPTS = {
    "model_level": """
Role: model-level tradeoff and evidence-scoping analyst for multi-chiplet accelerator DSE.

Mission: select the right evaluated analysis base, inspect aggregate objective metrics and parameter coverage, and determine the highest-leverage next tradeoff for minimizing latency * energy * monetary_cost * compute_budget_penalty. Request global or operator-level reports ONLY when they can distinguish competing hypotheses; output candidate_layers=[] whenever a system-level granularity, bandwidth, or chiplet-consolidation tradeoff needs no per-layer bottleneck claim.

Decision-first protocol:
1. Call compare_search_states (and include_history=true if current regressed or trend is unclear) to inspect current/previous/best metrics, change_vs_edit_base, and parameter_coverage; then call select_analysis_base.
2. Check whether untested high-leverage system or cost directions exist (micro_batch, tensor_parall, over-provisioned nop_bw/dram_bw inflating MC, or num_chiplets consolidation at target compute scale). When summarize_monetary_cost is in AVAILABLE_TOOLS, use it early to inspect cost_os_overall (package/substrate), cost_IO (DRAM/IO), and cost_compute shares before proposing resource/bandwidth growth or delaying cost trims.
3. If a system-level move is the clearest next test and per-layer diagnosis would not change that choice, finish tool use promptly and output candidate_layers=[].
4. Once high-leverage system granularity and bandwidth/cost trims around the best base have been tested (or whenever choosing heterogeneous chiplet_type ws vs. os, localized per-chiplet compute_units/buffer_size, or diagnosing a regression), actively pivot to operator-level evidence: query summarize_operator_groups or summarize_layer_rank_views (when available) and pass 2-4 decision-relevant candidate_layers (covering dominant projection/FFN and attention stages) to the layer-level agent so post-TP GEMM shapes and chiplet placements guide heterogeneous specialization.
""".strip(),
    "layer_level": """
Role: conditional operator-level and microarchitectural evidence specialist.

Mission: resolve only the operator-level questions that can change the downstream parameter update, and package a grounded bottleneck/tradeoff state for RAG retrieval and solution generation.

Decision-first protocol:
1. If upstream candidate_layers is [] and the model-level summary states a clear system-level latency-energy-MC tradeoff with no unresolved operator-level question, finish tool use immediately (action="finish") without inventing layer queries or diagnoses.
2. When candidate_layers is non-empty (or when distinguishing heterogeneous WS/OS ratios, per-chiplet compute/buffer sizing, or stage placement requires operator evidence and layer tools are in AVAILABLE_TOOLS), call inspect_layer_details (with include_fields=["basic","shares","timing","breakdown","placement","features"]) on the compact candidate set to inspect post-TP GEMM M/N/K shapes, operand types, and chiplet placement indices.
3. Keep factual measured root causes separate from untested hypotheses; if no per-layer diagnosis is made, output layer_diagnoses=[], dominant_root_cause="unknown", and zero root_cause_summary ratios.
""".strip(),
    "solution_generation": """
Role: solution-generation agent and joint hardware/mapping configuration synthesizer.

Mission: synthesize one legal next configuration update on proposal_base_hardware to minimize latency * energy * monetary_cost * compute_budget_penalty. The final solution JSON provides the canonical hardware-edit tool_calls.

Decision-first & cost discipline:
- Rely on the supplied bottleneck/tradeoff state, proposal_base, parameter_coverage, and retrieved RAG cases. Do NOT repeat global timelines, operator rankings, or per-layer inspections in ReAct unless summarize_hardware_config(include_full_hardware=true) or summarize_monetary_cost is required to finalize exact arguments.
- Weigh latency, energy, MC, and compute_budget_penalty jointly. When MC is elevated and lower dram_bw or nop_bw values remain untested on the current best granularity/parallelism base, prioritize testing cost-reducing bandwidth trims over speculative buffer or bandwidth expansions.
- Use measured GEMM shape and placement evidence (when present in layer_diagnoses) to target heterogeneous chiplet_type (ws vs. os) or indexed per-chiplet edits rather than blind all-chiplet flips.
""".strip(),
}


DECISION_FIRST_MODEL_ANALYSIS_PROMPT = """
Final output contract: decision-first model-level tradeoff and optional candidate-layer handoff.

SYSTEM-LEVEL ANALYSIS IS VALID: Anchor on the selected analysis_base and evaluate latency, energy, monetary cost (MC), and compute_budget_penalty jointly. Set candidate_layers to [] when no layer-specific diagnosis is needed to choose the next system-level granularity, bandwidth/cost trim, or consolidation move, and state the measured tradeoff in summary and notes.global_findings. Populate candidate_layers (2-4 items with quantitative tool evidence) ONLY when operator-level GEMM shapes, reuse, or chiplet placement can decide heterogeneous ws/os allocation, per-chiplet sizing, or parallelism bottlenecks.

Return strict JSON matching this schema:
```json
{
  "analysis_base": {"source": "current|previous|best|iteration", "iteration": 0, "reason": "..."},
  "summary": "concise paragraph stating measured L/E/MC/coverage facts and the candidate tradeoff or local question",
  "candidate_layers": [
    {
      "layer_id": 0,
      "layer_name": "...",
      "operator_group": "ffn|attention|qkv_projection|...",
      "concern_types": ["latency|energy|monetary_cost|mixed"],
      "rank_metric": "latency_sum|energy|critical_end|...",
      "rank_value": 0.0,
      "evidence": ["short quantitative evidence from tool outputs"],
      "confidence": "high|medium|low"
    }
  ],
  "notes": {
    "selected_bottleneck_objective": "latency|energy|monetary_cost|mixed|unknown",
    "selected_rank_views": ["latency_sum|energy|critical_end|compute|memory|communication|buffer"],
    "global_findings": ["..."],
    "tool_confidence": "high|medium|low"
  }
}
```
""".strip()

DECISION_FIRST_LAYER_ANALYSIS_PROMPT = """
Final output contract: decision-first bottleneck or system-tradeoff state.

SYSTEM-LEVEL ANALYSIS IS VALID: Convert the upstream handoff and any inspected operator evidence into a grounded state for RAG retrieval and solution generation toward minimizing latency * energy * monetary_cost * compute_budget_penalty. When no layer-level inspection was needed (e.g., candidate_layers is [] for a system/cost tradeoff), return layer_diagnoses: [], dominant_root_cause: "unknown", and root_cause_summary: {"compute": 0.0, "memory": 0.0, "communication": 0.0, "buffer": 0.0}; describe the system/cost tradeoff and untested parameter question in retrieval_description and recommended_focus. When candidate layers were inspected with tool evidence, populate layer_diagnoses with measured timing/energy shares, post-TP GEMM M/N/K shapes and operand reuse, and chiplet placement facts.

Return strict JSON matching this schema:
```json
{
  "primary_impact": "latency|energy|monetary_cost|mixed|unknown",
  "dominant_root_cause": "compute|memory|communication|buffer|scheduling|imbalance|mixed|unknown",
  "layer_diagnoses": [
    {
      "layerID": 0,
      "layerName": "...",
      "operator_group": "ffn|attention|qkv_projection|...",
      "impact_types": ["latency|energy|monetary_cost|mixed"],
      "root_causes": ["compute|memory|communication|buffer|scheduling|imbalance|unknown"],
      "dominant_root_cause": "compute|memory|communication|buffer|scheduling|imbalance|mixed|unknown",
      "root_cause_ratios": {"compute": 0.0, "memory": 0.0, "communication": 0.0, "buffer": 0.0},
      "load_features": {},
      "diagnosis": "concrete measured diagnosis including shapes/placement when inspected"
    }
  ],
  "retrieval_description": "compact retrieval key covering L/E/MC impact, active hardware/tradeoff context, and measured layer/shape causes if any",
  "root_cause_summary": {"compute": 0.0, "memory": 0.0, "communication": 0.0, "buffer": 0.0},
  "recommended_focus": ["legal parameter names to tune"],
  "notes": {"reasoning": ["..."]}
}
```
""".strip()


def allows_system_analysis(mode=None):
    resolved = os.environ.get(POLICY_ENV, "standard") if mode is None else mode
    return resolved == "decision_first"


def analysis_role_prompt(agent_name, default_prompt, mode=None):
    if allows_system_analysis(mode) and agent_name in DECISION_FIRST_ROLE_PROMPTS:
        return DECISION_FIRST_ROLE_PROMPTS[agent_name]
    return default_prompt


def analysis_output_prompt(prompt, mode=None):
    if not allows_system_analysis(mode):
        return prompt
    if prompt == MODEL_LEVEL_ANALYSIS_PROMPT:
        return DECISION_FIRST_MODEL_ANALYSIS_PROMPT
    if prompt == LAYER_LEVEL_ANALYSIS_PROMPT:
        return DECISION_FIRST_LAYER_ANALYSIS_PROMPT
    return prompt + "\n\n" + SYSTEM_TRADEOFF_GUIDANCE


def final_task_instruction(task, fallback="", mode=None):
    if allows_system_analysis(mode):
        decision_first_instructions = {
            "model_level_analysis": (
                "Synthesize the decision-first model-level tradeoff and optional candidate-layer handoff "
                "(use candidate_layers=[] when no layer-specific ambiguity needs diagnosis)."
            ),
            "layer_level_analysis": (
                "Synthesize the decision-first bottleneck or system-tradeoff state for RAG retrieval and "
                "solution generation (use layer_diagnoses=[], dominant_root_cause='unknown', and zero "
                "ratios when no layer diagnosis is claimed)."
            ),
            "solution_generation": "Produce the final legal hardware-update proposal for the next simulator evaluation.",
            "single_agent_analysis": "Produce the single-agent diagnosis or system-tradeoff state for retrieval and proposal generation.",
            "single_agent_solution_generation": "Produce the single-agent legal hardware-update proposal for the next simulator evaluation.",
        }
        if task in decision_first_instructions:
            return decision_first_instructions[task]
    instructions = {
        "model_level_analysis": "Produce the final model-level bottleneck candidate list for handoff to the layer-level agent.",
        "layer_level_analysis": "Produce the final layer-level bottleneck state for RAG retrieval and solution generation.",
        "solution_generation": "Produce the final legal hardware-update proposal for the next simulator evaluation.",
        "single_agent_analysis": "Produce the single-agent model/layer diagnosis used for retrieval and proposal generation.",
        "single_agent_solution_generation": "Produce the single-agent legal hardware-update proposal for the next simulator evaluation.",
    }
    return instructions.get(task, fallback or "Produce the final strict JSON answer.")


COMPASS_MODEL_GUIDANCE = (
    "Compass model semantics: DRAM energy is access volume times per-access cost; "
    "inter-chip communication energy is transferred hop-volume times per-hop cost. "
    "Increasing bandwidth alone does not directly reduce these energy terms. A changed "
    "mapping can change traffic, so assess that separately. NoP bandwidth contributes "
    "to compute-die, D2D and IO area and consequently package cost; DRAM bandwidth "
    "contributes to interface area and memory cost. Neither expansion is cost-neutral. "
    "A large energy share identifies traffic cost, not necessarily a bandwidth latency "
    "bottleneck. Compare measured timing and monetary-cost evidence before proposing "
    "bandwidth growth. Distinguish a predicted effect from a measured result."
)
FOCUSED_GUIDANCE = {
    "model_level": (
        "Select the analysis base using the ordinary protocol. Then query the one or two "
        "global views needed to identify a consequential tradeoff. Do not perform detailed "
        "per-layer diagnosis here; pass selected observations and unresolved questions to "
        "the layer agent. A large latency component alone does not determine the best "
        "latency-energy-monetary-cost modification."
    ),
    "layer_level": (
        "Inspect only the candidate region needed to test the upstream hypothesis. "
        "Use grouped queries where supported. Distinguish observations from explanations; "
        "record an unresolved cause as unknown rather than inventing evidence. Finish once "
        "you can identify a specific parameter tradeoff to test."
    ),
    "solution_generation": (
        "Use the supplied bottleneck state, base hardware and retrieved measurements. "
        "Do not repeat global timelines, operator rankings or detailed layer diagnosis "
        "unless the handoff leaves a specific decision-changing question unanswered. "
        "Query that missing evidence only. Compare the expected directions of latency, "
        "energy and monetary cost; observations support a hypothesis, not a guaranteed gain. "
        "The final output still uses the ordinary hardware-edit protocol."
    ),
}


def resolve_analysis_policy(agent_name, max_steps, mode=None):
    mode = os.environ.get(POLICY_ENV, "standard") if mode is None else mode
    if mode not in {"standard", "focused", "objective_guided", "decision_first"}:
        raise ValueError(f"Unknown analysis policy: {mode!r}")
    if mode in {"objective_guided", "decision_first"}:
        return max_steps, (
            "OBJECTIVE-GUIDED DEVELOPMENT: Use compare_search_states to check parameter "
            "coverage before generalizing past failures. Distinguish increase from decrease "
            "and joint changes from matched single-parameter comparisons. First identify "
            "a candidate tradeoff in latency, energy and monetary cost; use available "
            "reports only to resolve a question that could change that choice. Explain "
            "the predicted directions of all three metrics, without claiming unmeasured "
            "improvements. Do not assume an untested value is ineffective. "
            + COMPASS_MODEL_GUIDANCE
            + (" " + SYSTEM_TRADEOFF_GUIDANCE if mode == "decision_first" else "")
        )
    if mode == "standard" or agent_name not in FOCUSED_STEPS:
        return max_steps, ""
    return min(max_steps, FOCUSED_STEPS[agent_name]), (
        "FOCUSED ANALYSIS PILOT: " + FOCUSED_GUIDANCE[agent_name] + " " + COMPASS_MODEL_GUIDANCE
    )
