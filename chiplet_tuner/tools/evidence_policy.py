"""Tool-level evidence restriction for the scalar-information ablation."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from chiplet_tuner.core.io import read_json, write_json
from chiplet_tuner.core.schemas import EvaluationResult, ToolResult, ToolSpec
from chiplet_tuner.tools.analysis_tools import AnalysisToolbox, ToolContext


SCALAR_TOOLS = frozenset({
    "compare_search_states", "select_analysis_base", "summarize_metrics",
    "summarize_hardware_config", "modify_hardware_parameter", "step_hardware_parameter",
})

SCALAR_EVIDENCE_PROMPT = """
SCALAR-INFORMATION ABLATION (overrides report-analysis requirements):
You have only evaluated hardware configurations, aggregate latency/energy/MC,
objective values, legal parameter candidates, and this run's own scalar history.
Layer timing, traffic, utilization, operator rankings, timelines, cost breakdowns,
and original reports are unavailable. Never invent report observations or layer IDs.
Reason about parameter interactions and metric changes from the permitted data.
For model-level final output use candidate_layers=[] and an aggregate summary.
For layer-level output use layer_diagnoses=[], dominant_root_cause="unknown",
and zero root_cause_summary ratios; recommended_focus may list hardware parameters.
Do not treat an architectural prior as a measured per-layer root cause.
The solution still uses the ordinary final hardware-edit tool_calls protocol.
""".strip()


class ScalarEvidenceToolbox(AnalysisToolbox):
    """Allow aggregate/configuration tools and reject detailed-report calls."""

    evidence_mode = "scalar"
    evidence_prompt = SCALAR_EVIDENCE_PROMPT

    def specs(self) -> List[ToolSpec]:
        return [spec for spec in super().specs() if spec.name in SCALAR_TOOLS]

    def run(
        self,
        tool_name: str,
        context: ToolContext | EvaluationResult,
        output_dir: Optional[Path] = None,
        arguments: Optional[Dict[str, Any]] = None,
    ) -> ToolResult:
        if tool_name not in SCALAR_TOOLS:
            raise ValueError(f"Tool {tool_name!r} is unavailable under scalar evidence policy.")
        return super().run(tool_name, context, output_dir=output_dir, arguments=arguments)


def configure_evidence_policy(
    mode: str,
    output_dir: Path,
    history_db: Path,
    import_history: Optional[str] = None,
    agent_mode: str = "multi",
) -> AnalysisToolbox:
    """Prevent mode changes or importing report-bearing history into scalar runs."""
    if mode not in {"full", "scalar"}:
        raise ValueError(f"Unknown evidence mode: {mode!r}")
    policy_path = output_dir / "information_policy.json"
    if policy_path.exists():
        previous = read_json(policy_path)
        if previous.get("evidence_mode") != mode:
            raise ValueError("Cannot change evidence mode when resuming/reusing an output directory.")
    if mode == "scalar":
        if agent_mode != "multi":
            raise ValueError("The scalar evidence control currently keeps the multi-agent organization fixed.")
        if import_history:
            raise ValueError("Scalar evidence runs cannot import potentially report-bearing history.")
        if history_db.resolve() != (output_dir / "history_vector_store.sqlite").resolve():
            raise ValueError("Scalar evidence runs require their own output-local history database.")
        if history_db.exists() and not policy_path.exists():
            raise ValueError("Existing history has no scalar evidence provenance; use a new output directory.")
    if not policy_path.exists():
        write_json(policy_path, {
            "evidence_mode": mode,
            "allowed_tools": sorted(SCALAR_TOOLS) if mode == "scalar" else "all ordinary role-scoped tools",
            "history": "run-local, no external imports" if mode == "scalar" else "ordinary configuration",
            "scalar_control_changes_action_space": False,
            "notes": "This controls evidence access, not matching action granularity to traditional baselines.",
        })
    return ScalarEvidenceToolbox() if mode == "scalar" else AnalysisToolbox()
