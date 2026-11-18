"""Three-role, report-accessible controller for the AgenticDSE adaptation."""
from __future__ import annotations

import time
import json
import os
from pathlib import Path

from chiplet_tuner.core.io import write_json
from chiplet_tuner.llm.tracing import LLMTraceRecorder
from baseline.agentic_region import validate_selections
from chiplet_tuner.core.search_coverage import summarize_parameter_coverage
from chiplet_tuner.core.report_semantics import TIMING_FIELD_DEFINITIONS
from chiplet_tuner.agents.analysis_policy import COMPASS_MODEL_GUIDANCE


class AgenticController:
    @staticmethod
    def _has_analysis(response):
        value = response.get("analysis")
        return (isinstance(value, str) and bool(value.strip())) or (
            isinstance(value, dict) and bool(value)
        )

    def __init__(self, client, output_dir, model, max_queries=5, review_attempts=2):
        self.client = client
        self.output_dir = Path(output_dir)
        self.model = model
        self.max_queries = max_queries
        self.review_attempts = review_attempts
        if max_queries < 1 or review_attempts < 1:
            raise ValueError("Query and review budgets must be positive")
        self.client.trace_recorder = LLMTraceRecorder(self.output_dir / "llm_trace")
        self.client.trace_enabled = True

    def config(self):
        config = getattr(self.client, "config", None)
        settings = {key: getattr(config, key, None) for key in (
            "temperature", "set_temperature", "max_tokens", "reasoning_effort", "timeout",
        )}
        return {"kind": "three_role_llm", "model": self.model, "llm_settings": settings,
                "analysis_policy": os.environ.get("CHIPLET_TUNER_ANALYSIS_POLICY", "standard"),
                "max_queries": self.max_queries, "review_attempts": self.review_attempts}

    def _call(self, role, instruction, payload):
        request = {"agent_name": role, "task": "agentic_phase_" + role, **payload}
        prompt = (
            "You are the " + role + " in an AgenticDSE adaptation for Compass. "
            "Minimize the supplied common scalar objective, including its compute-budget penalty. "
            "Use measured reports to guide phase-wise candidate-space changes. "
            "Do not fabricate evaluations or treat a hypothesis as a measured effect. "
            "Return one valid JSON object, without XML tags or prose outside the JSON. " + instruction
        )
        if os.environ.get("CHIPLET_TUNER_ANALYSIS_POLICY") in {"objective_guided", "decision_first"}:
            prompt += (
                " Use parameter_coverage to distinguish untested values from failed "
                "directions. Reconsider fields frozen without matched evidence when "
                "a region stagnates. The scalar product can improve when one component "
                "worsens; do not require simultaneous improvement of every metric or "
                "invent hard constraints beyond the supplied legal space. "
                + COMPASS_MODEL_GUIDANCE
            )
        for attempt in range(max(4, int(self.review_attempts))):
            try:
                return self.client.complete_json(prompt, request)
            except (json.JSONDecodeError, ValueError, RuntimeError) as exc:
                if attempt + 1 >= max(4, int(self.review_attempts)):
                    raise
                request = dict(request, json_correction={
                    "attempt": attempt + 1, "error": str(exc),
                    "instruction": "Repeat the response using the requested JSON schema only.",
                })

    @staticmethod
    def _compact_space(domains, current):
        # Per-chip domains repeat; enumerate the field pattern once, preserving overrides.
        system = {k: v for k, v in domains.items() if not k.startswith("chiplets.")}
        chip = {k.split(".")[-1]: v for k, v in domains.items() if k.startswith("chiplets.0.")}
        overrides = {k: v for k, v in current.items() if v != domains[k]}
        return {"system_domains": system, "per_chiplet_domains": chip,
                "chiplet_field_format": "chiplets.<zero_based_index>.<parameter>",
                "field_examples": {"chiplets.0.compute_units": chip.get("compute_units", []),
                                   "chiplets.0.type": chip.get("type", [])},
                "maximum_chiplet_slots": max(domains["num_chiplets"]),
                "current_restrictions": overrides,
                "update_rule": "Return field-to-candidate-list selections. Omitted fields keep their "
                               "previous range. To expand a field, select from its full legal domain."}

    @staticmethod
    def _query(request, history):
        if not isinstance(request, dict):
            raise ValueError("Query must be an object")
        iteration = request.get("iteration")
        if type(iteration) is not int:
            raise ValueError("Query iteration must be an integer")
        record = next((r for r in history if r.iteration == iteration), None)
        if record is None:
            raise ValueError("Query refers to an unknown evaluation")
        report = request.get("report")
        if report == "hardware":
            return {"hardware": record.hardware, "metrics": record.metrics}
        files = {"latency": "search_latency_detail.json", "energy": "search_energy_detail.json",
                 "monetary_cost": "search_mc_detail.json"}
        if report not in files:
            raise ValueError("Report must be hardware, latency, energy, or monetary_cost")
        offset = request.get("offset", 0)
        if type(offset) is not int or offset < 0:
            raise ValueError("Report offset must be a nonnegative integer")
        path = record.run_dir / files[report]
        if not path.is_file():
            return {"available": False, "report": report}
        content = path.read_text(encoding="utf-8")
        page = content[offset:offset + 16000]
        return {"available": True, "report": report, "offset": offset, "text": page,
                "next_offset": offset + len(page) if offset + len(page) < len(content) else None}

    def select_region(self, phase, domains, current, history, reason="scheduled"):
        started = time.perf_counter()
        best = min(history, key=lambda r: r.objective)
        context = {"phase": phase, "phase_trigger": reason, "space": self._compact_space(domains, current),
                   "report_field_definitions": {
                       **TIMING_FIELD_DEFINITIONS,
                       "cost_compute": "manufacturing cost of compute dies, across all dataflow types",
                       "cost_IO": "IO die manufacturing cost plus bandwidth-dependent DRAM cost",
                       "cost_os_overall": "package substrate cost; os here does NOT mean output-stationary dataflow",
                       "os_area": "package substrate area, derived from total die area",
                       "mc": "hardware monetary cost, not LLM API charges",
                   },
                   "incumbent_hardware": best.hardware, "incumbent_metrics": best.metrics,
                   "evaluations": [{"iteration": r.iteration, "metrics": r.metrics} for r in history]}
        if os.environ.get("CHIPLET_TUNER_ANALYSIS_POLICY") in {"objective_guided", "decision_first"}:
            context["parameter_coverage"] = summarize_parameter_coverage(
                [{"iteration": r.iteration, "hardware": r.hardware, "objective": r.objective}
                 for r in history],
                {k: v for k, v in domains.items() if not k.startswith("chiplets.")},
                best.hardware,
            )
        directive = self._call("Orchestrator", 'Return {"directive": "phase objective and evidence to inspect"}.', context)
        observations = []
        feedback = None
        review = None
        analysis = None
        analyst_corrections = []
        for attempt in range(self.review_attempts):
            for query_index in range(self.max_queries + 1):
                corrections = []
                for correction_attempt in range(self.review_attempts):
                    analysis = self._call("ArchitectureAnalyst",
                        'Return one JSON object, not XML or tool_use tags. '
                        'Use either {"query": {"iteration": 0, "report": "hardware|latency|energy|monetary_cost", '
                        '"offset": 0}} or {"analysis": "evidence and proposed subspace changes"}. '
                        'Analysis may be nonempty text or a nonempty structured evidence object; '
                        'it is reviewed, not executed as hardware actions. '
                        'The outer query key is required for queries. '
                        'Reports are paged raw evaluator outputs; follow next_offset to read further. '
                        'When queries_remaining is zero, return analysis.',
                        {**context, "directive": directive, "observations": observations,
                         "review_feedback": feedback, "queries_remaining": self.max_queries - query_index,
                         "validation_feedback": corrections})
                    has_analysis = self._has_analysis(analysis)
                    has_query = isinstance(analysis.get("query"), dict)
                    if (has_analysis and "query" not in analysis) or (
                        has_query and "analysis" not in analysis and query_index < self.max_queries
                    ):
                        break
                    failure = {"response": analysis, "error":
                        "Return a nonempty analysis, or a query object while queries_remaining > 0; "
                        "use exactly one of the two outer keys."}
                    corrections.append(failure)
                    analyst_corrections.append(failure)
                    if correction_attempt + 1 == self.review_attempts:
                        raise ValueError("Analyst response invalid after bounded format correction")
                if self._has_analysis(analysis):
                    break
                if "query" not in analysis or query_index == self.max_queries:
                    raise ValueError("Analyst must finish with a nonempty analysis within the query budget")
                try:
                    result = self._query(analysis["query"], history)
                except ValueError as exc:
                    result = {"error": str(exc)}
                observations.append({"query": analysis["query"], "result": result})
            review = self._call("Orchestrator",
                'Review strategic alignment and evidence. Return {"approved": true, "feedback": "..."}; '
                'use false to request a revised analysis.',
                {**context, "directive": directive, "analysis": analysis, "observations": observations})
            if type(review.get("approved")) is not bool:
                raise ValueError("Review approved field must be boolean")
            if review["approved"]:
                break
            feedback = review
        if not review["approved"]:
            raise ValueError("AgenticDSE phase analysis rejected after review budget")
        engineer_instruction = (
            'Translate the approved analysis to a phase subspace. Return {"base": "incumbent|previous", "selections": '
            '{"field_name": ["legal_candidate_values"]}}. Use actual JSON numeric/string types. '
            'base=incumbent fixes the chiplet count, all active chiplet parameters and system parameters '
            'at the incumbent BEFORE applying selections. Use it for type-only or other controlled sweeps. '
            'base=previous retains the prior phase domain BEFORE applying selections; omitted fields remain '
            'searchable over their prior domain, NOT fixed at incumbent values. '
            'No new values, aliases, hardware objects or action lists. '
            'compute_units, buffer_size and type are per-chip fields, never top-level fields. '
            'Use e.g. chiplets.0.compute_units; enumerate separate chiplet keys when changing several chips. '
            'Total compute scale is sum(active compute_units)/512; if num_chiplets is varied, include matching '
            'per-chiplet compute_units candidates so target compute scale (compute_budget_penalty=1) remains reachable.'
        )
        corrections = []
        for attempt in range(self.review_attempts):
            decision = self._call("OptimizationEngineer", engineer_instruction,
                {**context, "directive": directive, "analysis": analysis, "review": review,
                 "validation_feedback": corrections})
            try:
                if decision.get("base") not in {"incumbent", "previous"}:
                    raise ValueError("Engineer base must be incumbent or previous")
                selections = decision.get("selections")
                validate_selections(domains, selections)
                break
            except ValueError as exc:
                corrections.append({"decision": decision, "error": str(exc)})
                if attempt + 1 == self.review_attempts:
                    raise
        effective = {}
        if decision["base"] == "incumbent":
            for field in domains:
                if field.startswith("chiplets."):
                    _, index, parameter = field.split(".")
                    if int(index) < best.hardware["num_chiplets"]:
                        effective[field] = [best.hardware["chiplets"][int(index)][parameter]]
                else:
                    effective[field] = [best.hardware[field]]
        effective.update(selections)
        write_json(self.output_dir / f"phase_{phase:03d}.json", {
            "phase": phase, "phase_trigger": reason, "directive": directive, "analysis": analysis,
            "review": review, "decision": decision, "observations": observations,
            "engineer_corrections": corrections,
            "analyst_corrections": analyst_corrections,
            "effective_selections": effective,
            "elapsed_seconds": time.perf_counter() - started,
        })
        return effective
