from __future__ import annotations

import copy
import csv
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Protocol, Sequence, Set

from baseline.actions import ActionCandidate, hardware_fingerprint
from chiplet_tuner.core.io import write_json
from chiplet_tuner.core.objective import metrics_with_objective
from chiplet_tuner.simulators.base import MappingEvaluationUnavailable, SimulatorAdapter


@dataclass
class SearchRecord:
    iteration: int
    parent_iteration: Optional[int]
    action: Optional[Dict[str, Any]]
    hardware: Dict[str, Any]
    metrics: Dict[str, float]
    run_dir: Path
    elapsed_seconds: float
    completed_elapsed_seconds: float = 0.0
    within_time_budget: bool = True

    @property
    def objective(self) -> float:
        return float(self.metrics["objective"])

    @property
    def fingerprint(self) -> str:
        return hardware_fingerprint(self.hardware)

    def compact_dict(self) -> Dict[str, Any]:
        return {
            "iteration": self.iteration,
            "parent_iteration": self.parent_iteration,
            "action": self.action,
            "fingerprint": self.fingerprint,
            "metrics": self.metrics,
            "hardware": self.hardware,
            "run_dir": str(self.run_dir),
            "elapsed_seconds": self.elapsed_seconds,
            "completed_elapsed_seconds": self.completed_elapsed_seconds,
            "within_time_budget": self.within_time_budget,
        }


class SearchStrategy(Protocol):
    name: str

    def propose(
        self,
        history: Sequence[SearchRecord],
        forbidden_fingerprints: Set[str],
    ) -> ActionCandidate:
        ...

    def observe(
        self,
        parent: SearchRecord,
        result: SearchRecord,
        proposal: ActionCandidate,
    ) -> None:
        ...

    def finalize(self, history: Sequence[SearchRecord]) -> None:
        ...


class BaselineSearchRunner:
    """Evaluate proposals from a baseline under a shared budget and objective."""

    def __init__(
        self,
        simulator: SimulatorAdapter,
        strategy: SearchStrategy,
        initial_hardware: Mapping[str, Any],
        search_space: Mapping[str, Any],
        output_dir: Path,
        evaluations: int,
        time_budget_seconds: Optional[float] = None,
        started_at: Optional[float] = None,
    ) -> None:
        if evaluations < 0:
            raise ValueError("evaluations must be non-negative.")
        if time_budget_seconds is not None and (
            not math.isfinite(time_budget_seconds) or time_budget_seconds <= 0
        ):
            raise ValueError("time_budget_seconds must be finite and positive.")
        self.simulator = simulator
        self.strategy = strategy
        self.initial_hardware = copy.deepcopy(dict(initial_hardware))
        self.search_space = dict(search_space)
        self.output_dir = output_dir.resolve()
        self.evaluations = evaluations
        self.history: List[SearchRecord] = []
        self.failed_evaluations: List[Dict[str, Any]] = []
        self.time_budget_seconds = time_budget_seconds
        self.started_at = started_at
        self.stop_reason = "evaluation_limit"

    def run(self) -> Dict[str, Any]:
        if self.started_at is None:
            self.started_at = time.perf_counter()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._write_run_config()
        initial = self._evaluate(
            hardware=self.initial_hardware,
            iteration=0,
            parent_iteration=None,
            action=None,
        )
        self.history.append(initial)
        self._append_record(initial)
        self._write_best()

        for iteration in range(1, self.evaluations + 1):
            if self._budget_expired():
                self.stop_reason = "time_budget"
                break
            forbidden = {record.fingerprint for record in self.history}
            forbidden.update(record["fingerprint"] for record in self.failed_evaluations)
            proposal = self.strategy.propose(self.history, forbidden)
            if self._budget_expired():
                self.stop_reason = "time_budget"
                break
            parent = self._record_by_iteration(proposal.parent_iteration)
            write_json(self.output_dir / "pending_proposal.json", {
                "iteration": iteration,
                "hardware": proposal.hardware,
                "action": proposal.compact_dict(),
                "proposed_elapsed_seconds": self._elapsed(),
            })
            try:
                result = self._evaluate(
                    hardware=proposal.hardware,
                    iteration=iteration,
                    parent_iteration=proposal.parent_iteration,
                    action=proposal.compact_dict(),
                )
            except MappingEvaluationUnavailable as exc:
                self._record_failure(iteration, proposal.hardware, proposal.compact_dict(), exc)
                continue
            self.history.append(result)
            if result.within_time_budget:
                self.strategy.observe(parent, result, proposal)
            self._append_record(result)
            self._write_best()
            if self._budget_expired():
                self.stop_reason = "time_budget"
                break

        # Finish in-flight evaluations for reproducibility, but never score an overrun.
        if not self._budget_expired():
            self.strategy.finalize(self.history)
        else:
            self.stop_reason = "time_budget"
        summary = self._summary()
        write_json(self.output_dir / "summary.json", summary)
        self._write_csv()
        return summary

    def _evaluate(
        self,
        hardware: Mapping[str, Any],
        iteration: int,
        parent_iteration: Optional[int],
        action: Optional[Dict[str, Any]],
    ) -> SearchRecord:
        started = time.perf_counter()
        result = self.simulator.evaluate(copy.deepcopy(dict(hardware)), iteration)
        metrics = metrics_with_objective(result.metrics, result.hardware, self.search_space)
        completed_elapsed = self._elapsed()
        return SearchRecord(
            iteration=iteration,
            parent_iteration=parent_iteration,
            action=action,
            hardware=copy.deepcopy(result.hardware),
            metrics=metrics,
            run_dir=result.run_dir,
            elapsed_seconds=time.perf_counter() - started,
            completed_elapsed_seconds=completed_elapsed,
            within_time_budget=(
                self.time_budget_seconds is None or completed_elapsed <= self.time_budget_seconds
            ),
        )

    def _elapsed(self) -> float:
        assert self.started_at is not None
        return time.perf_counter() - self.started_at

    def _record_failure(self, iteration, hardware, action, error) -> Dict[str, Any]:
        failure = {
            "iteration": iteration, "status": "mapping_unavailable",
            "fingerprint": hardware_fingerprint(hardware), "hardware": hardware,
            "action": action, "run_dir": str(error.run_dir),
            "completed_elapsed_seconds": self._elapsed(),
            "within_time_budget": not self._budget_expired(),
        }
        self.failed_evaluations.append(failure)
        write_json(self.output_dir / "failed_evaluations.json", self.failed_evaluations)
        return failure

    def _budget_expired(self) -> bool:
        return self.time_budget_seconds is not None and self._elapsed() >= self.time_budget_seconds

    def _eligible_records(self) -> List[SearchRecord]:
        return [record for record in self.history if record.within_time_budget]

    def _append_record(self, record: SearchRecord) -> None:
        with (self.output_dir / "history.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record.compact_dict(), ensure_ascii=False, sort_keys=True) + "\n")

    def _write_best(self) -> None:
        eligible = self._eligible_records()
        if not eligible:
            return
        best = min(eligible, key=lambda record: record.objective)
        write_json(self.output_dir / "best_hardware.json", best.hardware)
        write_json(self.output_dir / "best_record.json", best.compact_dict())

    def _write_csv(self) -> None:
        metric_keys = sorted({key for record in self.history for key in record.metrics})
        fields = [
            "iteration",
            "parent_iteration",
            "action_id",
            "action_category",
            "fingerprint",
            "elapsed_seconds",
            "completed_elapsed_seconds",
            "within_time_budget",
            *metric_keys,
        ]
        with (self.output_dir / "history.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for record in self.history:
                row = {
                    "iteration": record.iteration,
                    "parent_iteration": record.parent_iteration,
                    "action_id": (record.action or {}).get("action_id", "initial"),
                    "action_category": (record.action or {}).get("category", "initial"),
                    "fingerprint": record.fingerprint,
                    "elapsed_seconds": record.elapsed_seconds,
                    "completed_elapsed_seconds": record.completed_elapsed_seconds,
                    "within_time_budget": record.within_time_budget,
                }
                row.update(record.metrics)
                writer.writerow(row)

    def _summary(self) -> Dict[str, Any]:
        eligible = self._eligible_records()
        best = min(eligible, key=lambda record: record.objective) if eligible else None
        initial = self.history[0]
        return {
            "method": self.strategy.name,
            "requested_evaluations": self.evaluations,
            "total_evaluations_including_initial": len(self.history),
            "failed_evaluations": len(self.failed_evaluations),
            "attempted_evaluations_including_initial": len(self.history) + len(self.failed_evaluations),
            "eligible_evaluations_including_initial": len(eligible),
            "time_budget_seconds": self.time_budget_seconds,
            "total_elapsed_seconds": self._elapsed(),
            "stop_reason": self.stop_reason,
            "initial_objective": initial.objective,
            "best_objective": best.objective if best else None,
            "best_iteration": best.iteration if best else None,
            "improvement_over_initial": initial.objective / best.objective if best else None,
            "best_metrics": best.metrics if best else None,
            "best_hardware_path": str(self.output_dir / "best_hardware.json") if best else None,
        }

    def _record_by_iteration(self, iteration: int) -> SearchRecord:
        for record in self.history:
            if record.iteration == iteration:
                return record
        raise KeyError(f"Unknown parent iteration: {iteration}")

    def _write_run_config(self) -> None:
        write_json(
            self.output_dir / "baseline_config.json",
            {
                "method": self.strategy.name,
                "strategy_config": self.strategy.config() if hasattr(self.strategy, "config") else {},
                "evaluations": self.evaluations,
                "time_budget_seconds": self.time_budget_seconds,
                "timing_policy": "inclusive_wall_clock_completed_evaluations_only",
                "search_space": self.search_space,
                "simulator_schema": self.simulator.schema(),
            },
        )
