from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Optional

from chiplet_tuner.core.io import load_metrics, read_json
from chiplet_tuner.core.search_space import (
    DEFAULT_HARDWARE_SEARCH_SPACE,
    fixed_hardware_params,
    system_param_keys,
)
from chiplet_tuner.core.schemas import EvaluationResult


class MappingEvaluationUnavailable(RuntimeError):
    """The configured mapping search produced no valid measured design."""

    def __init__(self, run_dir: Path):
        self.run_dir = Path(run_dir)
        super().__init__(f"No valid mapping found; inspect {self.run_dir / 'search.log'}")


class SimulatorAdapter(ABC):
    """Interface implemented by simulator-specific adapters."""

    @abstractmethod
    def evaluate(self, hardware: Dict[str, Any], iteration: int) -> EvaluationResult:
        """Run one hardware design point and return normalized artifacts."""

    @abstractmethod
    def schema(self) -> Dict[str, Any]:
        """Return hardware search constraints and required simulator fields."""


class GenericFileEvaluationAdapter(SimulatorAdapter):
    """Read precomputed evaluation files using the normalized detail schema."""

    def __init__(
        self,
        run_dir: Path,
        hardware_path: Path,
        metrics_path: Optional[Path] = None,
        latency_detail_path: Optional[Path] = None,
        energy_detail_path: Optional[Path] = None,
        mc_detail_path: Optional[Path] = None,
        search_space: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.run_dir = run_dir.resolve()
        self.hardware_path = hardware_path.resolve()
        self.metrics_path = (metrics_path or _preferred_artifact(self.run_dir, "search_res.csv", "exec_res.csv")).resolve()
        self.latency_detail_path = (
            latency_detail_path
            or _preferred_artifact(self.run_dir, "search_latency_detail.json", "exec_latency_detail.json")
        ).resolve()
        self.energy_detail_path = (
            energy_detail_path
            or _preferred_artifact(self.run_dir, "search_energy_detail.json", "exec_energy_detail.json")
        ).resolve()
        self.mc_detail_path = (
            mc_detail_path
            or _preferred_artifact(self.run_dir, "search_mc_detail.json", "exec_mc_detail.json")
        ).resolve()
        self.search_space = search_space or DEFAULT_HARDWARE_SEARCH_SPACE

    def evaluate(self, hardware: Dict[str, Any], iteration: int = 0) -> EvaluationResult:
        if not self.metrics_path.exists():
            raise FileNotFoundError(f"Metrics file not found: {self.metrics_path}")
        if not self.latency_detail_path.exists():
            raise FileNotFoundError(f"Latency detail file not found: {self.latency_detail_path}")
        if not self.energy_detail_path.exists():
            raise FileNotFoundError(f"Energy detail file not found: {self.energy_detail_path}")
        detail_files = {
            "latency": self.latency_detail_path,
            "energy": self.energy_detail_path,
        }
        if self.mc_detail_path.exists():
            detail_files["monetary_cost"] = self.mc_detail_path
        return EvaluationResult(
            run_dir=self.run_dir,
            hardware=hardware,
            metrics=load_metrics(self.metrics_path),
            detail_files=detail_files,
            raw_files={"metrics": self.metrics_path, "hardware": self.hardware_path},
        )

    def load_hardware(self) -> Dict[str, Any]:
        return read_json(self.hardware_path)

    def schema(self) -> Dict[str, Any]:
        system_params = system_param_keys(self.search_space)
        return {
            "adapter": "generic-files",
            "design_space_kind": "explicit_count_per_chiplet",
            "search_space": self.search_space,
            "tunable_hardware_fields": {
                "chiplet_count": ["num_chiplets"],
                "per_chiplet_architecture": ["compute_units", "buffer_size", "chiplet_type"],
                "system_params": system_params,
            },
            "fixed_hardware_fields": fixed_hardware_params(self.search_space),
            "derived_hardware_fields": ["chip_y", "chip_x", "chiplets.macs"],
            "detail_schema": {
                "latency": "JSON keyed by resource/core, values include layerID, layerName, latencyBegin, latencyEnd, calcTime, nocTime, dramTime",
                "energy": "JSON keyed by resource/core, values include layerID, layerName, energy, calcEnergy, ubufEnergy, nocEnergy, dramEnergy",
                "monetary_cost": "Optional JSON with simulator monetary-cost breakdown, usually emitted as search_mc_detail.json",
            },
        }


def _preferred_artifact(run_dir: Path, primary: str, legacy: str) -> Path:
    primary_path = run_dir / primary
    if primary_path.exists():
        return primary_path
    return run_dir / legacy
