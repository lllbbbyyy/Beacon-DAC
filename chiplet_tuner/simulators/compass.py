from __future__ import annotations

import copy
import hashlib
import subprocess
from pathlib import Path
from typing import Any, Dict

from chiplet_tuner.core.io import load_csv_first_row, read_json, write_json
from chiplet_tuner.core.search_space import (
    DEFAULT_HARDWARE_SEARCH_SPACE,
    fixed_hardware_params,
    system_param_keys,
)
from chiplet_tuner.core.schemas import EvaluationResult
from chiplet_tuner.simulators.base import MappingEvaluationUnavailable, SimulatorAdapter


def resolve_config_paths(
    config: Dict[str, Any],
    source_path: Path,
    target_dir: Path,
    compass_root: Path | None = None,
) -> Dict[str, Any]:
    """Resolve Compass config paths after copying into a run directory."""

    resolved = copy.deepcopy(config)
    source_dir = source_path.resolve().parent
    if resolved.get("request_trace_path"):
        request_path = Path(resolved["request_trace_path"])
        if not request_path.is_absolute():
            request_path = source_dir / request_path
        raw = request_path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != resolved.get("request_trace_sha256"):
            raise ValueError("Request trace hash missing or changed; freeze the workload before search")
        target_dir.mkdir(parents=True, exist_ok=True)
        snapshot = target_dir / "request_trace.json"
        snapshot.write_bytes(raw)
        resolved["request_trace_path"] = str(snapshot.resolve())
    model = resolved.get("model_info", {})
    if model.get("moe_routing_trace_path"):
        trace_path = Path(model["moe_routing_trace_path"])
        if not trace_path.is_absolute():
            trace_path = source_dir / trace_path
        raw = trace_path.read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        if digest != model.get("moe_routing_trace_sha256"):
            raise ValueError("MoE routing trace hash missing or changed; freeze the workload before search")
        target_dir.mkdir(parents=True, exist_ok=True)
        snapshot = target_dir / "moe_routing_trace.json"
        snapshot.write_bytes(raw)
        model["moe_routing_trace_path"] = str(snapshot.resolve())
    for key in ["req_generator_input_length_path", "req_generator_output_length_path"]:
        value = resolved.get(key)
        if isinstance(value, str) and value:
            candidate = Path(value)
            if not candidate.is_absolute():
                local_candidate = (source_dir / candidate).resolve()
                if local_candidate.exists() or compass_root is None:
                    candidate = local_candidate
                else:
                    candidate = (compass_root / candidate).resolve()
            resolved[key] = str(candidate)

    run_local_paths = {
        "best_mapping_save_path": "best_mapping.json",
        "search_process_save_path": "search_process.csv",
        "detail_latency_save_path": "search_latency_detail.json",
        "detail_energy_save_path": "search_energy_detail.json",
        "detail_mc_save_path": "search_mc_detail.json",
    }
    for key, filename in run_local_paths.items():
        if key not in resolved:
            continue
        if key.startswith("detail_") and not resolved.get(key):
            continue
        if key == "search_process_save_path" and not resolved.get(key):
            continue
        resolved[key] = str((target_dir / filename).resolve())
    return resolved


class CompassSimulatorAdapter(SimulatorAdapter):
    """Compass-specific adapter that returns simulator-independent artifacts."""

    def __init__(
        self,
        compass_root: Path,
        search_config_path: Path,
        output_root: Path,
        search_space: Dict[str, Any] | None = None,
        binary_path: Path | None = None,
    ) -> None:
        self.compass_root = compass_root.resolve()
        self.binary = binary_path.resolve() if binary_path else self.compass_root / "build" / "compass"
        self.search_config_path = search_config_path.resolve()
        self.output_root = output_root.resolve()
        self.search_space = search_space or DEFAULT_HARDWARE_SEARCH_SPACE

    def evaluate(self, hardware: Dict[str, Any], iteration: int) -> EvaluationResult:
        if not self.binary.exists():
            raise FileNotFoundError(f"Compass binary not found: {self.binary}")

        run_dir = self.output_root / f"iter_{iteration:03d}"
        run_dir.mkdir(parents=True, exist_ok=True)
        hardware_path = run_dir / "hardware.json"
        search_config_path = run_dir / "search_config.json"
        search_csv = run_dir / "search_res.csv"
        search_log = run_dir / "search.log"

        evaluated_hardware = copy.deepcopy(hardware)
        evaluated_hardware.update(fixed_hardware_params(self.search_space))
        write_json(hardware_path, evaluated_hardware)

        search_config = resolve_config_paths(
            read_json(self.search_config_path),
            self.search_config_path,
            run_dir,
            compass_root=self.compass_root,
        )
        search_config["run_mode"] = "GA"
        search_config["best_mapping_save_path"] = str((run_dir / "best_mapping.json").resolve())
        search_config["detail_latency_save_path"] = str((run_dir / "search_latency_detail.json").resolve())
        search_config["detail_energy_save_path"] = str((run_dir / "search_energy_detail.json").resolve())
        search_config["detail_mc_save_path"] = str((run_dir / "search_mc_detail.json").resolve())
        write_json(search_config_path, search_config)

        with search_log.open("w", encoding="utf-8") as f:
            process = subprocess.run(
                [str(self.binary), str(search_config_path), str(hardware_path), str(search_csv)],
                cwd=run_dir,
                stdout=f,
                stderr=f,
                check=False,
            )
        if process.returncode == 42:
            raise MappingEvaluationUnavailable(run_dir)
        process.check_returncode()

        return EvaluationResult(
            run_dir=run_dir,
            hardware=evaluated_hardware,
            metrics=load_csv_first_row(search_csv),
            detail_files={
                "latency": run_dir / "search_latency_detail.json",
                "energy": run_dir / "search_energy_detail.json",
                "monetary_cost": run_dir / "search_mc_detail.json",
            },
            raw_files={
                "hardware": hardware_path,
                "search_config": search_config_path,
                "metrics": search_csv,
                "search_csv": search_csv,
                "search_log": search_log,
            },
        )

    def schema(self) -> Dict[str, Any]:
        system_params = system_param_keys(self.search_space)
        return {
            "adapter": "compass",
            "design_space_kind": "explicit_count_per_chiplet",
            "required_hardware_fields": [
                "num_chiplets",
                "chip_x",
                "chip_y",
                "nop_bw",
                "dram_bw",
                "tensor_parall",
                "micro_batch",
                "chiplets",
            ],
            "chiplet_fields": ["type", "buffer_size", "compute_units", "macs"],
            "tunable_hardware_fields": {
                "chiplet_count": ["num_chiplets"],
                "per_chiplet_architecture": ["compute_units", "buffer_size", "chiplet_type"],
                "system_params": system_params,
            },
            "fixed_hardware_fields": fixed_hardware_params(self.search_space),
            "derived_hardware_fields": ["chip_y", "chip_x", "chiplets.macs"],
            "search_space": self.search_space,
            "detail_schema": {
                "latency": "search_latency_detail.json keyed by core/resource with best-mapping timing and communication fields",
                "energy": "search_energy_detail.json keyed by core/resource with best-mapping compute/buffer/noc/dram energy fields",
                "monetary_cost": "search_mc_detail.json with Compass monetary-cost breakdown",
            },
        }
