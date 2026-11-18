from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

from chiplet_tuner.core.io import write_json
from chiplet_tuner.core.search_space import (
    all_system_param_candidates,
    buffer_size_candidates,
    chip_type_candidates,
    compute_unit_candidates,
    materialize_hardware,
    make_hardware_search_space,
    num_chiplet_candidates,
)


# Baseline-aligned Compass configuration source.
# Keep these values synchronized with Compass/exp.py.
CONFIG_TEMPLATE: Dict[str, Any] = {
    "seed": 42,
    "req_generator_input_length_path": "../../config/sharegpt_input_token_lens.json",
    "req_generator_output_length_path": "../../config/sharegpt_output_token_lens.json",
    "req_gen_mode": "fixed",
    "is_chunked_prefill": False,
    "req_number": 3,
    "req_prefill_number": 1,
    "req_decode_number": 127,
    "batch_size": 128,
    "model_info": {
        "type": "gpt3_merged",
        "mapping_merge_mode": "stage_post",
        "n_layer": 1,
        "d_model": 4096,
        "n_head": 32,
        "d_head": 128,
        "d_ffn": 16384,
    },
    "dram_num": 4,
    "run_mode": "GA",
    "best_mapping_save_path": "./best_mapping.json",
    "search_process_save_path": "./search_process.csv",
    "GA_population_size": 120,
    "GA_generations": 100,
    "exec_load_path": "./best_mapping.json",
    "detail_stats_mode": "mapping_node",
    "detail_latency_save_path": "./search_latency_detail.json",
    "detail_energy_save_path": "./search_energy_detail.json",
    "detail_mc_save_path": "./search_mc_detail.json",
}

COMPASS_DATASETS = ["sharegpt", "govreport"]

COMPASS_WORKLOAD_REQ_INFO: Dict[str, Dict[str, Any]] = {
    "prefill": {
        "req_number": 30,
        "req_prefill_number": 4,
        "req_decode_number": 0,
        "batch_size": 4,
    },
    "decode": {
        "req_number": 1,
        "req_prefill_number": 0,
        "req_decode_number": 128,
        "batch_size": 128,
    },
    "mixed": {
        "req_number": 1,
        "req_prefill_number": 1,
        "req_decode_number": 127,
        "batch_size": 128,
    },
    "chunked_prefill": {
        "is_chunked_prefill": True,
        "chunked_prefill_size": 2048,
        "req_number": 1,
        "req_prefill_number": 1,
        "req_decode_number": 127,
        "batch_size": 128,
    },
}

COMPASS_SCALE_MODEL_INFO: Dict[str, Dict[str, Any]] = {
    "64": {
        "type": "gpt3_merged",
        "mapping_merge_mode": "stage_post",
        "n_layer": 1,
        "d_model": 4096,
        "n_head": 32,
        "d_head": 128,
        "d_ffn": 16384,
    },
    "512": {
        "type": "gpt3_merged",
        "mapping_merge_mode": "stage_post",
        "n_layer": 1,
        "d_model": 5120,
        "n_head": 40,
        "d_head": 128,
        "d_ffn": 20480,
    },
    "2048": {
        "type": "llama3_merged",
        "mapping_merge_mode": "stage_post",
        "n_layer": 1,
        "d_model": 8192,
        "n_head": 64,
        "d_head": 128,
        "n_kv_head": 8,
        "d_ffn": 28672,
    },
}

DEFAULT_INITIAL_COMPUTE_UNITS = 32768


def build_compass_search_config(
    compass_root: Path,
    workload: str,
    dataset: str,
    compute_scale: str | int,
    chunked_prefill_size: Optional[int] = None,
    batch_size: Optional[int] = None,
    qkv_projection_mode: str = "fused_tp",
) -> Dict[str, Any]:
    if workload not in COMPASS_WORKLOAD_REQ_INFO:
        raise ValueError(f"Unsupported workload {workload!r}; legal values are {sorted(COMPASS_WORKLOAD_REQ_INFO)}")
    scale_key = str(compute_scale)
    if scale_key not in COMPASS_SCALE_MODEL_INFO:
        raise ValueError(f"Unsupported compute_scale {compute_scale!r}; legal values are {sorted(COMPASS_SCALE_MODEL_INFO)}")
    if qkv_projection_mode not in {"separate", "fused_tp"}:
        raise ValueError(
            f"Unsupported qkv_projection_mode {qkv_projection_mode!r}; "
            "legal values are ['fused_tp', 'separate']."
        )

    input_lens_path, output_lens_path = dataset_token_length_paths(compass_root, dataset)
    config = copy.deepcopy(CONFIG_TEMPLATE)
    # exp.py uses ../../config/{dataset}_*.json from its generated experiment directory.
    # The tuner writes configs under tmp/runs, so absolute paths preserve the same dataset
    # selection without depending on a Compass-relative working directory layout.
    config["req_generator_input_length_path"] = str(input_lens_path)
    config["req_generator_output_length_path"] = str(output_lens_path)
    config.update(copy.deepcopy(COMPASS_WORKLOAD_REQ_INFO[workload]))
    if batch_size is not None:
        config.update(_workload_batch_config(workload, batch_size))
    if chunked_prefill_size is not None:
        if workload != "chunked_prefill":
            raise ValueError("chunked_prefill_size is only valid for the chunked_prefill workload.")
        if chunked_prefill_size <= 0:
            raise ValueError("chunked_prefill_size must be positive.")
        config["chunked_prefill_size"] = int(chunked_prefill_size)
    config["model_info"] = copy.deepcopy(COMPASS_SCALE_MODEL_INFO[scale_key])
    config["model_info"]["qkv_projection_mode"] = qkv_projection_mode
    return config


def write_compass_search_config(
    compass_root: Path,
    output_dir: Path,
    workload: str,
    dataset: str,
    compute_scale: str | int,
    chunked_prefill_size: Optional[int] = None,
    batch_size: Optional[int] = None,
    qkv_projection_mode: str = "fused_tp",
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    search_config = build_compass_search_config(
        compass_root=compass_root,
        workload=workload,
        dataset=dataset,
        compute_scale=compute_scale,
        chunked_prefill_size=chunked_prefill_size,
        batch_size=batch_size,
        qkv_projection_mode=qkv_projection_mode,
    )
    search_path = output_dir / "search_config.json"
    write_json(search_path, search_config)
    return search_path


def _workload_batch_config(workload: str, batch_size: int) -> Dict[str, int]:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    if workload in {"mixed", "chunked_prefill"}:
        if batch_size < 2:
            raise ValueError(f"{workload} requires batch_size >= 2 to include both prefill and decode requests.")
        return {
            "req_prefill_number": 1,
            "req_decode_number": batch_size - 1,
            "batch_size": batch_size,
        }
    if workload == "prefill":
        return {
            "req_prefill_number": batch_size,
            "req_decode_number": 0,
            "batch_size": batch_size,
        }
    if workload == "decode":
        return {
            "req_prefill_number": 0,
            "req_decode_number": batch_size,
            "batch_size": batch_size,
        }
    raise ValueError(f"Unsupported workload {workload!r}; legal values are {sorted(COMPASS_WORKLOAD_REQ_INFO)}")


def build_compass_initial_hardware(
    task_type: str,
    compute_scale: str | int,
    chip_size: Optional[int] = None,
    num_chiplets: Optional[int] = None,
    compute_units: Optional[int] = None,
    buffer_size: Optional[int] = None,
    chiplet_type: str = "balanced",
    tensor_parall: Optional[int] = None,
    micro_batch: Optional[int] = None,
    batch_size: Optional[int] = None,
) -> Dict[str, Any]:
    model_info = COMPASS_SCALE_MODEL_INFO.get(str(compute_scale), {})
    effective_batch_size = int(
        batch_size
        if batch_size is not None
        else COMPASS_WORKLOAD_REQ_INFO.get(task_type, {}).get("batch_size", 128)
    )
    search_space = make_hardware_search_space(
        task_type=task_type,
        compute_scale=compute_scale,
        model_type=model_info.get("type"),
        batch_size=effective_batch_size,
    )
    system_params = {
        key: _middle_candidate(candidates)
        for key, candidates in all_system_param_candidates(search_space).items()
        if candidates
    }
    if tensor_parall is not None:
        legal_tp = all_system_param_candidates(search_space).get("tensor_parall", [])
        requested_tp = int(tensor_parall)
        if requested_tp not in legal_tp:
            raise ValueError(
                f"Initial tensor_parall={requested_tp} is not legal; candidates are {legal_tp}."
            )
        system_params["tensor_parall"] = requested_tp
    legal_micro_batches = all_system_param_candidates(search_space).get("micro_batch", [])
    requested_micro_batch = 1 if micro_batch is None else int(micro_batch)
    if requested_micro_batch not in legal_micro_batches:
        raise ValueError(
            f"Initial micro_batch={requested_micro_batch} is not legal for "
            f"batch_size={effective_batch_size}; candidates are {legal_micro_batches}."
        )
    system_params["micro_batch"] = requested_micro_batch
    legal_counts = num_chiplet_candidates(search_space)
    if not legal_counts:
        raise ValueError("Cannot generate initial hardware because no legal num_chiplets candidates exist.")
    legal_compute_units = compute_unit_candidates(search_space)
    requested_compute_units = compute_units
    if requested_compute_units is None and chip_size is not None:
        try:
            requested_compute_units = legal_compute_units[int(chip_size)]
        except (IndexError, TypeError, ValueError) as exc:
            raise ValueError(
                f"Invalid chip_size {chip_size!r}; legal indices are 0..{len(legal_compute_units) - 1}."
            ) from exc
    selected_num_chiplets, selected_compute = _target_aligned_compute_layout(
        compute_scale=compute_scale,
        search_space=search_space,
        legal_counts=legal_counts,
        legal_compute_units=legal_compute_units,
        num_chiplets=num_chiplets,
        compute_units=requested_compute_units,
    )
    selected_buffer = (
        buffer_size
        if buffer_size is not None or chip_size is not None
        else _middle_candidate(buffer_size_candidates(search_space))
    )
    chiplet_types = _initial_chiplet_types(
        chiplet_type=chiplet_type,
        num_chiplets=selected_num_chiplets,
        legal_types=chip_type_candidates(search_space),
    )
    template = _synthetic_hardware_template(system_params)
    return materialize_hardware(
        template=template,
        chip_size=chip_size,
        chiplet_types=chiplet_types,
        num_chiplets=selected_num_chiplets,
        compute_units=selected_compute,
        buffer_size=selected_buffer,
        system_params=system_params,
        search_space=search_space,
        chiplet_spec_strategy="uniform",
        chiplet_type_strategy="preserve_prefix",
        chiplet_type_fill=None if chiplet_types is not None else chiplet_type,
    )


def write_compass_initial_hardware(
    output_dir: Path,
    task_type: str,
    compute_scale: str | int,
    chip_size: Optional[int] = None,
    num_chiplets: Optional[int] = None,
    compute_units: Optional[int] = None,
    buffer_size: Optional[int] = None,
    chiplet_type: str = "balanced",
    tensor_parall: Optional[int] = None,
    micro_batch: Optional[int] = None,
    batch_size: Optional[int] = None,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    hardware = build_compass_initial_hardware(
        task_type=task_type,
        compute_scale=compute_scale,
        chip_size=chip_size,
        num_chiplets=num_chiplets,
        compute_units=compute_units,
        buffer_size=buffer_size,
        chiplet_type=chiplet_type,
        tensor_parall=tensor_parall,
        micro_batch=micro_batch,
        batch_size=batch_size,
    )
    hardware_path = output_dir / "hardware_initial.json"
    write_json(hardware_path, hardware)
    return hardware_path


def _synthetic_hardware_template(system_params: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "num_chiplets": 1,
        "chip_x": 1,
        "chip_y": 1,
        "chiplets": [],
        **system_params,
    }


def _middle_candidate(candidates: Sequence[Any]) -> Any:
    if not candidates:
        raise ValueError("Cannot choose a middle candidate from an empty candidate list.")
    return list(candidates)[(len(candidates) - 1) // 2]


def _target_aligned_compute_layout(
    compute_scale: str | int,
    search_space: Dict[str, Any],
    legal_counts: Sequence[int],
    legal_compute_units: Sequence[int],
    num_chiplets: Optional[int],
    compute_units: Optional[int],
) -> tuple[int, int]:
    try:
        target_units_float = float(compute_scale) * float(search_space["compute_units_per_tops"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"Cannot derive target compute units from compute_scale={compute_scale!r}.") from exc
    target_units = int(target_units_float)
    if target_units <= 0 or target_units != target_units_float:
        raise ValueError(f"Target compute units must be a positive integer, got {target_units_float!r}.")

    counts = [int(value) for value in legal_counts]
    units = [int(value) for value in legal_compute_units]
    requested_count = int(num_chiplets) if num_chiplets is not None else None
    requested_units = int(compute_units) if compute_units is not None else None
    if requested_count is not None and requested_count not in counts:
        raise ValueError(f"Initial num_chiplets={requested_count} is not legal; candidates are {counts}.")
    if requested_units is not None and requested_units not in units:
        raise ValueError(f"Initial compute_units={requested_units} is not legal; candidates are {units}.")

    candidates = [
        (count, unit)
        for count in counts
        for unit in units
        if count * unit == target_units
        and (requested_count is None or count == requested_count)
        and (requested_units is None or unit == requested_units)
    ]
    if not candidates:
        requested = []
        if requested_count is not None:
            requested.append(f"num_chiplets={requested_count}")
        if requested_units is not None:
            requested.append(f"compute_units={requested_units}")
        suffix = f" with {', '.join(requested)}" if requested else ""
        raise ValueError(
            f"No legal initial architecture exactly matches compute_scale={compute_scale} TOPS "
            f"({target_units} total MAC units){suffix}."
        )

    # Prefer an even chiplet count so the default WS/OS split is exact. Keep
    # the neutral initial point near 32768 MACs per chiplet rather than moving
    # it whenever the legal upper bound is expanded.
    return min(
        candidates,
        key=lambda item: (
            item[0] % 2,
            item[0] < 2,
            abs(item[1] - DEFAULT_INITIAL_COMPUTE_UNITS),
            item[0],
        ),
    )


def _initial_chiplet_types(
    chiplet_type: str,
    num_chiplets: int,
    legal_types: Sequence[str],
) -> Optional[list[str]]:
    if chiplet_type != "balanced":
        if chiplet_type not in legal_types:
            raise ValueError(f"Invalid initial chiplet type {chiplet_type!r}; legal values are {list(legal_types)}.")
        return None
    if "ws" not in legal_types or "os" not in legal_types:
        raise ValueError("Balanced initial dataflows require both ws and os in the design space.")
    if num_chiplets % 2:
        raise ValueError(
            f"Balanced initial dataflows require an even chiplet count, got {num_chiplets}. "
            "Select --initial-chiplet-type ws or os to use an odd count."
        )
    return ["ws" if index % 2 == 0 else "os" for index in range(num_chiplets)]


def dataset_token_length_paths(compass_root: Path, dataset: str) -> tuple[Path, Path]:
    if dataset not in COMPASS_DATASETS:
        raise ValueError(f"Unsupported dataset {dataset!r}; legal values are {sorted(COMPASS_DATASETS)}")
    config_dir = compass_root / "config"
    input_lens_path = (config_dir / f"fixed_{dataset}_input_token_lens.json").resolve()
    output_lens_path = (config_dir / f"{dataset}_output_token_lens.json").resolve()
    missing = [str(path) for path in [input_lens_path, output_lens_path] if not path.exists()]
    if missing:
        raise FileNotFoundError(
            f"Dataset {dataset!r} is missing token length files: {', '.join(missing)}"
        )
    return input_lens_path, output_lens_path
