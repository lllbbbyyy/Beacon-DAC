from __future__ import annotations

import math
from typing import Any, List, Mapping, Sequence

import numpy as np

from baseline.actions import ACTION_CATEGORIES, ActionCandidate
from chiplet_tuner.core.objective import hardware_compute_scale, target_compute_scale
from chiplet_tuner.core.search_space import (
    buffer_size_candidates,
    chip_type_candidates,
    compute_unit_candidates,
    num_chiplet_candidates,
    system_param_candidates,
    system_param_keys,
)


METRIC_KEYS = ["latency", "energy", "mc", "base_objective", "objective"]


def hardware_vector(hardware: Mapping[str, Any], search_space: Mapping[str, Any]) -> np.ndarray:
    """Encode a variable-size chiplet array into a stable, normalized vector."""

    count_candidates = num_chiplet_candidates(search_space)
    compute_candidates = compute_unit_candidates(search_space)
    buffer_candidates = buffer_size_candidates(search_space)
    type_candidates = chip_type_candidates(search_space)
    chiplets = hardware.get("chiplets", [])
    if not isinstance(chiplets, Sequence) or isinstance(chiplets, (str, bytes)):
        chiplets = []

    count = max(1, len(chiplets))
    max_count = max(count_candidates or [count])
    chip_y = _positive_float(hardware.get("chip_y"), 1.0)
    chip_x = _positive_float(hardware.get("chip_x"), 1.0)
    target = target_compute_scale(search_space)
    actual = hardware_compute_scale(hardware, search_space)

    values: List[float] = [
        _candidate_position(int(hardware.get("num_chiplets", count)), count_candidates),
        math.log2(count) / max(1.0, math.log2(max_count)),
        chip_y / max(1.0, math.sqrt(max_count)),
        chip_x / max(1.0, math.sqrt(max_count)),
        math.log(max((actual or 1.0) / (target or actual or 1.0), 1e-12)),
    ]

    compute_indices: List[float] = []
    buffer_indices: List[float] = []
    compute_hist = [0.0] * len(compute_candidates)
    buffer_hist = [0.0] * len(buffer_candidates)
    type_hist = [0.0] * len(type_candidates)
    for chiplet in chiplets:
        if not isinstance(chiplet, Mapping):
            continue
        compute = int(chiplet.get("compute_units", compute_candidates[0]))
        buffer_size = int(chiplet.get("buffer_size", buffer_candidates[0]))
        chip_type = str(chiplet.get("type", type_candidates[0]))
        compute_index = _nearest_index(compute, compute_candidates)
        buffer_index = _nearest_index(buffer_size, buffer_candidates)
        compute_hist[compute_index] += 1.0 / count
        buffer_hist[buffer_index] += 1.0 / count
        compute_indices.append(_normalized_index(compute_index, compute_candidates))
        buffer_indices.append(_normalized_index(buffer_index, buffer_candidates))
        if chip_type in type_candidates:
            type_hist[type_candidates.index(chip_type)] += 1.0 / count

    values.extend(_mean_std(compute_indices))
    values.extend(_mean_std(buffer_indices))
    values.extend(compute_hist)
    values.extend(buffer_hist)
    values.extend(type_hist)
    for key in system_param_keys(search_space):
        values.append(_candidate_position(hardware.get(key), system_param_candidates(search_space, key)))
    return np.asarray(values, dtype=float)


def state_vector(
    record: Any,
    initial_record: Any,
    best_record: Any,
    search_space: Mapping[str, Any],
) -> np.ndarray:
    values = list(hardware_vector(record.hardware, search_space))
    for key in METRIC_KEYS:
        current = _positive_float(record.metrics.get(key), 1.0)
        initial = _positive_float(initial_record.metrics.get(key), current)
        best = _positive_float(best_record.metrics.get(key), current)
        values.append(math.log(current / initial))
        values.append(math.log(current / best))
    return np.asarray(values, dtype=float)


def action_vector(
    action: ActionCandidate,
    parent_hardware: Mapping[str, Any],
    search_space: Mapping[str, Any],
) -> np.ndarray:
    category = [1.0 if action.category == item else 0.0 for item in ACTION_CATEGORIES]
    direction = float(action.metadata.get("direction", 0.0) or 0.0)
    scope_one = 1.0 if action.metadata.get("scope") == "index" else 0.0
    index = float(action.metadata.get("index", -1) or 0.0)
    count = max(1, int(parent_hardware.get("num_chiplets", 1)))
    index = max(0.0, index) / count
    delta = hardware_vector(action.hardware, search_space) - hardware_vector(parent_hardware, search_space)
    return np.concatenate(
        [
            np.asarray(category + [direction, scope_one, index], dtype=float),
            delta,
        ]
    )


def policy_feature(state: np.ndarray, action: np.ndarray, category_count: int) -> np.ndarray:
    """Create a compact state-action interaction feature for the online policy."""

    category = action[:category_count]
    interactions = np.concatenate([state * flag for flag in category])
    return np.concatenate([np.asarray([1.0]), state, action, interactions])


def _candidate_position(value: Any, candidates: Sequence[Any]) -> float:
    if not candidates:
        return 0.0
    try:
        index = candidates.index(value)
    except ValueError:
        index = min(range(len(candidates)), key=lambda item: abs(float(candidates[item]) - float(value)))
    return _normalized_index(index, candidates)


def _nearest_index(value: float, candidates: Sequence[float]) -> int:
    return min(range(len(candidates)), key=lambda item: abs(float(candidates[item]) - float(value)))


def _normalized_index(index: int, candidates: Sequence[Any]) -> float:
    return float(index) / max(1, len(candidates) - 1)


def _mean_std(values: Sequence[float]) -> List[float]:
    if not values:
        return [0.0, 0.0]
    array = np.asarray(values, dtype=float)
    return [float(array.mean()), float(array.std())]


def _positive_float(value: Any, default: float) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(parsed) or parsed <= 0.0:
        return default
    return parsed
