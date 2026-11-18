from __future__ import annotations

import math
from typing import Any, Dict, Mapping, Optional


# Compass treats compute_units as MAC units. A MAC contributes two arithmetic
# operations in the usual TOPS accounting, so 512 MAC units correspond to 1 TOPS
# under the 1 GHz scale convention used by this tuner.
DEFAULT_COMPUTE_UNITS_PER_SCALE = 512.0


def base_objective(metrics: Mapping[str, Any]) -> float:
    return (
        _required_positive_metric(metrics, "latency")
        * _required_positive_metric(metrics, "energy")
        * _required_positive_metric(metrics, "mc")
    )


def hardware_compute_units(hardware: Optional[Mapping[str, Any]]) -> Optional[float]:
    if not isinstance(hardware, Mapping):
        return None
    chiplets = hardware.get("chiplets")
    if not isinstance(chiplets, list):
        return None
    total = 0.0
    found = False
    for chiplet in chiplets:
        if not isinstance(chiplet, Mapping):
            continue
        try:
            total += float(chiplet.get("compute_units", 0.0) or 0.0)
        except (TypeError, ValueError):
            continue
        found = True
    return total if found else None


def hardware_compute_scale(
    hardware: Optional[Mapping[str, Any]],
    search_space: Optional[Mapping[str, Any]] = None,
) -> Optional[float]:
    total_units = hardware_compute_units(hardware)
    if total_units is None:
        return None
    units_per_scale = _positive_float(
        _mapping_get(search_space, "compute_units_per_tops"),
        default=DEFAULT_COMPUTE_UNITS_PER_SCALE,
    )
    return total_units / units_per_scale


def target_compute_scale(search_space: Optional[Mapping[str, Any]]) -> Optional[float]:
    for key in ["target_compute_scale", "compute_scale"]:
        value = _mapping_get(search_space, key)
        parsed = _positive_float(value, default=None)
        if parsed is not None:
            return parsed
    return None


def compute_budget_penalty(
    hardware: Optional[Mapping[str, Any]],
    search_space: Optional[Mapping[str, Any]] = None,
) -> Dict[str, float]:
    target = target_compute_scale(search_space)
    actual = hardware_compute_scale(hardware, search_space)
    if target is None or actual is None:
        return {"compute_budget_penalty": 1.0}
    error_ratio = (actual - target) / target
    compute_ratio = max(actual, target) / min(actual, target)
    return {
        "now_compute_scale": actual,
        "target_compute_scale": target,
        "compute_scale_error_ratio": error_ratio,
        "compute_budget_penalty": compute_ratio,
    }


def objective(
    metrics: Mapping[str, Any],
    hardware: Optional[Mapping[str, Any]] = None,
    search_space: Optional[Mapping[str, Any]] = None,
) -> float:
    base = base_objective(metrics)
    penalty = compute_budget_penalty(hardware, search_space)["compute_budget_penalty"]
    return base * penalty


def metrics_with_objective(
    metrics: Mapping[str, Any],
    hardware: Optional[Mapping[str, Any]] = None,
    search_space: Optional[Mapping[str, Any]] = None,
) -> Dict[str, float]:
    enriched: Dict[str, float] = dict(metrics)
    base = base_objective(metrics)
    penalty = compute_budget_penalty(hardware, search_space)
    enriched["base_objective"] = base
    enriched.update(penalty)
    enriched["objective"] = base * penalty["compute_budget_penalty"]
    return enriched


def _required_positive_metric(metrics: Mapping[str, Any], key: str) -> float:
    if key not in metrics:
        raise KeyError(f"Evaluation metrics missing required key: {key}")
    try:
        value = float(metrics[key])
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Evaluation metric {key} must be numeric, got {metrics[key]!r}") from exc
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"Evaluation metric {key} must be finite and positive, got {value!r}")
    return value


def _mapping_get(mapping: Optional[Mapping[str, Any]], key: str) -> Any:
    if not isinstance(mapping, Mapping):
        return None
    return mapping.get(key)


def _positive_float(value: Any, default: Optional[float]) -> Optional[float]:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(parsed) or parsed <= 0.0:
        return default
    return parsed
