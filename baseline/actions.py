from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

from chiplet_tuner.core.search_space import (
    buffer_size_candidates,
    chip_type_candidates,
    compute_unit_candidates,
    macs_for_compute_units,
    materialize_hardware,
    step_array_shape,
    step_chiplet_resource,
    system_param_candidates,
    system_param_keys,
)


ACTION_CATEGORIES = [
    "chiplet_count",
    "compute_all",
    "compute_one",
    "buffer_all",
    "buffer_one",
    "dataflow_all",
    "dataflow_one",
    "system",
]


def hardware_fingerprint(hardware: Mapping[str, Any]) -> str:
    payload = json.dumps(hardware, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


@dataclass
class ActionCandidate:
    action_id: str
    category: str
    description: str
    hardware: Dict[str, Any]
    parent_iteration: int
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def fingerprint(self) -> str:
        return hardware_fingerprint(self.hardware)

    def compact_dict(self) -> Dict[str, Any]:
        return {
            "action_id": self.action_id,
            "category": self.category,
            "description": self.description,
            "parent_iteration": self.parent_iteration,
            "metadata": self.metadata,
        }


class UnifiedActionSpace:
    """Generate legal neighboring hardware candidates shared by all baselines."""

    def __init__(self, search_space: Mapping[str, Any]) -> None:
        self.search_space = dict(search_space)

    def candidates(
        self,
        hardware: Mapping[str, Any],
        parent_iteration: int,
        forbidden_fingerprints: Optional[Iterable[str]] = None,
    ) -> List[ActionCandidate]:
        forbidden = set(forbidden_fingerprints or ())
        current = copy.deepcopy(dict(hardware))
        current_fp = hardware_fingerprint(current)
        result: List[ActionCandidate] = []
        seen = {current_fp, *forbidden}

        def add(
            action_id: str,
            category: str,
            description: str,
            candidate: Mapping[str, Any],
            **metadata: Any,
        ) -> None:
            materialized = copy.deepcopy(dict(candidate))
            fingerprint = hardware_fingerprint(materialized)
            if fingerprint in seen:
                return
            seen.add(fingerprint)
            result.append(
                ActionCandidate(
                    action_id=action_id,
                    category=category,
                    description=description,
                    hardware=materialized,
                    parent_iteration=parent_iteration,
                    metadata=metadata,
                )
            )

        self._add_chiplet_count_actions(current, add)
        self._add_resource_actions(current, "compute_units", add)
        self._add_resource_actions(current, "buffer_size", add)
        self._add_dataflow_actions(current, add)
        self._add_system_actions(current, add)
        return result

    def _add_chiplet_count_actions(self, hardware: Dict[str, Any], add: Any) -> None:
        for direction in (-1, 1):
            shape = step_array_shape(hardware, self.search_space, direction)
            new_count = int(shape["num_chiplets"])
            if new_count == int(hardware.get("num_chiplets", 0)):
                continue
            expanded_types = _repeat_chiplet_types(hardware, new_count)
            candidate = materialize_hardware(
                template=hardware,
                num_chiplets=new_count,
                chiplet_types=expanded_types,
                search_space=self.search_space,
                chiplet_spec_strategy="preserve_prefix",
                chiplet_type_strategy="preserve_prefix",
            )
            add(
                f"chiplet_count:{direction:+d}",
                "chiplet_count",
                f"step num_chiplets to {new_count}",
                candidate,
                direction=direction,
                target_value=new_count,
                scope="all",
            )

    def _add_resource_actions(self, hardware: Dict[str, Any], field: str, add: Any) -> None:
        category_all = "compute_all" if field == "compute_units" else "buffer_all"
        category_one = "compute_one" if field == "compute_units" else "buffer_one"
        for direction in (-1, 1):
            candidate, actions = step_chiplet_resource(
                hardware=hardware,
                search_space=self.search_space,
                field=field,
                direction=direction,
                scope="all",
            )
            if actions:
                add(
                    f"{field}:all:{direction:+d}",
                    category_all,
                    "; ".join(actions),
                    candidate,
                    direction=direction,
                    scope="all",
                    parameter=field,
                )

        chiplets = hardware.get("chiplets", [])
        if not isinstance(chiplets, Sequence) or isinstance(chiplets, (str, bytes)):
            return
        candidates = compute_unit_candidates(self.search_space) if field == "compute_units" else buffer_size_candidates(self.search_space)
        for index, chiplet in enumerate(chiplets):
            if not isinstance(chiplet, Mapping):
                continue
            old = int(chiplet.get(field, candidates[0]))
            for direction in (-1, 1):
                new = _step_value(old, candidates, direction)
                if new == old:
                    continue
                updated = copy.deepcopy(hardware)
                updated_chip = updated["chiplets"][index]
                updated_chip[field] = new
                if field == "compute_units":
                    updated_chip["macs"] = macs_for_compute_units(
                        new,
                        str(updated_chip.get("type", "ws")),
                        self.search_space,
                    )
                add(
                    f"{field}:index:{index}:{direction:+d}",
                    category_one,
                    f"chiplets[{index}].{field}: {old}->{new}",
                    updated,
                    direction=direction,
                    scope="index",
                    index=index,
                    parameter=field,
                    target_value=new,
                )

    def _add_dataflow_actions(self, hardware: Dict[str, Any], add: Any) -> None:
        legal_types = chip_type_candidates(self.search_space)
        chiplets = hardware.get("chiplets", [])
        if not isinstance(chiplets, Sequence) or isinstance(chiplets, (str, bytes)):
            return
        for index, chiplet in enumerate(chiplets):
            if not isinstance(chiplet, Mapping):
                continue
            old = str(chiplet.get("type", legal_types[0]))
            for new in legal_types:
                if new == old:
                    continue
                updated = copy.deepcopy(hardware)
                updated["chiplets"][index]["type"] = new
                updated["chiplets"][index]["macs"] = macs_for_compute_units(
                    int(updated["chiplets"][index]["compute_units"]),
                    new,
                    self.search_space,
                )
                add(
                    f"dataflow:index:{index}:{new}",
                    "dataflow_one",
                    f"chiplets[{index}].type: {old}->{new}",
                    updated,
                    scope="index",
                    index=index,
                    target_value=new,
                )

        count = len(chiplets)
        for new in legal_types:
            if all(str(chiplet.get("type", legal_types[0])) == new for chiplet in chiplets if isinstance(chiplet, Mapping)):
                continue
            candidate = materialize_hardware(
                template=hardware,
                chiplet_types=[new] * count,
                num_chiplets=count,
                search_space=self.search_space,
                chiplet_type_strategy="preserve_prefix",
            )
            add(
                f"dataflow:all:{new}",
                "dataflow_all",
                f"all chiplets.type -> {new}",
                candidate,
                scope="all",
                target_value=new,
            )

    def _add_system_actions(self, hardware: Dict[str, Any], add: Any) -> None:
        for key in system_param_keys(self.search_space):
            candidates = system_param_candidates(self.search_space, key)
            if not candidates or key not in hardware:
                continue
            old = hardware[key]
            for direction in (-1, 1):
                new = _step_value(old, candidates, direction)
                if new == old:
                    continue
                candidate = materialize_hardware(
                    template=hardware,
                    num_chiplets=hardware.get("num_chiplets"),
                    system_params={key: new},
                    search_space=self.search_space,
                )
                add(
                    f"system:{key}:{direction:+d}",
                    "system",
                    f"{key}: {old}->{new}",
                    candidate,
                    direction=direction,
                    scope="all",
                    parameter=key,
                    target_value=new,
                )


def _step_value(value: Any, candidates: Sequence[Any], direction: int) -> Any:
    ordered = sorted(set(candidates))
    if value not in ordered:
        return min(ordered, key=lambda item: abs(float(item) - float(value)))
    index = ordered.index(value)
    index = max(0, min(len(ordered) - 1, index + (1 if direction > 0 else -1)))
    return ordered[index]


def _repeat_chiplet_types(hardware: Mapping[str, Any], count: int) -> Optional[List[str]]:
    chiplets = hardware.get("chiplets", [])
    if not isinstance(chiplets, Sequence) or isinstance(chiplets, (str, bytes)):
        return None
    current = [str(item.get("type", "ws")) for item in chiplets if isinstance(item, Mapping)]
    if not current:
        return None
    return [current[index % len(current)] for index in range(count)]
