"""Validated phase subspaces for the AgenticDSE adaptation.

The LLM selects candidate values, not new bounds or hardware fields. Omitted
fields retain their previous choices; selecting the full domain expands them.
"""
from __future__ import annotations

from typing import Any, Mapping
from itertools import product
from math import prod

import numpy as np

from baseline.config_encoding import FullDesignCodec


def validate_selections(domains, selections):
    """Validate the external phase contract before applying any changes."""
    if not isinstance(selections, Mapping) or not selections:
        raise ValueError("Phase selections must be a nonempty field-to-candidates mapping")
    for key, values in selections.items():
        if key not in domains:
            raise ValueError(f"Unknown design field: {key}; per-chip fields must use "
                             "chiplets.<zero_based_index>.<parameter>")
        if not isinstance(values, list) or not values:
            raise ValueError(f"{key}: provide a nonempty candidate list")
        seen = set()
        for value in values:
            matches = [i for i, candidate in enumerate(domains[key])
                       if type(value) is type(candidate) and value == candidate]
            if not matches:
                raise ValueError(f"{key}: illegal candidate {value!r}")
            if matches[0] in seen:
                raise ValueError(f"{key}: repeated candidate {value!r}")
            seen.add(matches[0])


class PhaseRegion:
    def __init__(self, codec: FullDesignCodec):
        self.codec = codec
        self.domains = {"num_chiplets": list(codec.counts)}
        for index in range(codec.max_count):
            for name, values in (
                ("compute_units", codec.compute),
                ("buffer_size", codec.buffers), ("type", codec.types),
            ):
                self.domains[f"chiplets.{index}.{name}"] = list(values)
        self.domains.update({key: list(values) for key, values in codec.system.items()})
        self.choices = {key: list(values) for key, values in self.domains.items()}

    def update(self, selections: Mapping[str, Any]) -> dict:
        """Apply one phase atomically; failed validation leaves the region intact."""
        validate_selections(self.domains, selections)
        changes = {}
        for key, requested in selections.items():
            if key not in self.domains:
                raise ValueError(f"Unknown design field: {key}")
            if not isinstance(requested, list) or not requested:
                raise ValueError(f"{key}: provide a nonempty candidate list")
            domain = self.domains[key]
            indices = []
            for value in requested:
                matches = [i for i, candidate in enumerate(domain)
                           if type(value) is type(candidate) and value == candidate]
                if not matches:
                    raise ValueError(f"{key}: illegal candidate {value!r}")
                if matches[0] in indices:
                    raise ValueError(f"{key}: repeated candidate {value!r}")
                indices.append(matches[0])
            # Canonical order makes identical regions independent of LLM list order.
            new = [domain[i] for i in sorted(indices)]
            old = self.choices[key]
            if new == old:
                kind = "unchanged"
            elif all(value in old for value in new):
                kind = "narrow"
            elif all(value in new for value in old):
                kind = "expand"
            else:
                kind = "shift"
            changes[key] = {"before": list(old), "after": new, "change": kind}
        for key, change in changes.items():
            self.choices[key] = list(change["after"])
        return changes

    def contains(self, hardware: Mapping[str, Any]) -> bool:
        encoded = self.codec.encode(hardware)
        count = int(hardware["num_chiplets"])
        for index, (key, domain) in enumerate(self.domains.items()):
            # Inactive slots must not exclude otherwise compatible history.
            if key.startswith("chiplets.") and int(key.split(".")[1]) >= count:
                continue
            if domain[int(encoded[index])] not in self.choices[key]:
                return False
        return True

    def decode_unit(self, point: np.ndarray) -> dict:
        """Map a Sobol/uniform point to this phase's discrete candidate lists."""
        point = np.asarray(point, dtype=float)
        if (point.shape != self.codec.nvec.shape or not np.isfinite(point).all()
                or np.any(point < 0) or np.any(point >= 1)):
            raise ValueError("Sampling point must match codec dimensions and lie in [0, 1)")
        action = np.zeros_like(self.codec.nvec)
        for index, (key, domain) in enumerate(self.domains.items()):
            choices = self.choices[key]
            value = choices[int(point[index] * len(choices))]
            action[index] = domain.index(value)
        return self.codec.decode(action)

    def snapshot(self) -> dict:
        return {key: list(values) for key, values in self.choices.items()}

    def enumerate_small(self, limit=256):
        """Enumerate small regions exactly, ignoring inactive-slot multiplicity."""
        fields = list(self.domains)
        layouts = []
        total = 0
        for count in self.choices["num_chiplets"]:
            active = [key for key in fields[1:] if not key.startswith("chiplets.")
                      or int(key.split(".")[1]) < count]
            total += prod(len(self.choices[key]) for key in active)
            if total > limit:
                return None
            layouts.append((count, active))
        result = []
        for count, active in layouts:
            for values in product(*(self.choices[key] for key in active)):
                selected = {key: choices[0] for key, choices in self.choices.items()}
                selected.update(zip(active, values))
                selected["num_chiplets"] = count
                action = np.array([self.domains[key].index(selected[key]) for key in fields])
                result.append(self.codec.decode(action))
        return result
