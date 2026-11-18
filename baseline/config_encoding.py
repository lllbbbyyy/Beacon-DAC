"""Finite full-design encoding shared by global optimizers and PPO."""
from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from chiplet_tuner.core.search_space import (
    buffer_size_candidates, chip_type_candidates, compute_unit_candidates,
    materialize_hardware, num_chiplet_candidates, system_param_candidates,
    system_param_keys,
)


class FullDesignCodec:
    def __init__(self, template: Mapping[str, Any], search_space: Mapping[str, Any]):
        self.template = dict(template)
        self.search_space = dict(search_space)
        self.counts = num_chiplet_candidates(search_space)
        self.compute = compute_unit_candidates(search_space)
        self.buffers = buffer_size_candidates(search_space)
        self.types = chip_type_candidates(search_space)
        self.system = {
            key: system_param_candidates(search_space, key)
            for key in system_param_keys(search_space)
        }
        self.max_count = max(self.counts)
        self.nvec = np.asarray(
            [len(self.counts)]
            + [len(self.compute), len(self.buffers), len(self.types)] * self.max_count
            + [len(values) for values in self.system.values()], dtype=np.int64,
        )

    def decode(self, action: np.ndarray) -> dict:
        action = np.asarray(action)
        if action.shape != self.nvec.shape or not np.issubdtype(action.dtype, np.integer):
            raise ValueError("Design action must be an integer vector matching the codec dimensions")
        if np.any(action < 0) or np.any(action >= self.nvec):
            raise ValueError("Design action is outside the candidate space")
        count = self.counts[int(action[0])]
        specs = []
        for index in range(count):
            offset = 1 + 3 * index
            specs.append({
                "compute_units": self.compute[int(action[offset])],
                "buffer_size": self.buffers[int(action[offset + 1])],
                "type": self.types[int(action[offset + 2])],
            })
        offset = 1 + 3 * self.max_count
        system = {key: values[int(action[offset + index])]
                  for index, (key, values) in enumerate(self.system.items())}
        return materialize_hardware(
            self.template, num_chiplets=count, per_chiplet_specs=specs,
            system_params=system, search_space=self.search_space,
        )

    def encode(self, hardware: Mapping[str, Any]) -> np.ndarray:
        action = np.zeros_like(self.nvec)
        action[0] = self.counts.index(int(hardware["num_chiplets"]))
        for index, chiplet in enumerate(hardware["chiplets"]):
            offset = 1 + 3 * index
            action[offset:offset + 3] = [
                self.compute.index(int(chiplet["compute_units"])),
                self.buffers.index(int(chiplet["buffer_size"])),
                self.types.index(str(chiplet["type"])),
            ]
        offset = 1 + 3 * self.max_count
        for index, (key, values) in enumerate(self.system.items()):
            action[offset + index] = values.index(hardware[key])
        return action

    def observation(self, hardware: Mapping[str, Any]) -> np.ndarray:
        # An occupancy channel distinguishes inactive slots from minimum-size chiplets.
        values = self.encode(hardware) / np.maximum(1, self.nvec - 1)
        occupancy = np.arange(self.max_count) < int(hardware["num_chiplets"])
        return np.concatenate([values, occupancy]).astype(np.float32)
