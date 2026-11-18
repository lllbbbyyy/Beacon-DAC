from __future__ import annotations

import random
from typing import Any, Mapping, Optional, Sequence, Set

import numpy as np

from baseline.actions import ActionCandidate, UnifiedActionSpace, hardware_fingerprint
from baseline.config_encoding import FullDesignCodec
from baseline.search import SearchRecord
from chiplet_tuner.core.objective import hardware_compute_scale


class RandomSearchStrategy:
    name = "random"

    def __init__(
        self,
        action_space: UnifiedActionSpace,
        seed: int = 42,
        search_space: Optional[Mapping[str, Any]] = None,
        global_prob: float = 0.5,
    ) -> None:
        self.action_space = action_space
        self.seed = seed
        self.rng = random.Random(seed)
        self.np_rng = np.random.default_rng(seed)
        self.search_space = dict(search_space) if search_space is not None else None
        self.global_prob = float(global_prob)
        self.codec: Optional[FullDesignCodec] = None

    def config(self) -> dict[str, Any]:
        return {
            "seed": self.seed,
            "sampling_mode": (
                "hybrid_scale_aware_global_and_neighbor"
                if self.search_space is not None
                else "uniform_neighbor_walk"
            ),
            "global_prob": self.global_prob if self.search_space is not None else 0.0,
        }

    def _sample_global(
        self,
        history: Sequence[SearchRecord],
        forbidden_fingerprints: Set[str],
    ) -> Optional[ActionCandidate]:
        assert self.search_space is not None
        if self.codec is None:
            self.codec = FullDesignCodec(history[0].hardware, self.search_space)
        target_scale = self.search_space.get("target_compute_scale")
        target = float(target_scale) if target_scale is not None else None
        fallback: Optional[dict[str, Any]] = None
        for _ in range(4096):
            action = np.array(
                [self.np_rng.integers(0, int(high)) for high in self.codec.nvec],
                dtype=np.int64,
            )
            hardware = self.codec.decode(action)
            fp = hardware_fingerprint(hardware)
            if fp in forbidden_fingerprints:
                continue
            if fallback is None:
                fallback = hardware
            if target is not None and target > 0:
                scale = hardware_compute_scale(hardware, self.search_space)
                ratio = max(scale, target) / max(min(scale, target), 1e-9)
                if ratio > 4.0:
                    continue
            return ActionCandidate(
                action_id="global_random_configuration",
                category="joint_configuration",
                description="sample a legal full hardware/mapping configuration",
                hardware=hardware,
                parent_iteration=0,
                metadata={"selection": "uniform_global_configuration"},
            )
        if fallback is not None:
            return ActionCandidate(
                action_id="global_random_configuration",
                category="joint_configuration",
                description="sample a legal full hardware/mapping configuration",
                hardware=fallback,
                parent_iteration=0,
                metadata={"selection": "uniform_global_configuration"},
            )
        return None

    def propose(
        self,
        history: Sequence[SearchRecord],
        forbidden_fingerprints: Set[str],
    ) -> ActionCandidate:
        if self.search_space is not None and self.rng.random() < self.global_prob:
            global_candidate = self._sample_global(history, forbidden_fingerprints)
            if global_candidate is not None:
                return global_candidate
        parents = list(history)
        self.rng.shuffle(parents)
        for parent in parents:
            candidates = self.action_space.candidates(
                parent.hardware,
                parent_iteration=parent.iteration,
                forbidden_fingerprints=forbidden_fingerprints,
            )
            if candidates:
                chosen = self.rng.choice(candidates)
                chosen.metadata.update(
                    {
                        "selection": "uniform_random_action",
                        "valid_action_count": len(candidates),
                    }
                )
                return chosen
        if self.search_space is not None:
            global_candidate = self._sample_global(history, forbidden_fingerprints)
            if global_candidate is not None:
                return global_candidate
        raise RuntimeError("The unified action space has no unevaluated hardware candidates.")

    def observe(
        self,
        parent: SearchRecord,
        result: SearchRecord,
        proposal: ActionCandidate,
    ) -> None:
        return None

    def finalize(self, history: Sequence[SearchRecord]) -> None:
        return None

