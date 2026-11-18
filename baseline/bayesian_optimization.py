from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Any, List, Mapping, Sequence, Set, Tuple

import numpy as np

from baseline.actions import ActionCandidate, UnifiedActionSpace
from baseline.encoding import hardware_vector
from baseline.search import SearchRecord
from chiplet_tuner.core.search_space import (
    buffer_size_candidates,
    chip_type_candidates,
    compute_unit_candidates,
)


@dataclass
class _KernelPoint:
    global_vector: np.ndarray
    shape: Tuple[int, int]
    layout: np.ndarray


class HardwareAwareGaussianProcess:
    """Small dependency-free GP with a topology-aware composite kernel."""

    def __init__(
        self,
        search_space: Mapping[str, Any],
        length_scale: float = 0.35,
        layout_weight: float = 1.0,
        noise: float = 1e-5,
    ) -> None:
        self.search_space = dict(search_space)
        self.length_scale = length_scale
        self.layout_weight = layout_weight
        self.noise = noise
        self._points: List[_KernelPoint] = []
        self._cholesky: np.ndarray | None = None
        self._alpha: np.ndarray | None = None
        self._y_mean = 0.0
        self._y_std = 1.0

    def fit(self, hardware: Sequence[Mapping[str, Any]], objectives: Sequence[float]) -> None:
        self._cholesky = None
        self._alpha = None
        self._points = [self._point(item) for item in hardware]
        y = np.log(np.asarray(objectives, dtype=float))
        self._y_mean = float(y.mean())
        self._y_std = max(float(y.std()), 1e-8)
        normalized = (y - self._y_mean) / self._y_std
        size = len(self._points)
        kernel = np.empty((size, size), dtype=float)
        for row in range(size):
            for column in range(row, size):
                value = self._kernel(self._points[row], self._points[column])
                kernel[row, column] = value
                kernel[column, row] = value
        jitter = self.noise
        for _ in range(8):
            try:
                self._cholesky = np.linalg.cholesky(kernel + np.eye(size) * jitter)
                break
            except np.linalg.LinAlgError:
                jitter *= 10.0
        if self._cholesky is None:
            raise np.linalg.LinAlgError("Unable to stabilize the hardware-aware GP kernel.")
        self._alpha = np.linalg.solve(
            self._cholesky.T,
            np.linalg.solve(self._cholesky, normalized),
        )

    def predict(self, hardware: Sequence[Mapping[str, Any]]) -> Tuple[np.ndarray, np.ndarray]:
        if self._cholesky is None or self._alpha is None:
            raise RuntimeError("The GP must be fitted before prediction.")
        points = [self._point(item) for item in hardware]
        cross = np.asarray(
            [[self._kernel(point, train) for train in self._points] for point in points],
            dtype=float,
        )
        means = cross @ self._alpha
        solved = np.linalg.solve(self._cholesky, cross.T)
        diagonal = np.asarray([self._kernel(point, point) for point in points], dtype=float)
        variances = np.maximum(diagonal - np.sum(np.square(solved), axis=0), 1e-12)
        return means, np.sqrt(variances)

    def _point(self, hardware: Mapping[str, Any]) -> _KernelPoint:
        compute = compute_unit_candidates(self.search_space)
        buffers = buffer_size_candidates(self.search_space)
        types = chip_type_candidates(self.search_space)
        layout: List[Tuple[int, int, int]] = []
        for chiplet in hardware.get("chiplets", []):
            compute_value = int(chiplet.get("compute_units", compute[0]))
            buffer_value = int(chiplet.get("buffer_size", buffers[0]))
            chip_type = str(chiplet.get("type", types[0]))
            layout.append(
                (
                    _nearest_index(compute_value, compute),
                    _nearest_index(buffer_value, buffers),
                    types.index(chip_type) if chip_type in types else 0,
                )
            )
        return _KernelPoint(
            global_vector=hardware_vector(hardware, self.search_space),
            shape=(int(hardware.get("chip_y", 1)), int(hardware.get("chip_x", 1))),
            layout=np.asarray(layout, dtype=int),
        )

    def _kernel(self, left: _KernelPoint, right: _KernelPoint) -> float:
        distance = float(np.mean(np.square(left.global_vector - right.global_vector)))
        global_similarity = math.exp(-0.5 * distance / (self.length_scale**2))
        if left.shape != right.shape:
            return global_similarity
        layout_similarity = self._layout_similarity(left, right)
        return global_similarity * (1.0 + self.layout_weight * layout_similarity)

    @staticmethod
    def _layout_similarity(left: _KernelPoint, right: _KernelPoint) -> float:
        height, width = left.shape
        if left.layout.size == 0 or len(left.layout) != len(right.layout):
            return 0.0
        left_grid = left.layout.reshape(height, width, 3)
        right_grid = right.layout.reshape(height, width, 3)
        best = np.zeros((height, width), dtype=float)
        for delta_row, delta_column in [(0, 0), (-1, 0), (1, 0), (0, -1), (0, 1)]:
            left_rows, right_rows = _overlap_slices(height, delta_row)
            left_columns, right_columns = _overlap_slices(width, delta_column)
            left_values = left_grid[left_rows, left_columns]
            right_values = right_grid[right_rows, right_columns]
            similarity = (
                np.exp(-np.abs(left_values[..., 0] - right_values[..., 0]))
                * np.exp(-np.abs(left_values[..., 1] - right_values[..., 1]))
                * np.where(left_values[..., 2] == right_values[..., 2], 1.0, 0.25)
            )
            similarity *= math.exp(-(abs(delta_row) + abs(delta_column)))
            target = best[left_rows, left_columns]
            np.maximum(target, similarity, out=target)
        return float(best.mean())


class BayesianOptimizationStrategy:
    name = "bayesian_optimization"

    def __init__(
        self,
        action_space: UnifiedActionSpace,
        search_space: Mapping[str, Any],
        seed: int = 42,
        warmup: int = 8,
        candidate_pool_size: int = 2048,
        max_parents: int = 16,
        exploration: float = 0.01,
    ) -> None:
        self.action_space = action_space
        self.search_space = dict(search_space)
        self.rng = random.Random(seed)
        self.warmup = max(1, warmup)
        self.candidate_pool_size = max(1, candidate_pool_size)
        self.max_parents = max(1, max_parents)
        self.exploration = exploration

    def propose(
        self,
        history: Sequence[SearchRecord],
        forbidden_fingerprints: Set[str],
    ) -> ActionCandidate:
        pool = self._candidate_pool(history, forbidden_fingerprints)
        if not pool:
            raise RuntimeError("The unified action space has no unevaluated hardware candidates.")
        if len(history) <= self.warmup:
            chosen = self.rng.choice(pool)
            chosen.metadata.update(
                {
                    "selection": "random_warmup",
                    "candidate_pool_size": len(pool),
                }
            )
            return chosen

        gp = HardwareAwareGaussianProcess(self.search_space)
        gp.fit(
            [record.hardware for record in history],
            [record.objective for record in history],
        )
        means, stds = gp.predict([candidate.hardware for candidate in pool])
        best = min((math.log(record.objective) - gp._y_mean) / gp._y_std for record in history)
        improvement = best - means - self.exploration
        z = improvement / np.maximum(stds, 1e-12)
        cdf = np.asarray([0.5 * (1.0 + math.erf(value / math.sqrt(2.0))) for value in z])
        pdf = np.exp(-0.5 * np.square(z)) / math.sqrt(2.0 * math.pi)
        expected_improvement = improvement * cdf + stds * pdf
        chosen_index = int(np.argmax(expected_improvement))
        chosen = pool[chosen_index]
        chosen.metadata.update(
            {
                "selection": "expected_improvement",
                "candidate_pool_size": len(pool),
                "predicted_log_objective": float(means[chosen_index] * gp._y_std + gp._y_mean),
                "predicted_normalized_std": float(stds[chosen_index]),
                "expected_improvement": float(expected_improvement[chosen_index]),
            }
        )
        return chosen

    def observe(
        self,
        parent: SearchRecord,
        result: SearchRecord,
        proposal: ActionCandidate,
    ) -> None:
        return None

    def finalize(self, history: Sequence[SearchRecord]) -> None:
        return None

    def _candidate_pool(
        self,
        history: Sequence[SearchRecord],
        forbidden_fingerprints: Set[str],
    ) -> List[ActionCandidate]:
        best = sorted(history, key=lambda record: record.objective)[: max(2, self.max_parents // 2)]
        recent = list(history[-max(2, self.max_parents // 4) :])
        remaining = list(history)
        self.rng.shuffle(remaining)
        parents: List[SearchRecord] = []
        seen_iterations = set()
        for record in [*best, *recent, *remaining]:
            if record.iteration in seen_iterations:
                continue
            parents.append(record)
            seen_iterations.add(record.iteration)
            if len(parents) >= self.max_parents:
                break

        candidates: List[ActionCandidate] = []
        seen = set(forbidden_fingerprints)
        for parent in parents:
            generated = self.action_space.candidates(
                parent.hardware,
                parent_iteration=parent.iteration,
                forbidden_fingerprints=seen,
            )
            candidates.extend(generated)
            seen.update(candidate.fingerprint for candidate in generated)
        if len(candidates) > self.candidate_pool_size:
            candidates = self.rng.sample(candidates, self.candidate_pool_size)
        return candidates


def _nearest_index(value: int, candidates: Sequence[int]) -> int:
    return min(range(len(candidates)), key=lambda item: abs(candidates[item] - value))


def _overlap_slices(size: int, delta: int) -> Tuple[slice, slice]:
    if delta < 0:
        return slice(-delta, size), slice(0, size + delta)
    if delta > 0:
        return slice(0, size - delta), slice(delta, size)
    return slice(0, size), slice(0, size)
