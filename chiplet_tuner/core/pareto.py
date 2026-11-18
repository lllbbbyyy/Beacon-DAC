"""Two-objective minimization utilities with a fixed hypervolume reference."""

from __future__ import annotations

import math
from typing import Iterable, Sequence


def _points(values: Iterable[Sequence[float]]) -> list[tuple[float, float]]:
    points = []
    for value in values:
        if len(value) != 2:
            raise ValueError("Expected two objective values per point.")
        point = (float(value[0]), float(value[1]))
        if not all(math.isfinite(v) for v in point):
            raise ValueError("Pareto objectives must be finite.")
        points.append(point)
    return points


def nondominated_indices(values: Iterable[Sequence[float]]) -> list[int]:
    """Return a stable first occurrence for each nondominated coordinate pair."""
    points = _points(values)
    indices = []
    seen = set()
    for i, point in enumerate(points):
        if point in seen:
            continue
        seen.add(point)
        if not any(other[0] <= point[0] and other[1] <= point[1] and other != point for other in points):
            indices.append(i)
    return indices


def hypervolume_2d(values: Iterable[Sequence[float]], reference: Sequence[float]) -> float:
    """Exact dominated area. Points outside the reference contribute nothing."""
    ref = _points([reference])[0]
    points = [p for p in _points(values) if p[0] < ref[0] and p[1] < ref[1]]
    front = sorted(points[i] for i in nondominated_indices(points))
    area = 0.
    ceiling = ref[1]
    for x, y in front:
        area += (ref[0] - x) * (ceiling - y)
        ceiling = y
    return area
