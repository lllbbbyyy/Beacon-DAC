"""A persisted wall-clock deadline, including setup and interrupted time."""
from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Callable

from chiplet_tuner.core.io import read_json, write_json


class SearchBudget:
    def __init__(self, seconds: float | None, started_at: float,
                 clock: Callable[[], float] = time.time):
        if seconds is not None and (not math.isfinite(seconds) or seconds <= 0):
            raise ValueError("time budget must be finite and positive")
        self.seconds = seconds
        self.started_at = started_at
        self.clock = clock

    @classmethod
    def open(cls, path: Path, seconds: float | None, started_at: float, resume: bool):
        if resume and path.exists():
            saved = read_json(path)
            if seconds is not None and seconds != saved["time_budget_seconds"]:
                raise ValueError("Cannot change the search time budget on resume")
            return cls(saved["time_budget_seconds"], saved["started_at_unix"])
        if resume and seconds is not None:
            raise ValueError("Cannot add a time budget to a legacy run on resume")
        if not resume and path.exists():
            raise FileExistsError(f"Search clock already exists: {path}; use a new run directory")
        budget = cls(seconds, started_at)
        if not resume:
            write_json(path, {"time_budget_seconds": seconds, "started_at_unix": started_at,
                              "resume_policy": "fixed_deadline_includes_interrupted_time"})
        return budget

    def elapsed(self) -> float:
        return max(0.0, self.clock() - self.started_at)

    def eligible(self, elapsed: float) -> bool:
        return self.seconds is None or elapsed <= self.seconds

    def expired(self) -> bool:
        return self.seconds is not None and self.elapsed() >= self.seconds
