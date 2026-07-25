"""
naive_rep_counter.py

A faithful reimplementation of the pre-refactor rep-counting logic from
injury_risk_mvp_backup.py: a single `in_rep` flag with no debounce and
no distinction between "still descending" and "at the bottom" -- a rep
counts the instant the angle crosses back above the shallow threshold,
no matter how it got there.

This exists only so validate_reps.py can benchmark the modern
debounced state machine (rep_tracker.py) against the original approach
on the same labeled clips. It is not used by the live app.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class NaiveRepResult:
    index: int


class NaiveRepCounter:
    def __init__(self, deep_threshold: float, shallow_threshold: float):
        self.deep_threshold = deep_threshold
        self.shallow_threshold = shallow_threshold
        self._in_rep = False
        self._rep_count = 0

    def update(self, angle: float) -> Optional[NaiveRepResult]:
        if angle < self.deep_threshold:
            self._in_rep = True

        if self._in_rep and angle > self.shallow_threshold:
            self._in_rep = False
            self._rep_count += 1
            return NaiveRepResult(index=self._rep_count)

        return None

    @property
    def rep_count(self) -> int:
        return self._rep_count
