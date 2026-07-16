"""
calibration.py

Figures out this specific athlete's standing angle and deep-squat angle
during a short calibration window, then derives personalized rep
thresholds from that range.

Design notes (why it's built this way):
- Calibration progress is measured in *valid-pose seconds*, not wall
  clock time. If the athlete steps out of frame, the clock pauses
  instead of burning through the window on bad data.
- It never looks at rep_count or the rep tracker. Calibration must
  finish before reps can be tracked at all, so depending on rep_count
  here would be a circular dependency (issue explicitly called out in
  the refactor brief).
- Standing/deep angle estimates use the 90th/10th percentile of the
  collected samples rather than raw max/min, so one jittery outlier
  frame can't blow out the calibrated range.
"""

from dataclasses import dataclass
from typing import List, Optional

import numpy as np

import config


@dataclass
class CalibrationResult:
    standing_angle: float
    deep_angle: float
    deep_threshold: float
    shallow_threshold: float


class Calibrator:
    """Call `update()` once per frame that has a valid, in-bounds pose."""

    def __init__(
        self,
        required_seconds: float = config.CALIBRATION_SECONDS,
        min_samples: int = config.MIN_CALIBRATION_SAMPLES,
        min_movement_range: float = config.MIN_MOVEMENT_RANGE_DEG,
        deep_ratio: float = config.DEEP_THRESHOLD_RATIO,
        shallow_ratio: float = config.SHALLOW_THRESHOLD_RATIO,
        debug: bool = False,
    ):
        self._required_seconds = required_seconds
        self._min_samples = min_samples
        self._min_movement_range = min_movement_range
        self._deep_ratio = deep_ratio
        self._shallow_ratio = shallow_ratio
        self._debug = debug

        self._samples: List[float] = []
        self._valid_seconds = 0.0
        self._result: Optional[CalibrationResult] = None
        self.last_reset_reason: Optional[str] = None

    @property
    def is_calibrated(self) -> bool:
        return self._result is not None

    @property
    def result(self) -> Optional[CalibrationResult]:
        return self._result

    @property
    def progress(self) -> float:
        if self._required_seconds <= 0:
            return 1.0
        return min(1.0, self._valid_seconds / self._required_seconds)

    @property
    def time_remaining(self) -> float:
        return max(0.0, self._required_seconds - self._valid_seconds)

    def reset(self, reason: str = "manual reset") -> None:
        self._samples = []
        self._valid_seconds = 0.0
        self._result = None
        self.last_reset_reason = reason
        if self._debug:
            print(f"[calibration] reset: {reason}")

    def update(self, smooth_mean_angle: float, dt: float) -> None:
        """
        Feed one valid-pose frame into calibration. `dt` should be the
        elapsed seconds since the previous *valid* frame (pass 0.0, or
        skip the call entirely, for frames where the pose is unreliable).
        """
        if self._result is not None:
            return  # already calibrated; nothing more to do

        self._samples.append(smooth_mean_angle)
        self._valid_seconds += max(dt, 0.0)

        if self._valid_seconds >= self._required_seconds:
            self._finalize()

    def _finalize(self) -> None:
        if len(self._samples) < self._min_samples:
            self.reset("not enough valid-pose samples collected")
            return

        samples = np.array(self._samples, dtype=np.float32)
        standing_angle = float(np.percentile(samples, config.CALIBRATION_HIGH_PERCENTILE))
        deep_angle = float(np.percentile(samples, config.CALIBRATION_LOW_PERCENTILE))
        movement_range = standing_angle - deep_angle

        if movement_range < self._min_movement_range:
            self.reset(
                f"movement range too small ({movement_range:.1f} deg < "
                f"{self._min_movement_range:.1f} deg) -- do deeper, cleaner reps"
            )
            return

        deep_threshold = standing_angle - self._deep_ratio * movement_range
        shallow_threshold = standing_angle - self._shallow_ratio * movement_range

        self._result = CalibrationResult(
            standing_angle=standing_angle,
            deep_angle=deep_angle,
            deep_threshold=deep_threshold,
            shallow_threshold=shallow_threshold,
        )

        if self._debug:
            print(
                "[calibration] complete: "
                f"standing={standing_angle:.1f} deep={deep_angle:.1f} "
                f"deep_thr={deep_threshold:.1f} shallow_thr={shallow_threshold:.1f}"
            )
