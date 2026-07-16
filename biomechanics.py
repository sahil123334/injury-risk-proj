"""
biomechanics.py

Pure math: turning landmark coordinates into joint angles, and smoothing
those angles over a short rolling window. Nothing in this file knows
about MediaPipe, calibration, reps, or risk -- it only deals in floats
and points, so it has no dependency on any other project module.
"""

from collections import deque
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

import config
from pose_detector import get_point

Point = Tuple[float, float]


def angle_between_three_points(a: Point, b: Point, c: Point) -> Optional[float]:
    """
    Angle at vertex `b`, formed by rays b->a and b->c, in degrees.
    Returns None if either ray has zero length (can't define an angle).
    """
    a_arr = np.array(a, dtype=np.float32)
    b_arr = np.array(b, dtype=np.float32)
    c_arr = np.array(c, dtype=np.float32)

    ba = a_arr - b_arr
    bc = c_arr - b_arr

    norm_ba = np.linalg.norm(ba)
    norm_bc = np.linalg.norm(bc)

    if norm_ba == 0 or norm_bc == 0:
        return None

    cosine = np.dot(ba, bc) / (norm_ba * norm_bc)
    cosine = np.clip(cosine, -1.0, 1.0)

    return float(np.degrees(np.arccos(cosine)))


@dataclass
class RawKneeAngles:
    left: float
    right: float


def compute_knee_angles(landmarks, width: int, height: int) -> Optional[RawKneeAngles]:
    """Hip-knee-ankle angle for each leg. None if either angle is degenerate."""
    left_hip = get_point(landmarks, config.LEFT_HIP, width, height)
    left_knee = get_point(landmarks, config.LEFT_KNEE, width, height)
    left_ankle = get_point(landmarks, config.LEFT_ANKLE, width, height)

    right_hip = get_point(landmarks, config.RIGHT_HIP, width, height)
    right_knee = get_point(landmarks, config.RIGHT_KNEE, width, height)
    right_ankle = get_point(landmarks, config.RIGHT_ANKLE, width, height)

    left_angle = angle_between_three_points(left_hip, left_knee, left_ankle)
    right_angle = angle_between_three_points(right_hip, right_knee, right_ankle)

    if left_angle is None or right_angle is None:
        return None

    return RawKneeAngles(left=left_angle, right=right_angle)


@dataclass
class SmoothedAngles:
    left: float
    right: float
    mean: float
    asymmetry: float
    variance: float  # std-dev of the recent smoothed mean-angle history


class AngleSmoother:
    """
    Rolling moving-average filter over the last `window` frames.
    Equivalent to a fixed-size ring buffer in C++ -- `deque(maxlen=...)`
    automatically drops the oldest sample once it's full.
    """

    def __init__(self, window: int = config.ANGLE_WINDOW):
        self._left_hist: deque = deque(maxlen=window)
        self._right_hist: deque = deque(maxlen=window)
        self._mean_hist: deque = deque(maxlen=window)

    def update(self, left_angle: float, right_angle: float) -> SmoothedAngles:
        mean_angle = (left_angle + right_angle) / 2.0

        self._left_hist.append(left_angle)
        self._right_hist.append(right_angle)
        self._mean_hist.append(mean_angle)

        smooth_left = float(np.mean(self._left_hist))
        smooth_right = float(np.mean(self._right_hist))
        smooth_mean = float(np.mean(self._mean_hist))

        asymmetry = abs(smooth_left - smooth_right)
        variance = float(np.std(self._mean_hist)) if len(self._mean_hist) > 1 else 0.0

        return SmoothedAngles(
            left=smooth_left,
            right=smooth_right,
            mean=smooth_mean,
            asymmetry=asymmetry,
            variance=variance,
        )
