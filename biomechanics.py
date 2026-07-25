"""
biomechanics.py

Pure math: turning landmark coordinates into joint angles, and smoothing
those angles over a short rolling window. Nothing in this file knows
about MediaPipe, calibration, reps, or risk -- it only deals in floats
and points, so it has no dependency on any other project module.

Smoothing is confidence-weighted: each frame's contribution to the
rolling average is scaled by that leg's landmark visibility for that
frame, instead of every frame counting equally. A flat average lets a
single low-confidence frame (dim lighting, partial occlusion, standing
farther from the camera) drag the smoothed angle around just as much as
a fully-visible frame; weighting by confidence means unreliable frames
still contribute, but proportionally less, which is exactly the
condition under which false asymmetry/instability flags are most likely.
"""

from collections import deque
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

import config
from pose_detector import average_visibility, get_point

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
    left_confidence: float = 1.0
    right_confidence: float = 1.0


def compute_knee_angles(landmarks, width: int, height: int) -> Optional[RawKneeAngles]:
    """Hip-knee-ankle angle and landmark-visibility confidence for each leg. None if either angle is degenerate."""
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

    return RawKneeAngles(
        left=left_angle,
        right=right_angle,
        left_confidence=average_visibility(landmarks, config.LEFT_LEG_LANDMARKS),
        right_confidence=average_visibility(landmarks, config.RIGHT_LEG_LANDMARKS),
    )


@dataclass
class SmoothedAngles:
    left: float
    right: float
    mean: float
    asymmetry: float
    variance: float  # confidence-weighted std-dev of the recent smoothed mean-angle history


def _weighted_mean(values: Tuple[float, ...], weights: Tuple[float, ...]) -> float:
    total_weight = sum(weights)
    if total_weight <= 1e-6:
        # Every frame in the window was essentially unreadable -- fall back
        # to a flat average rather than dividing by ~zero.
        return float(np.mean(values))
    return float(sum(v * w for v, w in zip(values, weights)) / total_weight)


def _weighted_std(values: Tuple[float, ...], weights: Tuple[float, ...]) -> float:
    total_weight = sum(weights)
    if total_weight <= 1e-6:
        return float(np.std(values))
    mean = _weighted_mean(values, weights)
    variance = sum(w * (v - mean) ** 2 for v, w in zip(values, weights)) / total_weight
    return float(variance ** 0.5)


class AngleSmoother:
    """
    Confidence-weighted rolling average over the last `window` frames.
    Equivalent to a fixed-size ring buffer in C++ -- `deque(maxlen=...)`
    automatically drops the oldest sample once it's full.
    """

    def __init__(self, window: int = config.ANGLE_WINDOW):
        self._left_hist: deque = deque(maxlen=window)   # (angle, confidence)
        self._right_hist: deque = deque(maxlen=window)  # (angle, confidence)
        self._mean_hist: deque = deque(maxlen=window)    # (mean_angle, combined_confidence)

    def update(
        self,
        left_angle: float,
        right_angle: float,
        left_confidence: float = 1.0,
        right_confidence: float = 1.0,
    ) -> SmoothedAngles:
        mean_angle = (left_angle + right_angle) / 2.0
        combined_confidence = (left_confidence + right_confidence) / 2.0

        self._left_hist.append((left_angle, left_confidence))
        self._right_hist.append((right_angle, right_confidence))
        self._mean_hist.append((mean_angle, combined_confidence))

        left_values, left_weights = zip(*self._left_hist)
        right_values, right_weights = zip(*self._right_hist)
        mean_values, mean_weights = zip(*self._mean_hist)

        smooth_left = _weighted_mean(left_values, left_weights)
        smooth_right = _weighted_mean(right_values, right_weights)
        smooth_mean = (smooth_left + smooth_right) / 2.0

        asymmetry = abs(smooth_left - smooth_right)
        variance = _weighted_std(mean_values, mean_weights) if len(self._mean_hist) > 1 else 0.0

        return SmoothedAngles(
            left=smooth_left,
            right=smooth_right,
            mean=smooth_mean,
            asymmetry=asymmetry,
            variance=variance,
        )
