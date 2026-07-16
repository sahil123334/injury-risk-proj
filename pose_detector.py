"""
pose_detector.py

Wraps MediaPipe's PoseLandmarker. Nothing outside this file should ever
import `mediapipe` directly -- every other module works with plain
Python floats/lists so MediaPipe could be swapped out later without
touching biomechanics, calibration, rep_tracker, or risk_engine.

If you're coming from C++: think of PoseDetector as a small RAII wrapper
-- it owns the MediaPipe landmarker "resource" and you use it with a
`with` block so it always gets closed, even if an exception happens.
"""

import os
from dataclasses import dataclass
from typing import List, Optional, Sequence

import mediapipe as mp
import numpy as np

import config


@dataclass
class PoseReading:
    """Everything the rest of the pipeline needs to know about one frame."""
    landmarks: Optional[list]          # raw MediaPipe landmark list, or None
    lower_body_visible: bool           # visibility AND in-frame bounds both OK
    confidence: float                  # mean visibility of lower-body points, 0..1


def _landmark_visible(lm, min_visibility: float) -> bool:
    # Some MediaPipe builds omit `.visibility` on certain landmark types;
    # treat a missing attribute as "fully visible" rather than crashing.
    visibility = getattr(lm, "visibility", 1.0)
    return visibility >= min_visibility


def landmarks_visible_enough(landmarks, indices: Sequence[int], min_visibility: float = config.MIN_VISIBILITY) -> bool:
    """True only if every requested landmark meets the visibility floor."""
    for idx in indices:
        if not _landmark_visible(landmarks[idx], min_visibility):
            return False
    return True


def landmarks_in_bounds(landmarks, indices: Sequence[int], margin: float = config.BOUNDS_MARGIN) -> bool:
    """
    Normalized landmark coordinates should sit in [0, 1]. A small margin
    is allowed because MediaPipe can slightly extrapolate near an edge;
    beyond that margin we treat the point as effectively off-screen.
    """
    low = -margin
    high = 1.0 + margin
    for idx in indices:
        lm = landmarks[idx]
        if not (low <= lm.x <= high and low <= lm.y <= high):
            return False
    return True


def average_visibility(landmarks, indices: Sequence[int]) -> float:
    """Mean visibility score across the given landmarks; used as a confidence readout."""
    if not indices:
        return 0.0
    total = sum(getattr(landmarks[idx], "visibility", 1.0) for idx in indices)
    return float(total / len(indices))


def get_point(landmarks, index: int, width: int, height: int):
    """Convert a normalized landmark to pixel coordinates."""
    lm = landmarks[index]
    return (lm.x * width, lm.y * height)


class PoseDetector:
    """Owns the MediaPipe PoseLandmarker instance for the life of the program."""

    def __init__(self, model_path: str = config.MODEL_PATH, min_visibility: float = config.MIN_VISIBILITY):
        if not os.path.isfile(model_path):
            raise FileNotFoundError(
                f"Pose model not found at '{model_path}'. "
                "Download/copy pose_landmarker.task into the project root before running."
            )

        self._min_visibility = min_visibility

        base_options = mp.tasks.BaseOptions(model_asset_path=model_path)
        options = mp.tasks.vision.PoseLandmarkerOptions(
            base_options=base_options,
            running_mode=mp.tasks.vision.RunningMode.VIDEO,
        )
        self._landmarker = mp.tasks.vision.PoseLandmarker.create_from_options(options)

    def __enter__(self) -> "PoseDetector":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    def close(self) -> None:
        self._landmarker.close()

    def detect(self, rgb_frame: np.ndarray, timestamp_ms: int) -> PoseReading:
        """Run pose detection on one already-RGB frame and validate the result."""
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        result = self._landmarker.detect_for_video(mp_image, timestamp_ms)

        if not result.pose_landmarks:
            return PoseReading(landmarks=None, lower_body_visible=False, confidence=0.0)

        landmarks = result.pose_landmarks[0]
        visible = landmarks_visible_enough(landmarks, config.LOWER_BODY_LANDMARKS, self._min_visibility)
        in_bounds = landmarks_in_bounds(landmarks, config.LOWER_BODY_LANDMARKS)
        confidence = average_visibility(landmarks, config.LOWER_BODY_LANDMARKS)

        return PoseReading(
            landmarks=landmarks,
            lower_body_visible=visible and in_bounds,
            confidence=confidence,
        )
