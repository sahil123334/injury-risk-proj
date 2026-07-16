"""
config.py

All tunable constants live here. If you are used to C++, think of this
module as a header full of `const` values -- nothing in here does any
work, it just names numbers so the rest of the code doesn't have magic
numbers scattered through it.

Change values here to tune behavior; you should not need to edit any
other file to adjust thresholds.
"""

import os
from dataclasses import dataclass
from typing import Optional


# -----------------------------
# PATHS
# -----------------------------
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(PROJECT_ROOT, "pose_landmarker.task")

DATA_DIR = os.path.join(PROJECT_ROOT, "data")
PER_FRAME_CSV = os.path.join(DATA_DIR, "session_metrics.csv")
PER_REP_CSV = os.path.join(DATA_DIR, "rep_summary.csv")


# -----------------------------
# CAMERA
# -----------------------------
@dataclass
class CameraConfig:
    """Groups the handful of settings that describe how we open the webcam."""
    index: int = 0
    # cv2.CAP_AVFOUNDATION is the macOS-native backend. Kept as a separate
    # flag (rather than hardcoding the enum) so main.py can fall back to
    # the default backend on non-mac machines without editing this file.
    use_avfoundation: bool = True
    requested_width: Optional[int] = None
    requested_height: Optional[int] = None


CAMERA = CameraConfig()

WINDOW_NAME = "Movement Quality Monitor"

# Some backends (e.g. AVFoundation bridging a Continuity Camera / iPhone) can
# return a handful of failed reads right after opening before frames actually
# start flowing. Tolerate that many consecutive failures before giving up.
MAX_CONSECUTIVE_READ_FAILURES = 60


# -----------------------------
# POSE / VISIBILITY VALIDATION
# -----------------------------
MIN_VISIBILITY = 0.65

# MediaPipe landmark coordinates are normalized to [0, 1] relative to the
# frame. A landmark can drift slightly outside that range when the body
# part is near the frame edge (the model is extrapolating). Anything past
# this margin is treated as "off-screen" / unreliable rather than clamped.
BOUNDS_MARGIN = 0.05

LEFT_HIP, LEFT_KNEE, LEFT_ANKLE = 23, 25, 27
RIGHT_HIP, RIGHT_KNEE, RIGHT_ANKLE = 24, 26, 28
LOWER_BODY_LANDMARKS = [
    LEFT_HIP,
    LEFT_KNEE,
    LEFT_ANKLE,
    RIGHT_HIP,
    RIGHT_KNEE,
    RIGHT_ANKLE,
]


# -----------------------------
# SMOOTHING
# -----------------------------
ANGLE_WINDOW = 5  # number of recent frames averaged into a smoothed angle


# -----------------------------
# CALIBRATION
# -----------------------------
CALIBRATION_SECONDS = 10.0  # accumulated *valid pose* seconds, not wall clock
MIN_CALIBRATION_SAMPLES = 15
MIN_MOVEMENT_RANGE_DEG = 20.0  # standing_angle - deep_angle must exceed this

# Robust percentiles instead of raw min/max so one noisy frame can't
# collapse or inflate the calibrated range.
CALIBRATION_LOW_PERCENTILE = 10
CALIBRATION_HIGH_PERCENTILE = 90

DEEP_THRESHOLD_RATIO = 0.60   # fraction of movement_range below standing angle
SHALLOW_THRESHOLD_RATIO = 0.15


# -----------------------------
# REP STATE MACHINE
# -----------------------------
# A candidate state transition must hold for this many consecutive valid
# frames before it "commits". This is the debounce that stops landmark
# jitter near a threshold from being counted as a phase change.
MIN_STATE_FRAMES = 3

# A completed rep shorter than this (seconds) is discarded as noise
# (e.g. a jitter spike that crossed both thresholds within one frame).
MIN_REP_DURATION_SEC = 0.3

BASELINE_REPS = 3


# -----------------------------
# RISK / MOVEMENT-QUALITY THRESHOLDS
# -----------------------------
ASYMMETRY_THRESHOLD = 12.0
ANGLE_VARIANCE_THRESHOLD = 18.0
FATIGUE_DEPTH_DROP_THRESHOLD = 12.0
FATIGUE_SPEED_DROP_THRESHOLD = 0.25

HIGH_RISK_SCORE = 3
MED_RISK_SCORE = 1


# -----------------------------
# LOGGING / DEBUG
# -----------------------------
LOG_PER_FRAME = True
DEBUG_MODE = False
