"""
ui.py

All cv2 drawing lives here. main.py calls exactly one of these
draw_*_overlay functions per frame (depending on what state the
pipeline is in), then does a single cv2.imshow + cv2.waitKey itself.
Keeping that imshow/waitKey call in main.py (not here) is what
guarantees the camera feed is always shown, even on frames with no
usable pose -- one of the bugs called out in the refactor brief.
"""

from typing import Iterable, List, Optional, Tuple

import cv2

STATUS_COLORS = {
    "GREEN": (0, 180, 0),
    "YELLOW": (0, 215, 255),
    "RED": (0, 0, 255),
}

STOP_BUTTON_SIZE = (168, 46)  # (width, height) in pixels
STOP_BUTTON_MARGIN = 16


def draw_text(frame, text: str, x: int, y: int, size: float = 0.65, color=(255, 255, 255)) -> None:
    cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, size, color, 2, cv2.LINE_AA)


def draw_stop_button(frame, width: int, _height: int) -> Tuple[int, int, int, int]:
    """
    Draws a clickable-looking "End session" button in the top-right corner
    and returns its (x1, y1, x2, y2) pixel bounds so main.py can hit-test
    mouse clicks against it. Redrawn every frame since the frame width can
    change (e.g. a differently-sized uploaded video).
    """
    btn_w, btn_h = STOP_BUTTON_SIZE
    x2 = width - STOP_BUTTON_MARGIN
    x1 = x2 - btn_w
    y1 = STOP_BUTTON_MARGIN
    y2 = y1 + btn_h

    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 200), -1)
    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 1, cv2.LINE_AA)
    draw_text(frame, "END SESSION", x1 + 12, y1 + 20, 0.5)
    draw_text(frame, "(or press Q)", x1 + 16, y1 + 38, 0.38, color=(225, 225, 225))

    return (x1, y1, x2, y2)


def draw_pose_landmarks(frame, landmarks, width: int, height: int) -> None:
    for lm in landmarks:
        x = int(lm.x * width)
        y = int(lm.y * height)
        cv2.circle(frame, (x, y), 3, (0, 255, 0), -1)


def draw_status_box(frame, status_text: str, label: str, score: int) -> None:
    color = STATUS_COLORS.get(label, (200, 200, 200))
    cv2.rectangle(frame, (20, 20), (340, 95), color, -1)
    draw_text(frame, f"STATUS: {status_text}", 30, 55, 0.75)
    draw_text(frame, f"Score: {score}", 30, 82, 0.6)


def draw_no_pose_overlay(frame) -> None:
    draw_text(frame, "No pose detected. Step back / improve lighting.", 20, 40)
    draw_text(frame, "Camera is still running. Analysis paused.", 20, 70)


def draw_low_confidence_overlay(frame, confidence: float) -> None:
    draw_text(frame, "Move back: hips, knees, ankles must be fully visible", 20, 40)
    draw_text(frame, f"Analysis paused (pose confidence: {confidence * 100:.0f}%)", 20, 70)


def draw_calibration_overlay(frame, time_remaining: float, progress: float, current_angle: float) -> None:
    draw_text(frame, "CALIBRATING: perform 2-3 clean, deep reps", 20, 130)
    draw_text(frame, f"Time remaining (valid pose): {time_remaining:.1f}s", 20, 160)
    draw_text(frame, f"Progress: {progress * 100:.0f}%", 20, 190)
    draw_text(frame, f"Current knee angle: {current_angle:.1f}", 20, 220)


def draw_calibration_reset_overlay(frame, reason: str) -> None:
    draw_text(frame, "Calibration restarted:", 20, 130, color=(0, 215, 255))
    draw_text(frame, reason, 20, 160, size=0.55, color=(0, 215, 255))


def draw_tracking_overlay(
    frame,
    *,
    rep_count: int,
    smooth_left: float,
    smooth_right: float,
    smooth_mean: float,
    asymmetry: float,
    variance: float,
    depth_drop: float,
    speed_drop_pct: float,
    baseline_established: bool,
    baseline_reps: int,
    reasons: Iterable[str],
) -> None:
    draw_text(frame, f"Reps: {rep_count}", 20, 135)
    draw_text(frame, f"Left knee: {smooth_left:.1f}", 20, 165)
    draw_text(frame, f"Right knee: {smooth_right:.1f}", 20, 195)
    draw_text(frame, f"Mean knee: {smooth_mean:.1f}", 20, 225)
    draw_text(frame, f"Asymmetry: {asymmetry:.1f}", 20, 255)
    draw_text(frame, f"Variance: {variance:.1f}", 20, 285)
    draw_text(frame, f"Depth drop: {depth_drop:.1f}", 20, 315)
    draw_text(frame, f"Speed drop: {speed_drop_pct * 100:.1f}%", 20, 345)

    if not baseline_established:
        draw_text(frame, f"Building baseline: {rep_count}/{baseline_reps} reps", 20, 385, 0.55)

    reasons_list = list(reasons)
    if reasons_list:
        draw_text(frame, "Flags: " + ", ".join(reasons_list), 20, 415, 0.55)

    draw_text(frame, "Prototype: fatigue/movement-quality proxy, not a medical assessment", 20, 445, 0.45, color=(180, 180, 180))
