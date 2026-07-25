"""
validate_reps.py

Headless accuracy harness -- the tool behind the "improved rep-counting
accuracy from X% to Y%" resume claim. Runs both the pre-refactor naive
rep counter (naive_rep_counter.py) and the modern debounced FSM
(rep_tracker.py) over the same set of labeled video clips, and reports
how often each matches the human-labeled true rep count.

Both counters are fed the exact same smoothed-angle stream and the
exact same calibrated thresholds, so any accuracy difference is
attributable only to the counting algorithm itself -- not to a
different camera angle, a different calibration, or luck.

Usage:
    python validate_reps.py

Expects validation/manifest.csv and the clips it lists under
validation/clips/. See validation/README.md for the recording shot
list and manifest format.
"""

import csv
import os
import sys
from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2

import config
from biomechanics import AngleSmoother, compute_knee_angles
from calibration import Calibrator
from naive_rep_counter import NaiveRepCounter
from pose_detector import PoseDetector
from rep_tracker import RepTracker
from video_source import FileVideoSource

VALIDATION_DIR = os.path.join(config.PROJECT_ROOT, "validation")
MANIFEST_PATH = os.path.join(VALIDATION_DIR, "manifest.csv")
CLIPS_DIR = os.path.join(VALIDATION_DIR, "clips")

AngleStream = List[Tuple[float, float]]  # (elapsed_seconds, smoothed_mean_angle)


@dataclass
class ClipSpec:
    filename: str
    true_reps: int
    notes: str
    is_calibration_source: bool


def load_manifest() -> List[ClipSpec]:
    if not os.path.isfile(MANIFEST_PATH):
        return []

    clips = []
    with open(MANIFEST_PATH, newline="") as f:
        for row in csv.DictReader(f):
            clips.append(ClipSpec(
                filename=row["filename"].strip(),
                true_reps=int(row["true_reps"]),
                notes=row.get("notes", "").strip(),
                is_calibration_source=row.get("is_calibration_source", "").strip().lower() in ("1", "true", "yes"),
            ))
    return clips


def extract_angle_stream(path: str) -> AngleStream:
    """
    Every valid-pose frame's (elapsed, smoothed mean knee angle), in order.

    Opens its own fresh PoseDetector rather than taking a shared one: a
    single MediaPipe landmarker instance requires strictly increasing
    timestamps for its entire lifetime, and each video file's timestamps
    restart near zero, so reusing one landmarker across multiple clips
    trips MediaPipe's monotonic-timestamp check.
    """
    source = FileVideoSource(path)
    smoother = AngleSmoother()
    stream: AngleStream = []
    last_timestamp_ms = -1

    with PoseDetector(config.MODEL_PATH) as detector:
        while True:
            ret, frame, elapsed = source.read()
            if not ret:
                break

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            timestamp_ms = max(int(elapsed * 1000), last_timestamp_ms + 1)
            last_timestamp_ms = timestamp_ms
            reading = detector.detect(rgb_frame, timestamp_ms)

            if reading.landmarks is None or not reading.lower_body_visible:
                continue

            raw_angles = compute_knee_angles(reading.landmarks, frame.shape[1], frame.shape[0])
            if raw_angles is None:
                continue

            smoothed = smoother.update(
                raw_angles.left, raw_angles.right,
                raw_angles.left_confidence, raw_angles.right_confidence,
            )
            stream.append((elapsed, smoothed.mean))

    source.release()
    return stream


def derive_thresholds(stream: AngleStream) -> Tuple[float, float]:
    calibrator = Calibrator()
    last_t: Optional[float] = None

    for t, angle in stream:
        dt = 0.0 if last_t is None else max(t - last_t, 0.0)
        last_t = t
        calibrator.update(angle, dt)
        if calibrator.is_calibrated:
            break

    if not calibrator.is_calibrated:
        raise RuntimeError(
            "The calibration clip never produced a successful calibration -- "
            "it needs 2-3 clean, deep reps within its first ~10s of valid pose time."
        )

    result = calibrator.result
    return result.deep_threshold, result.shallow_threshold


def count_naive(stream: AngleStream, deep_thr: float, shallow_thr: float) -> int:
    counter = NaiveRepCounter(deep_thr, shallow_thr)
    for _, angle in stream:
        counter.update(angle)
    return counter.rep_count


def count_fsm(stream: AngleStream, deep_thr: float, shallow_thr: float) -> int:
    tracker = RepTracker(deep_thr, shallow_thr)
    count = 0
    for t, angle in stream:
        if tracker.update(angle, t) is not None:
            count += 1
    return count


def main() -> int:
    clips = load_manifest()
    if not clips:
        print(f"No labeled clips found. Fill in {MANIFEST_PATH} -- see validation/README.md.")
        return 1

    calibration_clip = next((c for c in clips if c.is_calibration_source), clips[0])
    calibration_path = os.path.join(CLIPS_DIR, calibration_clip.filename)
    if not os.path.isfile(calibration_path):
        print(f"Fatal: calibration clip '{calibration_clip.filename}' not found in {CLIPS_DIR}.")
        print("Record it first -- see validation/README.md.")
        return 1

    print(f"Deriving thresholds from calibration clip: {calibration_clip.filename}")
    cal_stream = extract_angle_stream(calibration_path)
    try:
        deep_thr, shallow_thr = derive_thresholds(cal_stream)
    except RuntimeError as exc:
        print(f"Fatal: {exc}")
        return 1
    print(f"  deep_threshold={deep_thr:.1f}  shallow_threshold={shallow_thr:.1f}\n")

    header = f"{'clip':<28}{'true':>6}{'naive':>8}{'ok':>5}{'fsm':>8}{'ok':>5}  notes"
    print(header)
    print("-" * len(header))

    naive_correct = 0
    fsm_correct = 0

    for clip in clips:
        clip_path = os.path.join(CLIPS_DIR, clip.filename)
        if not os.path.isfile(clip_path):
            print(f"{clip.filename:<28} -- file not found, skipping")
            continue

        stream = extract_angle_stream(clip_path)
        naive_reps = count_naive(stream, deep_thr, shallow_thr)
        fsm_reps = count_fsm(stream, deep_thr, shallow_thr)

        naive_ok = naive_reps == clip.true_reps
        fsm_ok = fsm_reps == clip.true_reps
        naive_correct += naive_ok
        fsm_correct += fsm_ok

        print(
            f"{clip.filename:<28}{clip.true_reps:>6}{naive_reps:>8}"
            f"{'Y' if naive_ok else 'N':>5}{fsm_reps:>8}{'Y' if fsm_ok else 'N':>5}  {clip.notes}"
        )

    total = len(clips)
    print("-" * len(header))
    print(f"Naive accuracy: {naive_correct}/{total} ({naive_correct / total * 100:.0f}%)")
    print(f"FSM accuracy:   {fsm_correct}/{total} ({fsm_correct / total * 100:.0f}%)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
