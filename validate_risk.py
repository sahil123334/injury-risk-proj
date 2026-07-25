"""
validate_risk.py

Accuracy harness for the risk engine's 4 individual flags (asymmetry,
instability, depth drop, speed drop) -- the counterpart to
validate_reps.py, but for risk scoring instead of rep counting.

"Risk" has no single ground truth the way a rep count does, so this
doesn't try to grade one fuzzy "was this risky" judgment. Instead it
grades each flag separately against deliberately engineered clips
where you know, by construction, whether that specific flag should or
shouldn't fire (e.g. a clip where you intentionally favored one leg
should trip the asymmetry flag; a clean control clip shouldn't).

Runs the exact same calibration -> rep tracking -> risk scoring path
main.py uses, replaying each clip frame by frame, and grades every
completed rep against what validation/risk_manifest.csv says should
have happened for it.

Usage:
    python validate_risk.py

Expects validation/risk_manifest.csv and the clips it lists under
validation/clips/ (same folder validate_reps.py uses). See
validation/README.md for the shot list.
"""

import csv
import os
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

import cv2

import config
from biomechanics import AngleSmoother, compute_knee_angles
from pose_detector import PoseDetector
from rep_tracker import RepTracker
from risk_engine import (
    REASON_ASYMMETRY,
    REASON_DEPTH_DROP,
    REASON_INSTABILITY,
    REASON_SPEED_DROP,
    RiskEngine,
)
from validate_reps import CLIPS_DIR, VALIDATION_DIR, extract_angle_stream, derive_thresholds
from video_source import FileVideoSource

RISK_MANIFEST_PATH = os.path.join(VALIDATION_DIR, "risk_manifest.csv")

FLAG_REASONS = {
    "asymmetry": REASON_ASYMMETRY,
    "instability": REASON_INSTABILITY,
    "depth_drop": REASON_DEPTH_DROP,
    "speed_drop": REASON_SPEED_DROP,
}


@dataclass
class RiskExpectation:
    filename: str
    flag: str
    expected: bool
    from_rep: int
    to_rep: Optional[int]
    notes: str

    def applies_to(self, rep_index: int) -> bool:
        if rep_index < self.from_rep:
            return False
        if self.to_rep is not None and rep_index > self.to_rep:
            return False
        return True


def load_risk_manifest() -> List[RiskExpectation]:
    if not os.path.isfile(RISK_MANIFEST_PATH):
        return []

    rows = []
    with open(RISK_MANIFEST_PATH, newline="") as f:
        for row in csv.DictReader(f):
            flag = row["flag"].strip()
            if flag not in FLAG_REASONS:
                raise ValueError(f"Unknown flag '{flag}' in risk_manifest.csv -- expected one of {list(FLAG_REASONS)}")
            to_rep_raw = row.get("to_rep", "").strip()
            rows.append(RiskExpectation(
                filename=row["filename"].strip(),
                flag=flag,
                expected=row["expected"].strip().lower() in ("yes", "true", "1"),
                from_rep=int(row.get("from_rep", "1") or 1),
                to_rep=int(to_rep_raw) if to_rep_raw else None,
                notes=row.get("notes", "").strip(),
            ))
    return rows


def run_clip(path: str, deep_thr: float, shallow_thr: float) -> Dict[int, Set[str]]:
    """
    Replays one clip through the same pipeline main.py uses, and returns
    {rep_index: set of flag reasons active in that rep's completion assessment}.
    """
    source = FileVideoSource(path)
    smoother = AngleSmoother()
    tracker = RepTracker(deep_thr, shallow_thr)
    engine = RiskEngine()

    rep_depths: List[float] = []
    rep_speeds: List[float] = []
    baseline_depth: Optional[float] = None
    baseline_speed: Optional[float] = None

    flags_by_rep: Dict[int, Set[str]] = {}
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

            rep_result = tracker.update(smoothed.mean, elapsed)
            if rep_result is not None:
                rep_depths.append(rep_result.depth_angle)
                rep_speeds.append(rep_result.speed)
                if baseline_depth is None and len(rep_depths) == config.BASELINE_REPS:
                    baseline_depth = sum(rep_depths[:config.BASELINE_REPS]) / config.BASELINE_REPS
                if baseline_speed is None and len(rep_speeds) == config.BASELINE_REPS:
                    baseline_speed = sum(rep_speeds[:config.BASELINE_REPS]) / config.BASELINE_REPS

            assessment = engine.evaluate(
                pose_confident=True,
                asymmetry=smoothed.asymmetry,
                variance=smoothed.variance,
                rep_state=tracker.state,
                rep_depths=rep_depths,
                rep_speeds=rep_speeds,
                baseline_depth=baseline_depth,
                baseline_speed=baseline_speed,
            )

            if rep_result is not None:
                flags_by_rep[rep_result.index] = set(assessment.reasons)

    source.release()
    return flags_by_rep


def main() -> int:
    expectations = load_risk_manifest()
    if not expectations:
        print(f"No risk expectations found. Fill in {RISK_MANIFEST_PATH} -- see validation/README.md.")
        return 1

    calibration_path = os.path.join(CLIPS_DIR, "calibration.mp4")
    if not os.path.isfile(calibration_path):
        print(f"Fatal: calibration clip not found at {calibration_path}.")
        return 1

    print("Deriving thresholds from calibration.mp4 (same as validate_reps.py)")
    cal_stream = extract_angle_stream(calibration_path)
    deep_thr, shallow_thr = derive_thresholds(cal_stream)
    print(f"  deep_threshold={deep_thr:.1f}  shallow_threshold={shallow_thr:.1f}\n")

    # Group expectations by clip so each clip is only replayed once.
    clips: Dict[str, List[RiskExpectation]] = {}
    for exp in expectations:
        clips.setdefault(exp.filename, []).append(exp)

    header = f"{'clip':<24}{'flag':<14}{'expect':<8}{'reps ok':<10}{'total':<7}notes"
    print(header)
    print("-" * len(header))

    total_reps_checked = 0
    total_reps_correct = 0

    for filename, clip_expectations in clips.items():
        clip_path = os.path.join(CLIPS_DIR, filename)
        if not os.path.isfile(clip_path):
            print(f"{filename:<24} -- file not found, skipping")
            continue

        flags_by_rep = run_clip(clip_path, deep_thr, shallow_thr)

        for exp in clip_expectations:
            target_reason = FLAG_REASONS[exp.flag]
            applicable_reps = [r for r in flags_by_rep if exp.applies_to(r)]

            correct = 0
            for rep_index in applicable_reps:
                is_active = target_reason in flags_by_rep[rep_index]
                if is_active == exp.expected:
                    correct += 1

            total = len(applicable_reps)
            total_reps_checked += total
            total_reps_correct += correct

            print(
                f"{filename:<24}{exp.flag:<14}{'yes' if exp.expected else 'no':<8}"
                f"{correct}/{total:<8}{total:<7}{exp.notes}"
            )

    print("-" * len(header))
    if total_reps_checked:
        pct = total_reps_correct / total_reps_checked * 100
        print(f"Overall: {total_reps_correct}/{total_reps_checked} rep-flag checks correct ({pct:.0f}%)")
    else:
        print("No reps were checked -- clips may be missing or produced no completed reps.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
