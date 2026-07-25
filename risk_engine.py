"""
risk_engine.py

Turns biomechanical features into a plain-language movement-quality
readout. This is an early prototype, not a medical device -- the
language and the docstrings in this file are deliberately careful:

  - We say "elevated movement-risk indicator", "form degradation",
    "fatigue proxy", "movement-quality warning", "review recommended".
  - We never say "injury detected", "injury will occur", "medically
    safe", or "medically cleared". This tool does not diagnose or
    predict injury; it flags patterns in movement that may be worth a
    coach or clinician's attention.

The engine only ever runs on frames the rest of the pipeline has
already deemed reliable (pose visible, in bounds, calibrated). If
`pose_confident` is False, it returns a paused/neutral assessment
rather than guessing from stale numbers.
"""

from dataclasses import dataclass, field
from typing import List, Optional

import config
from rep_tracker import RepState

# Exposed as constants (not just inline strings in evaluate()) so
# validate_risk.py can check for a specific flag reliably instead of
# duplicating the exact wording.
REASON_ASYMMETRY = "left/right asymmetry"
REASON_INSTABILITY = "angle instability"
REASON_DEPTH_DROP = "form degradation: shallower reps"
REASON_SPEED_DROP = "fatigue proxy: slower reps"


@dataclass
class RiskAssessment:
    score: int
    label: str          # "GREEN" / "YELLOW" / "RED" -- used for UI color only
    status_text: str     # human-readable, non-medical phrasing
    reasons: List[str] = field(default_factory=list)
    confident: bool = True
    depth_drop: float = 0.0
    speed_drop_pct: float = 0.0


class RiskEngine:
    def __init__(
        self,
        asymmetry_threshold: float = config.ASYMMETRY_THRESHOLD,
        variance_threshold: float = config.ANGLE_VARIANCE_THRESHOLD,
        depth_drop_threshold: float = config.FATIGUE_DEPTH_DROP_THRESHOLD,
        speed_drop_threshold: float = config.FATIGUE_SPEED_DROP_THRESHOLD,
        high_score: int = config.HIGH_RISK_SCORE,
        med_score: int = config.MED_RISK_SCORE,
    ):
        self._asymmetry_threshold = asymmetry_threshold
        self._variance_threshold = variance_threshold
        self._depth_drop_threshold = depth_drop_threshold
        self._speed_drop_threshold = speed_drop_threshold
        self._high_score = high_score
        self._med_score = med_score

    def evaluate(
        self,
        *,
        pose_confident: bool,
        asymmetry: float,
        variance: float,
        rep_state: RepState,
        rep_depths: List[float],
        rep_speeds: List[float],
        baseline_depth: Optional[float],
        baseline_speed: Optional[float],
        baseline_reps: int = config.BASELINE_REPS,
    ) -> RiskAssessment:
        if not pose_confident:
            return RiskAssessment(
                score=0,
                label="GREEN",
                status_text="PAUSED - pose not reliable",
                reasons=[],
                confident=False,
            )

        score = 0
        reasons: List[str] = []

        if asymmetry > self._asymmetry_threshold:
            score += 1
            reasons.append(REASON_ASYMMETRY)

        # Angle variance is naturally high while actively descending/ascending
        # through a squat -- that's real motion, not instability. Only treat
        # variance as a movement-quality flag while the athlete is holding
        # near standing or near the bottom of the rep.
        if rep_state in (RepState.STANDING, RepState.BOTTOM) and variance > self._variance_threshold:
            score += 1
            reasons.append(REASON_INSTABILITY)

        depth_drop = 0.0
        if baseline_depth is not None and len(rep_depths) > baseline_reps and rep_depths:
            depth_drop = rep_depths[-1] - baseline_depth
            if depth_drop > self._depth_drop_threshold:
                score += 1
                reasons.append(REASON_DEPTH_DROP)

        speed_drop_pct = 0.0
        if (
            baseline_speed is not None
            and baseline_speed > 0
            and len(rep_speeds) > baseline_reps
            and rep_speeds
        ):
            speed_drop_pct = (baseline_speed - rep_speeds[-1]) / baseline_speed
            if speed_drop_pct > self._speed_drop_threshold:
                score += 1
                reasons.append(REASON_SPEED_DROP)

        if score >= self._high_score:
            label, status_text = "RED", "REVIEW RECOMMENDED"
        elif score >= self._med_score:
            label, status_text = "YELLOW", "MOVEMENT-QUALITY WARNING"
        else:
            label, status_text = "GREEN", "NOMINAL"

        return RiskAssessment(
            score=score,
            label=label,
            status_text=status_text,
            reasons=reasons,
            confident=True,
            depth_drop=depth_drop,
            speed_drop_pct=speed_drop_pct,
        )
