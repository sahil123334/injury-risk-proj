"""
rep_tracker.py

A small finite-state machine (FSM) that turns a stream of smoothed knee
angles into discrete, validated reps.

States: STANDING -> DESCENDING -> BOTTOM -> ASCENDING -> STANDING

Why an FSM instead of the original two-threshold flag: a single
"in_rep" boolean can't distinguish "still descending" from "at the
bottom" from "coming back up", so it can't measure eccentric vs.
concentric duration, and it has no debounce -- a single noisy frame
that blips across a threshold gets counted as a phase change. Here,
a candidate transition must hold for `min_state_frames` consecutive
frames before it commits (see `update`), and a completed rep shorter
than `min_rep_duration` is discarded as noise rather than counted.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional

import config


class RepState(Enum):
    STANDING = "standing"
    DESCENDING = "descending"
    BOTTOM = "bottom"
    ASCENDING = "ascending"


@dataclass
class RepResult:
    index: int
    depth_angle: float          # minimum smoothed knee angle reached
    duration: float              # descend-start -> standing (full rep)
    eccentric_duration: float    # descend-start -> bottom-start
    concentric_duration: float   # ascend-start -> standing
    speed: float                 # 1 / duration proxy, guarded against divide-by-zero


class RepTracker:
    """Construct one of these only after calibration succeeds -- it needs real thresholds."""

    def __init__(
        self,
        deep_threshold: float,
        shallow_threshold: float,
        min_state_frames: int = config.MIN_STATE_FRAMES,
        min_rep_duration: float = config.MIN_REP_DURATION_SEC,
        debug: bool = False,
    ):
        self.deep_threshold = deep_threshold
        self.shallow_threshold = shallow_threshold
        self._min_state_frames = min_state_frames
        self._min_rep_duration = min_rep_duration
        self._debug = debug

        self.state = RepState.STANDING
        self._pending_state: Optional[RepState] = None
        self._pending_count = 0

        self._t_descend_start: Optional[float] = None
        self._t_bottom_start: Optional[float] = None
        self._t_ascend_start: Optional[float] = None
        self._min_angle_this_rep: Optional[float] = None

        self._rep_index = 0

    @property
    def rep_count(self) -> int:
        return self._rep_index

    def update(self, angle: float, now: float) -> Optional[RepResult]:
        """Feed one smoothed-angle sample. Returns a RepResult only on rep completion."""
        target_state = self._candidate_state(angle)

        # Track the deepest angle seen while we're anywhere in the down phase,
        # even before a transition commits, so a short BOTTOM dwell doesn't
        # miss the true minimum.
        if self.state in (RepState.DESCENDING, RepState.BOTTOM):
            self._min_angle_this_rep = (
                angle if self._min_angle_this_rep is None else min(self._min_angle_this_rep, angle)
            )

        if target_state == self.state:
            self._pending_state = None
            self._pending_count = 0
            return None

        if target_state != self._pending_state:
            self._pending_state = target_state
            self._pending_count = 1
        else:
            self._pending_count += 1

        if self._pending_count < self._min_state_frames:
            return None  # candidate transition hasn't held long enough yet -- debounced

        return self._commit_transition(target_state, now, angle)

    def _candidate_state(self, angle: float) -> RepState:
        """What state does this single angle sample suggest, given where we are now?"""
        if self.state == RepState.STANDING:
            return RepState.DESCENDING if angle < self.shallow_threshold else RepState.STANDING

        if self.state == RepState.DESCENDING:
            if angle < self.deep_threshold:
                return RepState.BOTTOM
            if angle >= self.shallow_threshold:
                return RepState.STANDING  # bailed out before reaching bottom
            return RepState.DESCENDING

        if self.state == RepState.BOTTOM:
            return RepState.ASCENDING if angle > self.deep_threshold else RepState.BOTTOM

        if self.state == RepState.ASCENDING:
            if angle > self.shallow_threshold:
                return RepState.STANDING
            if angle <= self.deep_threshold:
                return RepState.BOTTOM  # dipped back down without completing
            return RepState.ASCENDING

        return self.state

    def _commit_transition(self, new_state: RepState, now: float, angle: float) -> Optional[RepResult]:
        old_state = self.state
        result: Optional[RepResult] = None

        if new_state == RepState.DESCENDING and old_state == RepState.STANDING:
            self._t_descend_start = now
            self._min_angle_this_rep = angle

        elif new_state == RepState.BOTTOM and old_state == RepState.DESCENDING:
            self._t_bottom_start = now

        elif new_state == RepState.ASCENDING and old_state == RepState.BOTTOM:
            self._t_ascend_start = now

        elif new_state == RepState.STANDING and old_state == RepState.ASCENDING:
            result = self._finish_rep(now)

        elif new_state == RepState.BOTTOM and old_state == RepState.ASCENDING:
            # dipped back below deep_threshold before finishing -- extend the hold
            self._t_bottom_start = now

        # (STANDING <- DESCENDING, i.e. bailed out early, needs no bookkeeping;
        #  the partial rep is simply dropped.)

        self.state = new_state
        self._pending_state = None
        self._pending_count = 0

        if self._debug:
            print(f"[rep_tracker] {old_state.value} -> {new_state.value} @ {now:.2f}s (angle={angle:.1f})")

        return result

    def _finish_rep(self, now: float) -> Optional[RepResult]:
        if self._t_descend_start is None or self._t_bottom_start is None or self._t_ascend_start is None:
            # Shouldn't normally happen, but never let incomplete timing data
            # masquerade as a valid rep.
            self._reset_rep_timers()
            return None

        duration = now - self._t_descend_start
        if duration < self._min_rep_duration:
            self._reset_rep_timers()
            return None

        eccentric_duration = max(self._t_bottom_start - self._t_descend_start, 0.0)
        concentric_duration = max(now - self._t_ascend_start, 0.0)
        depth_angle = self._min_angle_this_rep if self._min_angle_this_rep is not None else float("nan")
        speed = (1.0 / duration) if duration > 0 else 0.0

        self._rep_index += 1
        result = RepResult(
            index=self._rep_index,
            depth_angle=depth_angle,
            duration=duration,
            eccentric_duration=eccentric_duration,
            concentric_duration=concentric_duration,
            speed=speed,
        )

        self._reset_rep_timers()
        return result

    def _reset_rep_timers(self) -> None:
        self._t_descend_start = None
        self._t_bottom_start = None
        self._t_ascend_start = None
        self._min_angle_this_rep = None
