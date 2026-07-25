import pytest

from rep_tracker import RepState
from risk_engine import (
    REASON_ASYMMETRY,
    REASON_DEPTH_DROP,
    REASON_INSTABILITY,
    REASON_SPEED_DROP,
    RiskEngine,
)


def base_kwargs(**overrides):
    kwargs = dict(
        pose_confident=True,
        asymmetry=0.0,
        variance=0.0,
        rep_state=RepState.STANDING,
        rep_depths=[],
        rep_speeds=[],
        baseline_depth=None,
        baseline_speed=None,
    )
    kwargs.update(overrides)
    return kwargs


class TestPoseConfidence:
    def test_low_confidence_pauses_regardless_of_other_values(self):
        engine = RiskEngine()
        assessment = engine.evaluate(**base_kwargs(
            pose_confident=False, asymmetry=999.0, variance=999.0,
        ))

        assert assessment.confident is False
        assert assessment.score == 0
        assert assessment.label == "GREEN"
        assert assessment.reasons == []


class TestAsymmetry:
    def test_flags_when_over_threshold(self):
        engine = RiskEngine(asymmetry_threshold=12.0)
        assessment = engine.evaluate(**base_kwargs(asymmetry=13.0))

        assert REASON_ASYMMETRY in assessment.reasons
        assert assessment.score >= 1

    def test_does_not_flag_at_or_under_threshold(self):
        engine = RiskEngine(asymmetry_threshold=12.0)
        assessment = engine.evaluate(**base_kwargs(asymmetry=12.0))

        assert REASON_ASYMMETRY not in assessment.reasons


class TestInstability:
    @pytest.mark.parametrize("state", [RepState.STANDING, RepState.BOTTOM])
    def test_flags_during_static_phases(self, state):
        engine = RiskEngine(variance_threshold=18.0)
        assessment = engine.evaluate(**base_kwargs(variance=25.0, rep_state=state))

        assert REASON_INSTABILITY in assessment.reasons

    @pytest.mark.parametrize("state", [RepState.DESCENDING, RepState.ASCENDING])
    def test_never_flags_during_active_movement(self, state):
        # Angle naturally changes fast while actively moving -- that's not
        # instability, and should never be flagged no matter how high the
        # variance reads during that phase.
        engine = RiskEngine(variance_threshold=18.0)
        assessment = engine.evaluate(**base_kwargs(variance=999.0, rep_state=state))

        assert REASON_INSTABILITY not in assessment.reasons


class TestDepthDrop:
    def test_flags_when_shallower_than_baseline_past_threshold(self):
        engine = RiskEngine(depth_drop_threshold=12.0)
        assessment = engine.evaluate(**base_kwargs(
            rep_depths=[80.0, 82.0, 81.0, 100.0],  # rep 4 is 19deg shallower than baseline
            baseline_depth=81.0,
        ))

        assert REASON_DEPTH_DROP in assessment.reasons
        assert assessment.depth_drop == pytest.approx(100.0 - 81.0)

    def test_does_not_flag_before_baseline_reps_exceeded(self):
        # Only 3 reps so far (== baseline_reps) -- baseline exists but
        # there's no *post*-baseline rep yet to compare.
        engine = RiskEngine(depth_drop_threshold=12.0)
        assessment = engine.evaluate(**base_kwargs(
            rep_depths=[80.0, 82.0, 130.0],
            baseline_depth=81.0,
        ))

        assert REASON_DEPTH_DROP not in assessment.reasons
        assert assessment.depth_drop == 0.0

    def test_does_not_flag_when_baseline_missing(self):
        engine = RiskEngine(depth_drop_threshold=12.0)
        assessment = engine.evaluate(**base_kwargs(
            rep_depths=[80.0, 82.0, 81.0, 150.0],
            baseline_depth=None,
        ))

        assert REASON_DEPTH_DROP not in assessment.reasons

    def test_a_deeper_rep_than_baseline_is_not_flagged(self):
        engine = RiskEngine(depth_drop_threshold=12.0)
        assessment = engine.evaluate(**base_kwargs(
            rep_depths=[80.0, 82.0, 81.0, 60.0],  # deeper (lower angle), not shallower
            baseline_depth=81.0,
        ))

        assert REASON_DEPTH_DROP not in assessment.reasons
        assert assessment.depth_drop < 0


class TestSpeedDrop:
    def test_flags_when_slower_than_baseline_past_threshold(self):
        engine = RiskEngine(speed_drop_threshold=0.25)
        assessment = engine.evaluate(**base_kwargs(
            rep_speeds=[1.0, 1.0, 1.0, 0.5],  # 50% slower than baseline
            baseline_speed=1.0,
        ))

        assert REASON_SPEED_DROP in assessment.reasons
        assert assessment.speed_drop_pct == pytest.approx(0.5)

    def test_zero_baseline_speed_does_not_crash(self):
        engine = RiskEngine()
        assessment = engine.evaluate(**base_kwargs(
            rep_speeds=[0.0, 0.0, 0.0, 0.0],
            baseline_speed=0.0,
        ))

        assert REASON_SPEED_DROP not in assessment.reasons
        assert assessment.speed_drop_pct == 0.0

    def test_faster_rep_than_baseline_is_not_flagged(self):
        engine = RiskEngine(speed_drop_threshold=0.25)
        assessment = engine.evaluate(**base_kwargs(
            rep_speeds=[1.0, 1.0, 1.0, 1.5],
            baseline_speed=1.0,
        ))

        assert REASON_SPEED_DROP not in assessment.reasons


class TestScoreToLabelMapping:
    def test_zero_score_is_green_nominal(self):
        engine = RiskEngine()
        assessment = engine.evaluate(**base_kwargs())

        assert assessment.score == 0
        assert assessment.label == "GREEN"
        assert assessment.status_text == "NOMINAL"

    def test_med_score_is_yellow(self):
        engine = RiskEngine(asymmetry_threshold=12.0, med_score=1, high_score=3)
        assessment = engine.evaluate(**base_kwargs(asymmetry=13.0))

        assert assessment.score == 1
        assert assessment.label == "YELLOW"
        assert assessment.status_text == "MOVEMENT-QUALITY WARNING"

    def test_high_score_is_red(self):
        engine = RiskEngine(
            asymmetry_threshold=12.0, variance_threshold=18.0,
            depth_drop_threshold=12.0, speed_drop_threshold=0.25,
            med_score=1, high_score=3,
        )
        assessment = engine.evaluate(**base_kwargs(
            asymmetry=13.0,
            variance=25.0,
            rep_state=RepState.STANDING,
            rep_depths=[80.0, 82.0, 81.0, 100.0],
            baseline_depth=81.0,
            rep_speeds=[1.0, 1.0, 1.0, 0.5],
            baseline_speed=1.0,
        ))

        assert assessment.score == 4
        assert assessment.label == "RED"
        assert assessment.status_text == "REVIEW RECOMMENDED"
