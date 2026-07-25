import pytest

from calibration import Calibrator


def feed_triangle_wave(calibrator, total_seconds, dt, low=80.0, high=170.0, cycle_seconds=1.0):
    """Feeds a clean squat-like triangle wave (standing <-> deep) into a calibrator."""
    t = 0.0
    while t < total_seconds:
        phase = (t % cycle_seconds) / cycle_seconds
        angle = high - (high - low) * (1 - abs(2 * phase - 1))
        calibrator.update(angle, dt)
        t += dt


class TestCalibrator:
    def test_completes_with_sufficient_movement(self):
        cal = Calibrator(required_seconds=2.0, min_samples=5, min_movement_range=20.0)
        feed_triangle_wave(cal, total_seconds=2.5, dt=1 / 30)

        assert cal.is_calibrated
        result = cal.result
        assert result.standing_angle > result.deep_angle
        # deep_threshold should sit between deep_angle and standing_angle.
        assert result.deep_angle <= result.deep_threshold <= result.standing_angle
        assert result.deep_threshold < result.shallow_threshold < result.standing_angle

    def test_resets_when_movement_range_too_small(self):
        cal = Calibrator(required_seconds=1.0, min_samples=5, min_movement_range=20.0)
        # Barely any movement: 170 to 165, range = 5 degrees.
        feed_triangle_wave(cal, total_seconds=1.5, dt=1 / 30, low=165.0, high=170.0)

        assert not cal.is_calibrated
        assert "movement range too small" in cal.last_reset_reason

    def test_resets_when_not_enough_samples(self):
        cal = Calibrator(required_seconds=1.0, min_samples=50, min_movement_range=20.0)
        # A single huge dt jumps past required_seconds in one sample --
        # nowhere near the 50 samples required.
        cal.update(150.0, 5.0)

        assert not cal.is_calibrated
        assert "not enough valid-pose samples" in cal.last_reset_reason

    def test_zero_dt_frames_do_not_advance_progress(self):
        cal = Calibrator(required_seconds=1.0, min_samples=5, min_movement_range=20.0)
        for _ in range(10):
            cal.update(150.0, 0.0)

        assert cal.progress == 0.0
        assert not cal.is_calibrated

    def test_progress_and_time_remaining(self):
        cal = Calibrator(required_seconds=10.0, min_samples=5, min_movement_range=20.0)
        cal.update(150.0, 4.0)

        assert cal.progress == pytest.approx(0.4)
        assert cal.time_remaining == pytest.approx(6.0)

    def test_progress_clamped_to_one(self):
        cal = Calibrator(required_seconds=1.0, min_samples=5, min_movement_range=20.0)
        feed_triangle_wave(cal, total_seconds=1.5, dt=1 / 30)

        assert cal.progress == 1.0

    def test_update_is_noop_once_calibrated(self):
        cal = Calibrator(required_seconds=1.0, min_samples=5, min_movement_range=20.0)
        feed_triangle_wave(cal, total_seconds=1.5, dt=1 / 30)
        assert cal.is_calibrated
        first_result = cal.result

        cal.update(999.0, 1.0)  # should be ignored entirely

        assert cal.result is first_result

    def test_manual_reset_clears_state(self):
        cal = Calibrator(required_seconds=1.0, min_samples=5, min_movement_range=20.0)
        feed_triangle_wave(cal, total_seconds=1.5, dt=1 / 30)
        assert cal.is_calibrated

        cal.reset("testing manual reset")

        assert not cal.is_calibrated
        assert cal.result is None
        assert cal.progress == 0.0
        assert cal.last_reset_reason == "testing manual reset"

    def test_negative_dt_does_not_decrease_valid_seconds(self):
        cal = Calibrator(required_seconds=10.0, min_samples=5, min_movement_range=20.0)
        cal.update(150.0, 5.0)
        cal.update(150.0, -3.0)  # a bogus negative dt shouldn't reduce progress

        assert cal.progress == pytest.approx(0.5)
