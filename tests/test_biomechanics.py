import math

import pytest

import config
from biomechanics import AngleSmoother, angle_between_three_points, compute_knee_angles
from conftest import make_landmarks


class TestAngleBetweenThreePoints:
    def test_straight_line_is_180_degrees(self):
        angle = angle_between_three_points((0, 0), (1, 0), (2, 0))
        assert angle == pytest.approx(180.0, abs=1e-3)

    def test_right_angle_is_90_degrees(self):
        angle = angle_between_three_points((1, 0), (0, 0), (0, 1))
        assert angle == pytest.approx(90.0, abs=1e-3)

    def test_60_degree_angle(self):
        # Equilateral triangle: vertex b, with a and c each 1 unit away at 60 degrees apart.
        b = (0, 0)
        a = (1, 0)
        c = (math.cos(math.radians(60)), math.sin(math.radians(60)))
        angle = angle_between_three_points(a, b, c)
        assert angle == pytest.approx(60.0, abs=1e-3)

    def test_zero_length_vector_returns_none(self):
        # a == b means ray b->a has zero length -- no angle is defined.
        assert angle_between_three_points((1, 1), (1, 1), (2, 2)) is None

    def test_both_zero_length_returns_none(self):
        assert angle_between_three_points((0, 0), (0, 0), (0, 0)) is None


class TestComputeKneeAngles:
    def _landmarks_with_leg_positions(self, left_pts, right_pts, left_vis=1.0, right_vis=1.0):
        landmarks = make_landmarks(33)
        for idx, (x, y) in zip((config.LEFT_HIP, config.LEFT_KNEE, config.LEFT_ANKLE), left_pts):
            landmarks[idx].x, landmarks[idx].y, landmarks[idx].visibility = x, y, left_vis
        for idx, (x, y) in zip((config.RIGHT_HIP, config.RIGHT_KNEE, config.RIGHT_ANKLE), right_pts):
            landmarks[idx].x, landmarks[idx].y, landmarks[idx].visibility = x, y, right_vis
        return landmarks

    def test_straight_leg_gives_near_180(self):
        # Both legs vertically straight: hip -> knee -> ankle in a line.
        straight = [(0.5, 0.0), (0.5, 0.5), (0.5, 1.0)]
        landmarks = self._landmarks_with_leg_positions(straight, straight)

        result = compute_knee_angles(landmarks, width=100, height=100)

        assert result is not None
        assert result.left == pytest.approx(180.0, abs=1e-2)
        assert result.right == pytest.approx(180.0, abs=1e-2)

    def test_bent_knee_gives_less_than_180(self):
        bent = [(0.5, 0.0), (0.5, 0.5), (0.3, 0.7)]  # ankle kicked out to the side
        landmarks = self._landmarks_with_leg_positions(bent, bent)

        result = compute_knee_angles(landmarks, width=100, height=100)

        assert result is not None
        assert result.left < 180.0

    def test_degenerate_landmarks_return_none(self):
        # Hip and knee at the exact same point -- no defined angle.
        same_point = [(0.5, 0.5), (0.5, 0.5), (0.5, 1.0)]
        landmarks = self._landmarks_with_leg_positions(same_point, same_point)

        assert compute_knee_angles(landmarks, width=100, height=100) is None

    def test_confidence_reflects_landmark_visibility(self):
        straight = [(0.5, 0.0), (0.5, 0.5), (0.5, 1.0)]
        landmarks = self._landmarks_with_leg_positions(straight, straight, left_vis=0.3, right_vis=0.9)

        result = compute_knee_angles(landmarks, width=100, height=100)

        assert result.left_confidence == pytest.approx(0.3)
        assert result.right_confidence == pytest.approx(0.9)


class TestAngleSmoother:
    def test_constant_input_converges_to_that_value(self):
        smoother = AngleSmoother(window=5)
        for _ in range(5):
            result = smoother.update(150.0, 150.0)

        assert result.left == pytest.approx(150.0)
        assert result.right == pytest.approx(150.0)
        assert result.asymmetry == pytest.approx(0.0)

    def test_flat_average_when_confidence_uniform(self):
        smoother = AngleSmoother(window=3)
        smoother.update(100.0, 100.0, 1.0, 1.0)
        smoother.update(110.0, 100.0, 1.0, 1.0)
        result = smoother.update(120.0, 100.0, 1.0, 1.0)

        assert result.left == pytest.approx((100.0 + 110.0 + 120.0) / 3)

    def test_low_confidence_outlier_is_suppressed(self):
        smoother = AngleSmoother(window=5)
        for _ in range(4):
            smoother.update(150.0, 150.0, 1.0, 1.0)
        # A noisy outlier frame with low confidence shouldn't swing the
        # smoothed value nearly as much as it would under a flat average.
        result = smoother.update(170.0, 150.0, 0.1, 1.0)

        flat_average = (150.0 * 4 + 170.0) / 5  # == 154.0
        assert result.left < flat_average
        assert result.left == pytest.approx(150.49, abs=0.05)

    def test_zero_confidence_window_falls_back_to_flat_average(self):
        smoother = AngleSmoother(window=3)
        smoother.update(100.0, 100.0, 0.0, 0.0)
        smoother.update(200.0, 100.0, 0.0, 0.0)
        result = smoother.update(300.0, 100.0, 0.0, 0.0)

        # Guards against dividing by ~zero total weight.
        assert result.left == pytest.approx((100.0 + 200.0 + 300.0) / 3)

    def test_window_evicts_oldest_sample(self):
        smoother = AngleSmoother(window=2)
        smoother.update(10.0, 10.0)
        smoother.update(20.0, 10.0)
        result = smoother.update(30.0, 10.0)

        # Only the last 2 samples (20, 30) should still be in the window.
        assert result.left == pytest.approx(25.0)

    def test_variance_zero_on_first_sample(self):
        smoother = AngleSmoother(window=5)
        result = smoother.update(150.0, 150.0)

        assert result.variance == 0.0

    def test_variance_reflects_spread(self):
        smoother = AngleSmoother(window=5)
        for value in (140.0, 160.0, 140.0, 160.0):
            result = smoother.update(value, value)

        assert result.variance > 0.0
