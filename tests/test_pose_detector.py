import pytest

from pose_detector import average_visibility, landmarks_in_bounds, landmarks_visible_enough
from conftest import make_landmarks

INDICES = [23, 25, 27]  # left hip/knee/ankle, matches config.LEFT_LEG_LANDMARKS


class TestLandmarksVisibleEnough:
    def test_all_visible_returns_true(self):
        landmarks = make_landmarks(33, visibility=0.9)
        assert landmarks_visible_enough(landmarks, INDICES, min_visibility=0.65) is True

    def test_one_below_threshold_returns_false(self):
        landmarks = make_landmarks(33, visibility=0.9)
        landmarks[25].visibility = 0.4
        assert landmarks_visible_enough(landmarks, INDICES, min_visibility=0.65) is False

    def test_exactly_at_threshold_counts_as_visible(self):
        landmarks = make_landmarks(33, visibility=0.65)
        assert landmarks_visible_enough(landmarks, INDICES, min_visibility=0.65) is True


class TestLandmarksInBounds:
    def test_all_within_frame_returns_true(self):
        landmarks = make_landmarks(33)
        for idx in INDICES:
            landmarks[idx].x, landmarks[idx].y = 0.5, 0.5
        assert landmarks_in_bounds(landmarks, INDICES, margin=0.05) is True

    def test_far_outside_frame_returns_false(self):
        landmarks = make_landmarks(33)
        landmarks[25].x, landmarks[25].y = 1.5, 0.5
        assert landmarks_in_bounds(landmarks, INDICES, margin=0.05) is False

    def test_just_within_margin_is_still_in_bounds(self):
        landmarks = make_landmarks(33)
        for idx in INDICES:
            landmarks[idx].x, landmarks[idx].y = 0.5, 0.5
        landmarks[27].y = 1.04  # within the 0.05 margin past 1.0
        assert landmarks_in_bounds(landmarks, INDICES, margin=0.05) is True

    def test_just_past_margin_is_out_of_bounds(self):
        landmarks = make_landmarks(33)
        for idx in INDICES:
            landmarks[idx].x, landmarks[idx].y = 0.5, 0.5
        landmarks[27].y = 1.06  # just past the 0.05 margin
        assert landmarks_in_bounds(landmarks, INDICES, margin=0.05) is False


class TestAverageVisibility:
    def test_computes_mean(self):
        landmarks = make_landmarks(33, visibility=1.0)
        landmarks[23].visibility = 0.6
        landmarks[25].visibility = 0.8
        landmarks[27].visibility = 1.0

        assert average_visibility(landmarks, INDICES) == pytest.approx(0.8)

    def test_empty_indices_returns_zero(self):
        landmarks = make_landmarks(33)
        assert average_visibility(landmarks, []) == 0.0
