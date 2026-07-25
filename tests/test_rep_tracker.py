import pytest

from rep_tracker import RepState, RepTracker

DEEP = 120.0
SHALLOW = 150.0


def feed(tracker, angle_time_pairs):
    """Feeds (angle, now) pairs in order; returns the list of non-None RepResults."""
    results = []
    for angle, now in angle_time_pairs:
        result = tracker.update(angle, now)
        if result is not None:
            results.append(result)
    return results


# One full, clean standing->descending->bottom->ascending->standing cycle,
# using dt=0.05s so timing math stays comfortably clear of the default
# 0.3s min-rep-duration boundary (avoids a float-precision flaky test).
CLEAN_CYCLE = [
    (145, 0.05), (140, 0.10), (135, 0.15),   # STANDING -> DESCENDING (commits @0.15)
    (115, 0.20), (110, 0.25), (105, 0.30),   # DESCENDING -> BOTTOM (commits @0.30), depth=105
    (125, 0.35), (130, 0.40), (135, 0.45),   # BOTTOM -> ASCENDING (commits @0.45)
    (155, 0.50), (160, 0.55), (165, 0.60),   # ASCENDING -> STANDING (commits @0.60, rep completes)
]


class TestRepTrackerCleanCycle:
    def test_completes_exactly_one_rep(self):
        tracker = RepTracker(DEEP, SHALLOW)
        results = feed(tracker, CLEAN_CYCLE)

        assert len(results) == 1
        assert tracker.rep_count == 1
        assert tracker.state == RepState.STANDING

    def test_rep_result_values(self):
        tracker = RepTracker(DEEP, SHALLOW)
        [result] = feed(tracker, CLEAN_CYCLE)

        assert result.index == 1
        assert result.depth_angle == pytest.approx(105.0)
        assert result.duration == pytest.approx(0.45, abs=1e-6)
        assert result.eccentric_duration == pytest.approx(0.15, abs=1e-6)
        assert result.concentric_duration == pytest.approx(0.15, abs=1e-6)
        assert result.speed == pytest.approx(1 / 0.45)

    def test_eccentric_and_concentric_dont_double_count_the_bottom_hold(self):
        tracker = RepTracker(DEEP, SHALLOW)
        [result] = feed(tracker, CLEAN_CYCLE)

        # There's a 0.15s hold between reaching the bottom and starting to
        # ascend -- correctly excluded from both eccentric and concentric.
        assert result.eccentric_duration + result.concentric_duration < result.duration

    def test_two_consecutive_reps_increment_index(self):
        tracker = RepTracker(DEEP, SHALLOW)
        feed(tracker, CLEAN_CYCLE)

        second_cycle = [(angle, t + 0.60) for angle, t in CLEAN_CYCLE]
        [second_result] = feed(tracker, second_cycle)

        assert second_result.index == 2
        assert tracker.rep_count == 2


class TestRepTrackerDebounce:
    def test_transition_does_not_commit_before_min_state_frames(self):
        tracker = RepTracker(DEEP, SHALLOW, min_state_frames=3)
        # Only 2 frames below shallow_threshold, then back above -- should
        # never commit to DESCENDING at all.
        results = feed(tracker, [(145, 0.05), (140, 0.10), (170, 0.15)])

        assert results == []
        assert tracker.state == RepState.STANDING
        assert tracker.rep_count == 0

    def test_bailing_out_before_reaching_bottom_counts_no_rep(self):
        tracker = RepTracker(DEEP, SHALLOW, min_state_frames=3)
        # Commit into DESCENDING, then bail straight back to STANDING
        # without ever reaching the deep threshold.
        sequence = [
            (135, 0.05), (130, 0.10), (125, 0.15),  # -> DESCENDING
            (155, 0.20), (160, 0.25), (165, 0.30),  # bails back -> STANDING
        ]
        results = feed(tracker, sequence)

        assert results == []
        assert tracker.state == RepState.STANDING
        assert tracker.rep_count == 0


class TestRepTrackerMinDuration:
    def test_too_fast_a_rep_is_discarded(self):
        # Same clean cycle, but require a much longer minimum duration
        # than the 0.45s this cycle actually takes.
        tracker = RepTracker(DEEP, SHALLOW, min_rep_duration=10.0)
        results = feed(tracker, CLEAN_CYCLE)

        assert results == []
        assert tracker.rep_count == 0
        # State machine itself still completed the cycle; only the rep
        # counting was rejected.
        assert tracker.state == RepState.STANDING


class TestRepTrackerReDip:
    def test_dipping_back_to_bottom_during_ascent_does_not_crash_or_miscount(self):
        tracker = RepTracker(DEEP, SHALLOW, min_state_frames=3)
        sequence = [
            (135, 0.05), (130, 0.10), (125, 0.15),   # -> DESCENDING
            (115, 0.20), (110, 0.25), (105, 0.30),   # -> BOTTOM
            (125, 0.35), (130, 0.40), (135, 0.45),   # -> ASCENDING
            (118, 0.50), (115, 0.55), (112, 0.60),   # dips back -> BOTTOM again
            (125, 0.65), (130, 0.70), (135, 0.75),   # -> ASCENDING again
            (155, 0.80), (160, 0.85), (165, 0.90),   # -> STANDING, rep completes
        ]
        results = feed(tracker, sequence)

        assert len(results) == 1
        assert tracker.rep_count == 1
        assert results[0].depth_angle == pytest.approx(105.0)
