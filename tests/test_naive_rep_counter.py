from naive_rep_counter import NaiveRepCounter

DEEP = 120.0
SHALLOW = 150.0


class TestNaiveRepCounter:
    def test_one_clean_rep_counted(self):
        counter = NaiveRepCounter(DEEP, SHALLOW)
        angles = [170, 160, 140, 110, 115, 140, 160, 175]

        for angle in angles:
            counter.update(angle)

        assert counter.rep_count == 1

    def test_never_reaching_deep_threshold_counts_nothing(self):
        counter = NaiveRepCounter(DEEP, SHALLOW)
        angles = [170, 160, 155, 160, 170]  # shallow dip, never crosses 120

        for angle in angles:
            counter.update(angle)

        assert counter.rep_count == 0

    def test_multiple_reps_increment_correctly(self):
        counter = NaiveRepCounter(DEEP, SHALLOW)
        one_rep = [170, 140, 110, 140, 175]

        for _ in range(3):
            for angle in one_rep:
                counter.update(angle)

        assert counter.rep_count == 3

    def test_characterization_mid_rep_jitter_across_shallow_threshold_overcounts(self):
        """
        Documents a known limitation of the naive approach (the reason
        rep_tracker.py's debounced FSM replaced it): if the angle briefly
        crosses back above shallow_threshold mid-rep before finishing,
        the naive counter splits what a human would call one rep into
        two, because it has no concept of "still in the same rep."
        """
        counter = NaiveRepCounter(DEEP, SHALLOW)
        angles = [
            170, 140, 110,       # descend into a rep
            151, 108,            # brief flicker above shallow_threshold, then back down
            140, 175,            # finish standing
        ]

        for angle in angles:
            counter.update(angle)

        assert counter.rep_count == 2  # a human would count this as 1
