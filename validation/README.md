# Rep-counting accuracy validation

This is the harness behind a specific, honest claim: *how much did
switching from a single-threshold rep counter (the original script) to
a debounced 4-state machine (rep_tracker.py) actually improve accuracy?*

`validate_reps.py` runs both algorithms over the same labeled clips,
using the same calibrated thresholds for both, so any difference in
accuracy is attributable only to the counting algorithm.

## How to record the clips

Use your phone or webcam, same framing as normal use (full body, hips
through ankles visible). Each clip should be its own file, ~15-30
seconds. Place them in `validation/clips/`. You don't need all of
these to get a result, but more clips (and more that expose jitter)
make the comparison more convincing:

| Filename | What to do | Why it's a useful test |
|---|---|---|
| `calibration.mp4` | 3 clean, deep, unhurried reps | Used once to derive `deep_threshold`/`shallow_threshold` for every other clip -- keeps the comparison controlled |
| `clean_5reps.mp4` | 5 clean reps, normal tempo | Sanity check -- both algorithms should get this right |
| `jittery_5reps.mp4` | 5 real reps, but let your knee angle wobble/hesitate near the bottom of each one (don't fake it -- just don't be extra smooth) | The single-threshold approach can double-count a rep if the angle oscillates back and forth across the threshold; the debounced FSM shouldn't |
| `pause_at_bottom.mp4` | 3 reps, hold the bottom position for 2-3 seconds each | Tests whether a long hold gets miscounted as multiple reps |
| `fast_reps.mp4` | 5 reps done quickly, minimal pause anywhere | Tests robustness (and the FSM's minimum-rep-duration guard) at speed |
| `no_rep_shallow_dip.mp4` | A few shallow dips that never reach full squat depth (e.g. adjusting your stance) | Neither algorithm should count these -- checks for false positives |

Feel free to add more clips beyond this list (e.g. a clip with genuinely
poor lighting, or turning away from the camera briefly) -- just add a
row to `manifest.csv` for each one.

## Filling in `manifest.csv`

Each row is one clip:

```csv
filename,true_reps,is_calibration_source,notes
clean_5reps.mp4,5,false,5 clean reps at a normal tempo
```

- `filename` -- must match a file in `validation/clips/`
- `true_reps` -- the actual number of full reps you performed, counted
  by eye. This is the ground truth everything is graded against.
- `is_calibration_source` -- `true` for exactly one row (the clip whose
  thresholds get reused everywhere else), `false` for the rest.
- `notes` -- freeform, shows up in the report.

A starter `manifest.csv` with this exact shot list is already in this
folder -- just record the clips and adjust `true_reps` if your actual
rep counts differ from the plan.

## Running it

```bash
source ../.venv/bin/activate   # or however you've set up the project's venv
python validate_reps.py
```

Prints a table (true count vs. naive vs. FSM, with pass/fail per clip)
and an overall accuracy percentage for each algorithm.

---

# Risk-flag accuracy validation

A second, separate harness (`validate_risk.py`) for the risk engine's
4 individual flags: asymmetry, instability, depth drop, speed drop.
"Risk" has no single ground truth like a rep count does, so this grades
each flag on its own against clips where you deliberately engineered
whether that flag *should* fire -- not a fuzzy "was this risky" judgment.

## Clips to record (in addition to the ones above)

Same framing/length as the rep-counting clips. Add these to
`validation/clips/`:

| Filename | What to do | Tests |
|---|---|---|
| `asymmetry_bad.mp4` | 4-5 reps where you deliberately favor one leg -- shift weight, bend one knee noticeably more than the other | Does the asymmetry flag fire when it should? (`clean_5reps.mp4` is the "should stay quiet" control, already recorded) |
| `instability_bad.mp4` | 4-5 reps where you deliberately wobble/sway while standing or holding the bottom | Does the instability flag fire when it should? |
| `fatigue_session.mp4` | One longer clip: 3-4 clean, deep, normal-tempo reps first (baseline), then several reps you deliberately make shallower and/or slower | Do depth-drop/speed-drop stay quiet during the baseline reps, then correctly turn on once you degrade? |

`clean_5reps.mp4` (already recorded for the rep-counting harness) doubles
as the negative control here too -- no new recording needed for that one.

## Filling in `risk_manifest.csv`

Each row grades one flag against one clip, optionally restricted to a
rep-index range:

```csv
filename,flag,expected,from_rep,to_rep,notes
asymmetry_bad.mp4,asymmetry,yes,1,,deliberately favor one leg
```

- `flag` -- one of `asymmetry`, `instability`, `depth_drop`, `speed_drop`
- `expected` -- `yes` if that flag should be active on the reps in range, `no` if it shouldn't
- `from_rep` / `to_rep` -- 1-based rep index range this row applies to (`to_rep` blank = unbounded). Used for `fatigue_session.mp4` to separate the baseline reps (1-3, flag can't fire yet) from the degraded reps (4+, flag should fire)
- A starter `risk_manifest.csv` matching this shot list is already in this folder

## Running it

```bash
source ../.venv/bin/activate
python validate_risk.py
```

Prints a table (per clip, per flag: how many reps in the expected range
matched what should have happened) and an overall accuracy percentage
across every rep-flag check.
