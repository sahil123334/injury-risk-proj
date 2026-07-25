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
