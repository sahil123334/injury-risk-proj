# Movement-Quality / Fatigue Monitor (Prototype)

A webcam-based computer-vision prototype that tracks squat reps via
lower-body pose estimation and surfaces early **movement-quality** and
**fatigue-proxy** indicators.

> **Not a medical device.** This tool does not diagnose or predict
> injury. It flags patterns (asymmetry, angle instability, shallower
> or slower reps over time) that may be worth a coach's or clinician's
> attention. Treat any RED/YELLOW indicator as "review recommended,"
> never as a diagnosis.

## How it works

1. Opens the webcam and reads frames continuously (camera feed always
   stays visible, even when no reliable pose is found).
2. Runs MediaPipe Pose Landmarker on each frame.
3. Computes left/right knee angles from hip-knee-ankle landmarks.
4. Spends the first ~10 seconds of *valid pose time* calibrating to
   your standing and deep-squat angles (do 2-3 clean reps here).
5. Tracks reps with a standing -> descending -> bottom -> ascending
   state machine, with debounce so landmark jitter can't fake a rep.
6. Once a 3-rep baseline is established, flags asymmetry, angle
   instability, shallower reps, and slower reps as they emerge.
7. Logs a per-frame trace and a per-rep summary to CSV under `data/`.

## Project layout

```
injury-risk-proj/
├── main.py                 # camera loop, orchestrates everything
├── injury_risk_mvp.py       # thin backward-compatible entry point -> main.main()
├── config.py                # all constants/thresholds
├── pose_detector.py         # MediaPipe wrapper + visibility/bounds validation
├── biomechanics.py          # angle math + smoothing
├── calibration.py           # valid-pose-time calibration, robust thresholds
├── rep_tracker.py           # rep state machine (standing/descending/bottom/ascending)
├── risk_engine.py           # movement-quality scoring (non-medical language)
├── ui.py                    # all on-screen overlays
├── data_logger.py           # CSV writers
├── pose_landmarker.task     # MediaPipe model file
├── requirements.txt
├── data/                    # session_metrics.csv + rep_summary.csv land here
└── injury_risk_mvp_backup.py  # pre-refactor single-file version, kept for reference
```

## macOS setup

```bash
cd injury-risk-proj

# 1. Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run it
python main.py
```

On first run, macOS will prompt for **Camera** permission for your
terminal / IDE. If you don't see the prompt, grant it manually under
**System Settings -> Privacy & Security -> Camera**.

### Useful flags

```bash
python main.py --debug              # print calibration/rep state transitions to the console
python main.py --no-per-frame-log   # skip the per-frame CSV, keep only the per-rep summary
```

Press `q` with the camera window focused to quit at any time.

## Output

- `data/session_metrics.csv` — one row per analyzed frame (calibration + tracking phases).
- `data/rep_summary.csv` — one row per completed rep: depth, total/eccentric/concentric duration, speed, and the movement-quality flags active at that moment.

## Known limitations (read before trusting the output)

- Single-camera 2D pose estimation: no depth, so angles are sensitive to camera angle and body rotation relative to the lens.
- Thresholds (`config.py`) were chosen heuristically, not validated against injury outcomes or a labeled dataset.
- Calibration assumes 2-3 genuinely clean, full-depth reps during the window; a bad calibration produces bad downstream thresholds.
- The fatigue proxy (depth/speed drop vs. an early-session baseline) reflects *this session only* — it has no cross-session or population baseline.
- Designed for a single visible athlete performing bodyweight squats; not tested for other exercises, multiple people in frame, or loaded/barbell variations.
- This is a research/engineering prototype, not a validated clinical or medical tool.
