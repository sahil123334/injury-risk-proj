# KinetIQ

![Tests](https://github.com/sahil123334/injury-risk-proj/actions/workflows/tests.yml/badge.svg)

Movement-quality / fatigue monitor (prototype).

A webcam-based computer-vision prototype that tracks squat reps via
lower-body pose estimation and surfaces early **movement-quality** and
**fatigue-proxy** indicators.

> **Not a medical device.** This tool does not diagnose or predict
> injury. It flags patterns (asymmetry, angle instability, shallower
> or slower reps over time) that may be worth a coach's or clinician's
> attention. Treat any RED/YELLOW indicator as "review recommended,"
> never as a diagnosis.

## How it works

0. On launch, a small picker window lets you choose **record live** or
   **analyze an uploaded video file**. Everything downstream behaves
   identically either way -- calibration, rep tracking, and risk scoring
   only care about frames + elapsed time, not where they came from.
1. Reads frames continuously (feed always stays visible, even when no
   reliable pose is found).
2. Runs MediaPipe Pose Landmarker on each frame.
3. Computes left/right knee angles from hip-knee-ankle landmarks.
4. Spends the first ~10 seconds of *valid pose time* calibrating to
   your standing and deep-squat angles (do 2-3 clean reps here).
5. Tracks reps with a standing -> descending -> bottom -> ascending
   state machine, with debounce so landmark jitter can't fake a rep.
6. Once a 3-rep baseline is established, flags asymmetry, angle
   instability, shallower reps, and slower reps as they emerge.
7. Logs a per-frame trace and a per-rep summary to CSV under `data/`.
8. Click the **End Session** button in the top-right of the window (or
   press `q`) to stop -- or just let an uploaded video play to its end.
9. A native "Session Complete" window pops up with the headline numbers
   (duration, reps, final status). Click **View full report** from there
   to open the full HTML report (charts + CSV downloads) in your browser.

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
├── launcher.py               # startup picker window: record vs. upload
├── video_source.py           # live camera / video file, same interface either way
├── session_summary.py        # native "session complete" popup shown after each run
├── report_generator.py      # builds the post-session HTML report from the CSVs
├── report_template.html     # static shell/JS the report is built from
├── pose_landmarker.task     # MediaPipe model file
├── requirements.txt
├── data/                    # session_metrics.csv, rep_summary.csv, session_report.html land here
└── injury_risk_mvp_backup.py  # pre-refactor single-file version, kept for reference
```

## macOS setup

**Use a Homebrew Python, not the system/Command Line Tools one.** macOS's
built-in `python3` links against Apple's bundled Tcl/Tk 8.5, which has been
deprecated since 2009 and has a well-known bug where native windows
(the launcher, the session-complete popup) intermittently render blank
on modern macOS even though they're fully present and clickable. A
Homebrew Python links against a current Tk 8.6 instead, which doesn't
have this problem.

```bash
# 1. One-time: install a Homebrew Python with a modern Tk
brew install python@3.11 python-tk@3.11

cd injury-risk-proj

# 2. Create and activate a virtual environment from that Python
/opt/homebrew/bin/python3.11 -m venv .venv
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run it
python main.py
```

Every subsequent run just needs `source .venv/bin/activate` (or invoke
`.venv/bin/python main.py` directly) -- no need to repeat the `brew
install` or venv-creation steps.

A picker window opens: choose **Record and get live feedback** (pick a
camera if you have more than one) or **Upload a video file...** (opens a
native file dialog). On first run with a live camera, macOS will prompt
for **Camera** permission for your terminal / IDE. If you don't see the
prompt, grant it manually under **System Settings -> Privacy & Security
-> Camera**.

### Useful flags

```bash
python main.py --debug              # print calibration/rep state transitions to the console
python main.py --no-per-frame-log   # skip the per-frame CSV, keep only the per-rep summary
python main.py --no-report          # skip generating the HTML report and the summary popup entirely
python main.py --camera-index N     # record from this camera directly, skip the picker
python main.py --video path/to.mp4  # analyze this video file directly, skip the picker
python main.py --list-cameras       # print detected cameras and exit
```

Click **End Session** (top-right of the window) or press `q` to stop at
any time -- or let an uploaded video play to its end.

## Output

- `data/session_metrics.csv` — one row per analyzed frame (calibration + tracking phases).
- `data/rep_summary.csv` — one row per completed rep: depth, total/eccentric/concentric duration, speed, and the movement-quality flags active at that moment.
- `data/session_report.html` — charts for knee angle, asymmetry, and per-rep depth/duration, plus download buttons for both CSVs above. Regenerated (overwritten) every run; opened via the **View full report** button in the session-complete popup, not automatically.

## Testing

The core algorithm modules (`biomechanics.py`, `calibration.py`,
`rep_tracker.py`, `risk_engine.py`, `naive_rep_counter.py`) are pure
logic with no camera/GUI dependency, so they're covered by a real unit
test suite -- 63 tests, 97% statement coverage across those 5 modules.
UI/camera/file-I/O code (`main.py`, `ui.py`, `video_source.py`, etc.) is
covered by the manual/scripted verification described throughout this
README instead, since it needs a real display or camera to exercise.

```bash
source .venv/bin/activate
pip install pytest pytest-cov
python -m pytest tests/ --cov=. --cov-report=term-missing
```

Runs automatically on every push via GitHub Actions (`.github/workflows/tests.yml`).

## Known limitations (read before trusting the output)

- Single-camera 2D pose estimation: no depth, so angles are sensitive to camera angle and body rotation relative to the lens.
- Thresholds (`config.py`) were chosen heuristically, not validated against injury outcomes or a labeled dataset.
- Calibration assumes 2-3 genuinely clean, full-depth reps during the window; a bad calibration produces bad downstream thresholds.
- The fatigue proxy (depth/speed drop vs. an early-session baseline) reflects *this session only* — it has no cross-session or population baseline.
- Designed for a single visible athlete performing bodyweight squats; not tested for other exercises, multiple people in frame, or loaded/barbell variations.
- This is a research/engineering prototype, not a validated clinical or medical tool.
