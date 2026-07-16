"""
injury_risk_mvp.py

Thin backward-compatible entry point. All logic now lives in main.py
and the supporting modules (config.py, pose_detector.py, biomechanics.py,
calibration.py, rep_tracker.py, risk_engine.py, ui.py, data_logger.py).

`python injury_risk_mvp.py` still works; prefer `python main.py` going
forward. The pre-refactor single-file version is preserved as
injury_risk_mvp_backup.py for reference.
"""

import sys

from main import main

if __name__ == "__main__":
    sys.exit(main())
