"""
report_generator.py

Builds one self-contained HTML report from the CSVs a session just wrote
(data/session_metrics.csv, data/rep_summary.csv) so you get a readable,
chartable summary right after quitting instead of opening a CSV by hand.

The report's "download" buttons are plain relative links to the CSV
files sitting next to it in data/ -- nothing is re-embedded or
duplicated, so the report and the CSVs can never drift out of sync.
"""

import csv
import os
import json
import webbrowser
from typing import List, Optional

import config


def _read_frames(path: str) -> List[dict]:
    if not os.path.isfile(path):
        return []
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))

    frames = []
    for r in rows:
        frames.append({
            "t": float(r["time_sec"]),
            "phase": r["phase"],
            "rep": int(r["rep_count"]),
            "mean": float(r["mean_knee_angle"]),
            "asym": float(r["asymmetry"]),
            "var": float(r["variance"]),
            "deep": float(r["deep_threshold"]) if r["deep_threshold"] else None,
            "shallow": float(r["shallow_threshold"]) if r["shallow_threshold"] else None,
            "score": int(r["risk_score"]),
            "label": r["risk_label"],
        })
    return frames


def _read_reps(path: str) -> List[dict]:
    if not os.path.isfile(path):
        return []
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))

    reps = []
    for r in rows:
        reps.append({
            "idx": int(r["rep_index"]),
            "t": float(r["time_sec"]),
            "depth": float(r["depth_angle"]),
            "dur": float(r["duration_sec"]),
            "ecc": float(r["eccentric_duration_sec"]),
            "con": float(r["concentric_duration_sec"]),
            "speed": float(r["speed"]),
            "depth_drop": float(r["depth_drop"]),
            "speed_drop_pct": float(r["speed_drop_pct"]),
            "score": int(r["risk_score"]),
            "label": r["risk_label"],
        })
    return reps


def generate_report(
    per_frame_csv: str = config.PER_FRAME_CSV,
    per_rep_csv: str = config.PER_REP_CSV,
    output_path: str = config.REPORT_OUTPUT_PATH,
) -> Optional[str]:
    """
    Build the HTML report from whatever CSVs exist on disk. Returns the
    output path, or None if there's nothing worth reporting (e.g. the
    session ended before any frames or reps were logged).
    """
    frames = _read_frames(per_frame_csv)
    reps = _read_reps(per_rep_csv)

    if not frames and not reps:
        return None

    with open(config.REPORT_TEMPLATE_PATH) as f:
        template = f.read()

    payload = {
        "frames": frames,
        "reps": reps,
        "per_frame_csv_name": os.path.basename(per_frame_csv) if frames else None,
        "per_rep_csv_name": os.path.basename(per_rep_csv) if reps else None,
    }

    html = template.replace("__SESSION_DATA_JSON__", json.dumps(payload))

    with open(output_path, "w") as f:
        f.write(html)

    return output_path


def open_report(path: str) -> None:
    webbrowser.open("file://" + os.path.abspath(path))
