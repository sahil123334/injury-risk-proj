"""
data_logger.py

Owns both CSV files the program writes:
  - a per-frame trace (optional, on by default) for later analysis/debugging
  - a per-rep summary (always on) which is the more useful artifact for
    reviewing a session at a glance

Both files live under data/. `close()` is safe to call multiple times
and safe to call even if a file was never opened.
"""

import csv
import os
from typing import Optional, TextIO

import config

PER_FRAME_HEADER = [
    "time_sec",
    "phase",
    "rep_count",
    "left_knee_angle",
    "right_knee_angle",
    "mean_knee_angle",
    "asymmetry",
    "variance",
    "deep_threshold",
    "shallow_threshold",
    "depth_drop",
    "speed_drop_pct",
    "risk_score",
    "risk_label",
]

PER_REP_HEADER = [
    "rep_index",
    "time_sec",
    "depth_angle",
    "duration_sec",
    "eccentric_duration_sec",
    "concentric_duration_sec",
    "speed",
    "depth_drop",
    "speed_drop_pct",
    "risk_score",
    "risk_label",
]


class DataLogger:
    def __init__(
        self,
        data_dir: str = config.DATA_DIR,
        per_frame_path: str = config.PER_FRAME_CSV,
        per_rep_path: str = config.PER_REP_CSV,
        log_per_frame: bool = config.LOG_PER_FRAME,
    ):
        os.makedirs(data_dir, exist_ok=True)

        self._log_per_frame = log_per_frame
        self._per_frame_file: Optional[TextIO] = None
        self._per_frame_writer = None
        self._per_rep_file: Optional[TextIO] = None
        self._per_rep_writer = None

        if self._log_per_frame:
            self._per_frame_file = open(per_frame_path, "w", newline="")
            self._per_frame_writer = csv.writer(self._per_frame_file)
            self._per_frame_writer.writerow(PER_FRAME_HEADER)

        self._per_rep_file = open(per_rep_path, "w", newline="")
        self._per_rep_writer = csv.writer(self._per_rep_file)
        self._per_rep_writer.writerow(PER_REP_HEADER)

    def __enter__(self) -> "DataLogger":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    def log_frame(
        self,
        *,
        time_sec: float,
        phase: str,
        rep_count: int,
        left_knee_angle: float,
        right_knee_angle: float,
        mean_knee_angle: float,
        asymmetry: float,
        variance: float,
        deep_threshold: Optional[float],
        shallow_threshold: Optional[float],
        depth_drop: Optional[float],
        speed_drop_pct: Optional[float],
        risk_score: int,
        risk_label: str,
    ) -> None:
        if not self._log_per_frame or self._per_frame_writer is None:
            return

        self._per_frame_writer.writerow([
            round(time_sec, 3),
            phase,
            rep_count,
            round(left_knee_angle, 3),
            round(right_knee_angle, 3),
            round(mean_knee_angle, 3),
            round(asymmetry, 3),
            round(variance, 3),
            "" if deep_threshold is None else round(deep_threshold, 3),
            "" if shallow_threshold is None else round(shallow_threshold, 3),
            "" if depth_drop is None else round(depth_drop, 3),
            "" if speed_drop_pct is None else round(speed_drop_pct, 3),
            risk_score,
            risk_label,
        ])

    def log_rep(
        self,
        *,
        rep_index: int,
        time_sec: float,
        depth_angle: float,
        duration_sec: float,
        eccentric_duration_sec: float,
        concentric_duration_sec: float,
        speed: float,
        depth_drop: float,
        speed_drop_pct: float,
        risk_score: int,
        risk_label: str,
    ) -> None:
        if self._per_rep_writer is None:
            return

        self._per_rep_writer.writerow([
            rep_index,
            round(time_sec, 3),
            round(depth_angle, 3),
            round(duration_sec, 3),
            round(eccentric_duration_sec, 3),
            round(concentric_duration_sec, 3),
            round(speed, 4),
            round(depth_drop, 3),
            round(speed_drop_pct, 3),
            risk_score,
            risk_label,
        ])

    def close(self) -> None:
        if self._per_frame_file is not None:
            self._per_frame_file.close()
            self._per_frame_file = None
        if self._per_rep_file is not None:
            self._per_rep_file.close()
            self._per_rep_file = None
