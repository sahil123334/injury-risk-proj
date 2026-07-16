"""
main.py

Owns the camera loop and wires every module together. This is the only
file that knows about all the other modules; none of them import each
other in a cycle (config <- biomechanics/pose_detector <- calibration/
rep_tracker <- risk_engine, with ui/data_logger depending only on
config and plain values).

Run it directly:  python main.py [--debug] [--no-per-frame-log] [--camera-index N]
List cameras:      python main.py --list-cameras
"""

import argparse
import platform
import sys
import time
from typing import Optional

import cv2

import config
from biomechanics import AngleSmoother, compute_knee_angles
from calibration import Calibrator
from camera_utils import choose_camera_index, discover_cameras, print_camera_list
from data_logger import DataLogger
from pose_detector import PoseDetector
from rep_tracker import RepState, RepTracker
from risk_engine import RiskEngine
import ui


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Movement-quality / fatigue-proxy monitor (prototype).")
    parser.add_argument("--debug", action="store_true", help="Print state transitions and calibration events.")
    parser.add_argument(
        "--no-per-frame-log",
        action="store_true",
        help="Skip the per-frame CSV; only write the per-rep summary CSV.",
    )
    parser.add_argument(
        "--camera-index",
        type=int,
        default=None,
        help="Use this camera device index directly (skips auto-detect/prompt). See --list-cameras.",
    )
    parser.add_argument(
        "--list-cameras",
        action="store_true",
        help="Print detected camera devices and exit, without starting the app.",
    )
    return parser.parse_args()


def open_camera(index: int) -> cv2.VideoCapture:
    """Prefer the AVFoundation backend on macOS; fall back to the default elsewhere."""
    use_avfoundation = config.CAMERA.use_avfoundation and platform.system() == "Darwin"
    backend = cv2.CAP_AVFOUNDATION if use_avfoundation else cv2.CAP_ANY
    cap = cv2.VideoCapture(index, backend)

    if config.CAMERA.requested_width:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.CAMERA.requested_width)
    if config.CAMERA.requested_height:
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.CAMERA.requested_height)

    return cap


def main() -> int:
    args = parse_args()
    debug = args.debug or config.DEBUG_MODE
    log_per_frame = config.LOG_PER_FRAME and not args.no_per_frame_log

    if args.list_cameras:
        print_camera_list(discover_cameras())
        return 0

    camera_index = choose_camera_index(args.camera_index)

    # Fail fast and clearly if the model file is missing, before touching the camera.
    try:
        detector = PoseDetector(config.MODEL_PATH)
    except FileNotFoundError as exc:
        print(f"Fatal: {exc}")
        return 1

    cap = open_camera(camera_index)
    if not cap.isOpened():
        print("Fatal: could not open webcam.")
        detector.close()
        return 1

    logger = DataLogger(log_per_frame=log_per_frame)
    smoother = AngleSmoother()
    calibrator = Calibrator(debug=debug)
    risk_engine = RiskEngine()

    rep_tracker: Optional[RepTracker] = None
    rep_depths = []
    rep_speeds = []
    baseline_depth: Optional[float] = None
    baseline_speed: Optional[float] = None

    start_time = time.time()
    last_valid_time: Optional[float] = None
    consecutive_read_failures = 0

    try:
        with detector:
            while True:
                ret, frame = cap.read()
                if not ret:
                    consecutive_read_failures += 1
                    if consecutive_read_failures > config.MAX_CONSECUTIVE_READ_FAILURES:
                        print("Camera stopped returning frames; exiting.")
                        break
                    time.sleep(0.01)
                    continue
                consecutive_read_failures = 0

                frame = cv2.flip(frame, 1)
                height, width = frame.shape[:2]

                now = time.time()
                elapsed = now - start_time

                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                timestamp_ms = int(elapsed * 1000)
                reading = detector.detect(rgb_frame, timestamp_ms)

                if reading.landmarks is None:
                    ui.draw_no_pose_overlay(frame)

                elif not reading.lower_body_visible:
                    ui.draw_pose_landmarks(frame, reading.landmarks, width, height)
                    ui.draw_low_confidence_overlay(frame, reading.confidence)

                else:
                    ui.draw_pose_landmarks(frame, reading.landmarks, width, height)
                    raw_angles = compute_knee_angles(reading.landmarks, width, height)

                    if raw_angles is None:
                        # Degenerate geometry (e.g. overlapping points) -- treat like
                        # low confidence rather than computing a garbage angle.
                        ui.draw_low_confidence_overlay(frame, reading.confidence)
                    else:
                        smoothed = smoother.update(raw_angles.left, raw_angles.right)
                        dt = 0.0 if last_valid_time is None else max(now - last_valid_time, 0.0)
                        last_valid_time = now

                        if not calibrator.is_calibrated:
                            reason_before = calibrator.last_reset_reason
                            calibrator.update(smoothed.mean, dt)

                            if calibrator.last_reset_reason != reason_before and not calibrator.is_calibrated:
                                ui.draw_calibration_reset_overlay(frame, calibrator.last_reset_reason)
                            else:
                                ui.draw_calibration_overlay(
                                    frame, calibrator.time_remaining, calibrator.progress, smoothed.mean
                                )

                            logger.log_frame(
                                time_sec=elapsed,
                                phase="calibration",
                                rep_count=0,
                                left_knee_angle=smoothed.left,
                                right_knee_angle=smoothed.right,
                                mean_knee_angle=smoothed.mean,
                                asymmetry=smoothed.asymmetry,
                                variance=smoothed.variance,
                                deep_threshold=None,
                                shallow_threshold=None,
                                depth_drop=None,
                                speed_drop_pct=None,
                                risk_score=0,
                                risk_label="GREEN",
                            )

                            if calibrator.is_calibrated:
                                result = calibrator.result
                                rep_tracker = RepTracker(
                                    result.deep_threshold, result.shallow_threshold, debug=debug
                                )

                        else:
                            # rep_tracker is guaranteed non-None here: it's created in
                            # the same frame calibration completes, above.
                            rep_result = rep_tracker.update(smoothed.mean, now)

                            if rep_result is not None:
                                rep_depths.append(rep_result.depth_angle)
                                rep_speeds.append(rep_result.speed)

                                if baseline_depth is None and len(rep_depths) == config.BASELINE_REPS:
                                    baseline_depth = sum(rep_depths[: config.BASELINE_REPS]) / config.BASELINE_REPS
                                if baseline_speed is None and len(rep_speeds) == config.BASELINE_REPS:
                                    baseline_speed = sum(rep_speeds[: config.BASELINE_REPS]) / config.BASELINE_REPS

                                if debug:
                                    print(
                                        f"[main] rep {rep_result.index} complete: "
                                        f"depth={rep_result.depth_angle:.1f} duration={rep_result.duration:.2f}s"
                                    )

                            assessment = risk_engine.evaluate(
                                pose_confident=True,
                                asymmetry=smoothed.asymmetry,
                                variance=smoothed.variance,
                                rep_state=rep_tracker.state,
                                rep_depths=rep_depths,
                                rep_speeds=rep_speeds,
                                baseline_depth=baseline_depth,
                                baseline_speed=baseline_speed,
                            )

                            ui.draw_status_box(frame, assessment.status_text, assessment.label, assessment.score)
                            ui.draw_tracking_overlay(
                                frame,
                                rep_count=rep_tracker.rep_count,
                                smooth_left=smoothed.left,
                                smooth_right=smoothed.right,
                                smooth_mean=smoothed.mean,
                                asymmetry=smoothed.asymmetry,
                                variance=smoothed.variance,
                                depth_drop=assessment.depth_drop,
                                speed_drop_pct=assessment.speed_drop_pct,
                                baseline_established=baseline_depth is not None and baseline_speed is not None,
                                baseline_reps=config.BASELINE_REPS,
                                reasons=assessment.reasons,
                            )

                            logger.log_frame(
                                time_sec=elapsed,
                                phase="tracking",
                                rep_count=rep_tracker.rep_count,
                                left_knee_angle=smoothed.left,
                                right_knee_angle=smoothed.right,
                                mean_knee_angle=smoothed.mean,
                                asymmetry=smoothed.asymmetry,
                                variance=smoothed.variance,
                                deep_threshold=rep_tracker.deep_threshold,
                                shallow_threshold=rep_tracker.shallow_threshold,
                                depth_drop=assessment.depth_drop,
                                speed_drop_pct=assessment.speed_drop_pct,
                                risk_score=assessment.score,
                                risk_label=assessment.label,
                            )

                            if rep_result is not None:
                                logger.log_rep(
                                    rep_index=rep_result.index,
                                    time_sec=elapsed,
                                    depth_angle=rep_result.depth_angle,
                                    duration_sec=rep_result.duration,
                                    eccentric_duration_sec=rep_result.eccentric_duration,
                                    concentric_duration_sec=rep_result.concentric_duration,
                                    speed=rep_result.speed,
                                    depth_drop=assessment.depth_drop,
                                    speed_drop_pct=assessment.speed_drop_pct,
                                    risk_score=assessment.score,
                                    risk_label=assessment.label,
                                )

                # Single, centralized display + exit check. This runs every frame no
                # matter which branch above executed, so the camera feed never goes
                # black just because a pose wasn't found or wasn't reliable.
                cv2.imshow(config.WINDOW_NAME, frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

    finally:
        cap.release()
        logger.close()
        cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    sys.exit(main())
