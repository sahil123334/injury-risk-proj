"""
main.py

Owns the frame loop and wires every module together. This is the only
file that knows about all the other modules; none of them import each
other in a cycle (config <- biomechanics/pose_detector <- calibration/
rep_tracker <- risk_engine, with ui/data_logger depending only on
config and plain values).

Frames can come from a live camera or an uploaded video file -- see
video_source.py. Whichever one it is, the loop below only ever consumes
`ret, frame, elapsed` from it, so calibration/rep tracking/risk scoring
run identically either way.

Run it directly:     python main.py                  (shows a picker: record or upload)
Record a camera:      python main.py --camera-index N
Analyze a video file: python main.py --video path/to/clip.mp4
List cameras:         python main.py --list-cameras
"""

import argparse
import sys
from typing import Optional

import cv2

import config
from biomechanics import AngleSmoother, compute_knee_angles
from calibration import Calibrator
from camera_utils import discover_cameras, print_camera_list
from data_logger import DataLogger
from launcher import show_launcher
from pose_detector import PoseDetector
from rep_tracker import RepState, RepTracker
from report_generator import generate_report, open_report
from risk_engine import RiskEngine
from session_summary import show_session_summary
from video_source import FileVideoSource, LiveCameraSource
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
        help="Record from this camera device directly, skipping the launcher. See --list-cameras.",
    )
    parser.add_argument(
        "--video",
        type=str,
        default=None,
        help="Analyze this video file directly, skipping the launcher.",
    )
    parser.add_argument(
        "--list-cameras",
        action="store_true",
        help="Print detected camera devices and exit, without starting the app.",
    )
    parser.add_argument(
        "--no-report",
        action="store_true",
        help="Don't auto-generate/open the HTML session report after quitting.",
    )
    return parser.parse_args()


def resolve_source(args: argparse.Namespace):
    """
    Decide where frames come from. An explicit --video or --camera-index
    skips the launcher entirely (useful for scripting); otherwise show the
    picker window and let the user choose record vs. upload.

    Returns (source, error_message_or_None). Returns (None, None) if the
    user closed the launcher without choosing anything.
    """
    if args.video:
        try:
            return FileVideoSource(args.video), None
        except FileNotFoundError as exc:
            return None, str(exc)

    if args.camera_index is not None:
        return LiveCameraSource(args.camera_index), None

    choice = show_launcher()
    if choice is None:
        return None, None

    if choice.mode == "upload":
        try:
            return FileVideoSource(choice.video_path), None
        except FileNotFoundError as exc:
            return None, str(exc)

    return LiveCameraSource(choice.camera_index), None


def main() -> int:
    args = parse_args()
    debug = args.debug or config.DEBUG_MODE
    log_per_frame = config.LOG_PER_FRAME and not args.no_per_frame_log

    if args.list_cameras:
        print_camera_list(discover_cameras())
        return 0

    # Fail fast and clearly if the model file is missing, before showing any UI.
    try:
        detector = PoseDetector(config.MODEL_PATH)
    except FileNotFoundError as exc:
        print(f"Fatal: {exc}")
        return 1

    source, error = resolve_source(args)
    if error is not None:
        print(f"Fatal: {error}")
        detector.close()
        return 1
    if source is None:
        print("No input source chosen; exiting.")
        detector.close()
        return 0

    if not source.is_opened():
        which = "video file" if not source.is_live else "webcam"
        print(f"Fatal: could not open {which}.")
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

    last_valid_elapsed: Optional[float] = None
    last_timestamp_ms = -1
    consecutive_read_failures = 0
    # A file source ending is just "the video is over" -- exit on the first
    # failed read. A live camera can drop a handful of frames on startup
    # (e.g. Continuity Camera warm-up) and deserves some tolerance.
    max_read_failures = config.MAX_CONSECUTIVE_READ_FAILURES if source.is_live else 0

    window_name = f"{config.WINDOW_NAME} -- {source.describe()}"

    # The stop button is drawn fresh every frame (its position depends on
    # frame width), so the mouse callback reads its bounds from this list
    # rather than a fixed rect computed once.
    stop_button_rect = [0, 0, 0, 0]
    stop_requested = False

    def on_mouse(event, x, y, _flags, _param):
        nonlocal stop_requested
        if event == cv2.EVENT_LBUTTONDOWN:
            x1, y1, x2, y2 = stop_button_rect
            if x1 <= x <= x2 and y1 <= y <= y2:
                stop_requested = True

    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, on_mouse)

    session_elapsed = 0.0
    last_overall_label: Optional[str] = None

    try:
        with detector:
            while True:
                ret, frame, elapsed = source.read()
                if not ret:
                    consecutive_read_failures += 1
                    if consecutive_read_failures > max_read_failures:
                        print("No more frames available; exiting.")
                        break
                    continue
                consecutive_read_failures = 0
                session_elapsed = elapsed

                if source.is_live:
                    frame = cv2.flip(frame, 1)  # mirror for a selfie-style view
                height, width = frame.shape[:2]

                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                # MediaPipe's VIDEO mode requires strictly increasing timestamps;
                # guard against two frames rounding to the same millisecond.
                timestamp_ms = max(int(elapsed * 1000), last_timestamp_ms + 1)
                last_timestamp_ms = timestamp_ms
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
                        smoothed = smoother.update(
                            raw_angles.left, raw_angles.right,
                            raw_angles.left_confidence, raw_angles.right_confidence,
                        )
                        dt = 0.0 if last_valid_elapsed is None else max(elapsed - last_valid_elapsed, 0.0)
                        last_valid_elapsed = elapsed

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
                            rep_result = rep_tracker.update(smoothed.mean, elapsed)

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
                            last_overall_label = assessment.label

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
                # matter which branch above executed, so the feed never goes black
                # just because a pose wasn't found or wasn't reliable.
                stop_button_rect[:] = ui.draw_stop_button(frame, width, height)
                cv2.imshow(window_name, frame)
                key_pressed = cv2.waitKey(1) & 0xFF
                if key_pressed == ord("q") or stop_requested:
                    break

    finally:
        source.release()
        logger.close()
        cv2.destroyAllWindows()

    # Build the HTML report from the CSVs we just finished writing, then show
    # a native "session complete" window with the headline numbers -- opening
    # the full report is an explicit click from there, not an unannounced
    # browser tab. Only runs after a normal exit (finally above already ran);
    # a genuine crash skips this and the CSVs are still on disk as far as
    # they got.
    report_path = None if args.no_report else generate_report()
    rep_count = rep_tracker.rep_count if rep_tracker is not None else 0

    if report_path is None and args.no_report:
        pass  # report generation was explicitly disabled; nothing to show
    else:
        show_session_summary(
            duration_sec=session_elapsed,
            rep_count=rep_count,
            overall_label=last_overall_label,
            report_path=report_path,
            on_view_report=open_report,
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
