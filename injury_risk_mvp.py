import cv2
import csv
import time
import numpy as np
import mediapipe as mp
from collections import deque

# -----------------------------
# CONFIG
# -----------------------------
MODEL_PATH = "pose_landmarker.task"
OUTPUT_CSV = "session_metrics.csv"

CALIBRATION_SECONDS = 10
ANGLE_WINDOW = 5
MIN_VISIBILITY = 0.65
ASYMMETRY_THRESHOLD = 12
ANGLE_VARIANCE_THRESHOLD = 18
FATIGUE_DEPTH_DROP_THRESHOLD = 12
FATIGUE_SPEED_DROP_THRESHOLD = 0.25

BASELINE_REPS = 3

HIGH_RISK_SCORE = 3
MED_RISK_SCORE = 1

# -----------------------------
# MEDIAPIPE SETUP
# -----------------------------
BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=VisionRunningMode.VIDEO,
)

LEFT_HIP, LEFT_KNEE, LEFT_ANKLE = 23, 25, 27
RIGHT_HIP, RIGHT_KNEE, RIGHT_ANKLE = 24, 26, 28
LOWER_BODY_LANDMARKS = [
    LEFT_HIP,
    LEFT_KNEE,
    LEFT_ANKLE,
    RIGHT_HIP,
    RIGHT_KNEE,
    RIGHT_ANKLE
]

# -----------------------------
# HELPER FUNCTIONS
# -----------------------------
def angle_between_three_points(a, b, c):
    a = np.array(a, dtype=np.float32)
    b = np.array(b, dtype=np.float32)
    c = np.array(c, dtype=np.float32)

    ba = a - b
    bc = c - b

    norm_ba = np.linalg.norm(ba)
    norm_bc = np.linalg.norm(bc)

    if norm_ba == 0 or norm_bc == 0:
        return None

    cosine = np.dot(ba, bc) / (norm_ba * norm_bc)
    cosine = np.clip(cosine, -1.0, 1.0)

    return float(np.degrees(np.arccos(cosine)))


def get_point(landmarks, index, width, height):
    lm = landmarks[index]
    return (lm.x * width, lm.y * height)
def landmarks_visible_enough(landmarks, indices, min_visibility=MIN_VISIBILITY):
    for idx in indices:
        lm = landmarks[idx]

        # visibility = how likely the landmark is actually visible
        # if it is too low, MediaPipe may just be guessing
        if lm.visibility < min_visibility:
            return False

    return True

def draw_text(frame, text, x, y, size=0.65):
    cv2.putText(
        frame,
        text,
        (x, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        size,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )


def draw_risk_box(frame, label, score):
    if label == "GREEN":
        color = (0, 180, 0)
    elif label == "YELLOW":
        color = (0, 215, 255)
    else:
        color = (0, 0, 255)

    cv2.rectangle(frame, (20, 20), (285, 95), color, -1)
    draw_text(frame, f"RISK: {label}", 35, 55, 0.9)
    draw_text(frame, f"Score: {score}", 35, 82, 0.65)


# -----------------------------
# STATE VARIABLES
# -----------------------------
left_hist = deque(maxlen=ANGLE_WINDOW)
right_hist = deque(maxlen=ANGLE_WINDOW)
mean_hist = deque(maxlen=ANGLE_WINDOW)

calibration_angles = []
is_calibrated = False

deep_knee_bend_threshold = None
shallow_knee_bend_threshold = None

rep_count = 0
in_rep = False
rep_start_time = None
current_rep_min_angle = None

rep_depths = []
rep_speeds = []

baseline_depth = None
baseline_speed = None

# -----------------------------
# CSV SETUP
# -----------------------------
csv_file = open(OUTPUT_CSV, "w", newline="")
writer = csv.writer(csv_file)

writer.writerow([
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
    "risk_label"
])

# -----------------------------
# CAMERA SETUP
# -----------------------------
cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)

if not cap.isOpened():
    print("Could not open webcam.")
    csv_file.close()
    raise SystemExit

start_time = time.time()

with PoseLandmarker.create_from_options(options) as landmarker:
    while True:
        ret, frame = cap.read()

        if not ret:
            break

        frame = cv2.flip(frame, 1)
        height, width, _ = frame.shape

        elapsed = time.time() - start_time

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        timestamp_ms = int(elapsed * 1000)
        result = landmarker.detect_for_video(mp_image, timestamp_ms)

        risk_score = 0
        risk_label = "GREEN"
        risk_reasons = []

        if result.pose_landmarks:
            landmarks = result.pose_landmarks[0]
            if not landmarks_visible_enough(landmarks, LOWER_BODY_LANDMARKS):
                draw_text(frame, "Move back: hips, knees, ankles must be visible", 20, 130)
                draw_text(frame, "Risk paused: lower body landmarks unreliable", 20, 160)

                cv2.imshow("Injury Risk MVP", frame)

                if cv2.waitKey(1) & 0xFF == ord("q"):
                     break

                continue
            for lm in landmarks:
                x = int(lm.x * width)
                y = int(lm.y * height)
                cv2.circle(frame, (x, y), 3, (0, 255, 0), -1)

            left_hip = get_point(landmarks, LEFT_HIP, width, height)
            left_knee = get_point(landmarks, LEFT_KNEE, width, height)
            left_ankle = get_point(landmarks, LEFT_ANKLE, width, height)

            right_hip = get_point(landmarks, RIGHT_HIP, width, height)
            right_knee = get_point(landmarks, RIGHT_KNEE, width, height)
            right_ankle = get_point(landmarks, RIGHT_ANKLE, width, height)

            left_angle = angle_between_three_points(left_hip, left_knee, left_ankle)
            right_angle = angle_between_three_points(right_hip, right_knee, right_ankle)

            if left_angle is not None and right_angle is not None:
                mean_angle = (left_angle + right_angle) / 2

                left_hist.append(left_angle)
                right_hist.append(right_angle)
                mean_hist.append(mean_angle)

                smooth_left = float(np.mean(left_hist))
                smooth_right = float(np.mean(right_hist))
                smooth_mean = float(np.mean(mean_hist))

                asymmetry = abs(smooth_left - smooth_right)
                variance = float(np.std(mean_hist)) if len(mean_hist) > 1 else 0.0

                # -----------------------------
                # PHASE 1: TIME-BASED CALIBRATION
                # -----------------------------
                if not is_calibrated:
                    calibration_angles.append(smooth_mean)

                    remaining = max(0, CALIBRATION_SECONDS - elapsed)
                    draw_text(frame, "CALIBRATING: do 2-3 clean reps", 20, 130)
                    draw_text(frame, f"Time left: {remaining:.1f}s", 20, 160)
                    draw_text(frame, f"Current knee angle: {smooth_mean:.1f}", 20, 190)

                    if elapsed >= CALIBRATION_SECONDS:
                        athlete_deep_angle = min(calibration_angles)
                        athlete_standing_angle = max(calibration_angles)
                        movement_range = athlete_standing_angle - athlete_deep_angle

                        if movement_range < 20:
                            print("Calibration failed: movement range too small.")
                            print("Keep going: move back and do deeper clean reps.")
                            calibration_angles = []
                            start_time = time.time()
                            continue

                        deep_knee_bend_threshold = athlete_standing_angle - 0.60 * movement_range
                        shallow_knee_bend_threshold = athlete_standing_angle - 0.15 * movement_range

                        is_calibrated = True

                        print("Calibration complete")
                        print("Standing angle:", round(athlete_standing_angle, 2))
                        print("Deep angle:", round(athlete_deep_angle, 2))
                        print("Deep threshold:", round(deep_knee_bend_threshold, 2))
                        print("Shallow threshold:", round(shallow_knee_bend_threshold, 2))

                    writer.writerow([
                        round(elapsed, 3),
                        "calibration",
                        rep_count,
                        round(smooth_left, 3),
                        round(smooth_right, 3),
                        round(smooth_mean, 3),
                        round(asymmetry, 3),
                        round(variance, 3),
                        "",
                        "",
                        "",
                        "",
                        risk_score,
                        risk_label
                    ])

                # -----------------------------
                # PHASE 2: TRACKING + RISK SCORING
                # -----------------------------
                else:
                    # Rep starts when knee angle moves below personalized deep threshold.
                    if smooth_mean < deep_knee_bend_threshold:
                        if not in_rep:
                            in_rep = True
                            rep_start_time = time.time()
                            current_rep_min_angle = smooth_mean
                        else:
                            current_rep_min_angle = min(current_rep_min_angle, smooth_mean)

                    # Rep ends when athlete returns above personalized shallow threshold.
                    if in_rep and smooth_mean > shallow_knee_bend_threshold:
                        in_rep = False
                        rep_count += 1

                        rep_duration = max(time.time() - rep_start_time, 0.001)
                        rep_speed = 1 / rep_duration
                        rep_depth = current_rep_min_angle

                        rep_speeds.append(rep_speed)
                        rep_depths.append(rep_depth)

                        if len(rep_speeds) == BASELINE_REPS:
                            baseline_speed = float(np.mean(rep_speeds[:BASELINE_REPS]))

                        if len(rep_depths) == BASELINE_REPS:
                            baseline_depth = float(np.mean(rep_depths[:BASELINE_REPS]))

                        current_rep_min_angle = None
                        rep_start_time = None

                    depth_drop = 0
                    speed_drop_pct = 0

                    if asymmetry > ASYMMETRY_THRESHOLD:
                        risk_score += 1
                        risk_reasons.append("asymmetry")

                    if variance > ANGLE_VARIANCE_THRESHOLD:
                        risk_score += 1
                        risk_reasons.append("unstable motion")

                    if baseline_depth is not None and len(rep_depths) > BASELINE_REPS:
                        depth_drop = rep_depths[-1] - baseline_depth

                        if depth_drop > FATIGUE_DEPTH_DROP_THRESHOLD:
                            risk_score += 1
                            risk_reasons.append("shallower reps")

                    if baseline_speed is not None and len(rep_speeds) > BASELINE_REPS:
                        speed_drop_pct = (baseline_speed - rep_speeds[-1]) / baseline_speed

                        if speed_drop_pct > FATIGUE_SPEED_DROP_THRESHOLD:
                            risk_score += 1
                            risk_reasons.append("slower reps")

                    if risk_score >= HIGH_RISK_SCORE:
                        risk_label = "RED"
                    elif risk_score >= MED_RISK_SCORE:
                        risk_label = "YELLOW"
                    else:
                        risk_label = "GREEN"

                    draw_risk_box(frame, risk_label, risk_score)

                    draw_text(frame, f"Reps: {rep_count}", 20, 135)
                    draw_text(frame, f"Left knee: {smooth_left:.1f}", 20, 165)
                    draw_text(frame, f"Right knee: {smooth_right:.1f}", 20, 195)
                    draw_text(frame, f"Mean knee: {smooth_mean:.1f}", 20, 225)
                    draw_text(frame, f"Asymmetry: {asymmetry:.1f}", 20, 255)
                    draw_text(frame, f"Variance: {variance:.1f}", 20, 285)
                    draw_text(frame, f"Depth drop: {depth_drop:.1f}", 20, 315)
                    draw_text(frame, f"Speed drop: {speed_drop_pct * 100:.1f}%", 20, 345)

                    if baseline_depth is None or baseline_speed is None:
                        draw_text(frame, f"Building baseline: {rep_count}/{BASELINE_REPS} reps", 20, 385, 0.55)

                    if risk_reasons:
                        draw_text(frame, "Flags: " + ", ".join(risk_reasons), 20, 415, 0.55)

                    writer.writerow([
                        round(elapsed, 3),
                        "tracking",
                        rep_count,
                        round(smooth_left, 3),
                        round(smooth_right, 3),
                        round(smooth_mean, 3),
                        round(asymmetry, 3),
                        round(variance, 3),
                        round(deep_knee_bend_threshold, 3),
                        round(shallow_knee_bend_threshold, 3),
                        round(depth_drop, 3),
                        round(speed_drop_pct, 3),
                        risk_score,
                        risk_label
                    ])

        else:
            draw_text(frame, "No pose detected. Step back / improve lighting.", 20, 130)
            draw_text(frame, "Camera is still running. Risk detection paused.", 20, 160)

        cv2.imshow("Injury Risk MVP", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

cap.release()
csv_file.close()
cv2.destroyAllWindows()