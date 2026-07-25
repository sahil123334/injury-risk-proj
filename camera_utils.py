"""
camera_utils.py

Lets you pick which camera device to use instead of always grabbing
index 0. This matters on macOS in particular: if an iPhone is nearby
and signed into the same Apple ID, macOS can register it as a
Continuity Camera device, which may enumerate before your built-in
FaceTime camera.

`--list-cameras` prints what's detected and exits; `launcher.py` uses
`discover_cameras()` to populate the camera picker in the startup window,
and `--camera-index N` (main.py) skips both and uses that index directly.
"""

import json
import platform
import subprocess
import time
from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2

import config


@dataclass
class CameraInfo:
    index: int
    name: str
    resolution: Tuple[int, int]


def _probe_resolution(index: int, use_avfoundation: bool, warmup_attempts: int = 10) -> Optional[Tuple[int, int]]:
    """Try to open `index` and read one real frame. Returns (width, height) or None."""
    backend = cv2.CAP_AVFOUNDATION if use_avfoundation else cv2.CAP_ANY
    cap = cv2.VideoCapture(index, backend)

    if not cap.isOpened():
        cap.release()
        return None

    frame = None
    for _ in range(warmup_attempts):
        ret, candidate = cap.read()
        if ret:
            frame = candidate
            break
        time.sleep(0.02)

    cap.release()

    if frame is None:
        return None

    height, width = frame.shape[:2]
    return (width, height)


def _macos_camera_names() -> List[str]:
    """
    Best-effort friendly names via `system_profiler`. Order is not
    guaranteed to match OpenCV/AVFoundation's device index order, so
    this is a hint, not a guarantee -- callers should always show the
    resolution too so the user can visually confirm which is which.
    """
    if platform.system() != "Darwin":
        return []

    try:
        completed = subprocess.run(
            ["system_profiler", "SPCameraDataType", "-json"],
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        )
        data = json.loads(completed.stdout)
        entries = data.get("SPCameraDataType", [])
        return [entry.get("_name", "Unknown camera") for entry in entries]
    except (subprocess.SubprocessError, json.JSONDecodeError, OSError):
        return []


def discover_cameras(max_index: int = 5) -> List[CameraInfo]:
    use_avfoundation = config.CAMERA.use_avfoundation and platform.system() == "Darwin"
    names = _macos_camera_names()

    cameras: List[CameraInfo] = []
    for index in range(max_index):
        resolution = _probe_resolution(index, use_avfoundation)
        if resolution is None:
            continue
        name = names[index] if index < len(names) else f"Camera {index}"
        cameras.append(CameraInfo(index=index, name=name, resolution=resolution))

    return cameras


def print_camera_list(cameras: List[CameraInfo]) -> None:
    if not cameras:
        print("No cameras detected.")
        return
    print("Available cameras:")
    for cam in cameras:
        print(f"  [{cam.index}] {cam.name} ({cam.resolution[0]}x{cam.resolution[1]})")
