"""
camera_utils.py

Lets you pick which camera device to use instead of always grabbing
index 0. This matters on macOS in particular: if an iPhone is nearby
and signed into the same Apple ID, macOS can register it as a
Continuity Camera device, which may enumerate before your built-in
FaceTime camera.

Two ways to pick a camera:
  --list-cameras            print what's available and exit
  --camera-index N          use that index directly, no prompting

If neither is given and more than one camera is found, main.py prompts
interactively at startup (before the OpenCV window opens).
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


def choose_camera_index(explicit_index: Optional[int]) -> int:
    """
    Resolution order: an explicit --camera-index always wins. Otherwise,
    probe for cameras; auto-pick if there's exactly one, prompt if there
    are several, and fall back to config's default if none were found
    (matches original single-camera behavior rather than hard-failing).
    """
    if explicit_index is not None:
        return explicit_index

    cameras = discover_cameras()

    if not cameras:
        print(f"No cameras detected via probing; defaulting to index {config.CAMERA.index}.")
        return config.CAMERA.index

    if len(cameras) == 1:
        cam = cameras[0]
        print(f"Using the only camera found: [{cam.index}] {cam.name} ({cam.resolution[0]}x{cam.resolution[1]})")
        return cam.index

    print_camera_list(cameras)
    valid_indices = {cam.index for cam in cameras}
    default_index = cameras[0].index

    while True:
        try:
            choice = input(f"Pick a camera index [{default_index}]: ").strip()
        except EOFError:
            # No interactive stdin available (e.g. piped/non-interactive shell).
            print(f"No interactive input available; defaulting to index {default_index}.")
            print("Tip: pass --camera-index N directly to skip this prompt.")
            return default_index

        if choice == "":
            return default_index
        try:
            choice_index = int(choice)
        except ValueError:
            print("Please enter a number shown above.")
            continue
        if choice_index in valid_indices:
            return choice_index
        print("That index wasn't in the list above -- try again.")
