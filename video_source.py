"""
video_source.py

Abstracts "frames from a live webcam" vs "frames from a video file on
disk" behind one interface, so main.py's loop -- and everything
downstream of it (calibration, rep tracking, risk scoring) -- doesn't
need to know or care which one it's driving.

Both sources hand back an `elapsed` value in seconds: for the live
camera it's wall-clock time since the first frame; for a file it's
`frame_index / fps`, i.e. the video's own timeline. Calibration and rep
tracking only ever consume *that* clock, and only ever look at
differences between readings, so the exact same downstream logic runs
whether you're standing in front of the camera live or replaying a
clip recorded yesterday.
"""

import os
import time
from typing import Optional, Tuple

import cv2

import config


class LiveCameraSource:
    is_live = True

    def __init__(self, index: int):
        use_avfoundation = config.CAMERA.use_avfoundation and _is_macos()
        backend = cv2.CAP_AVFOUNDATION if use_avfoundation else cv2.CAP_ANY
        self._cap = cv2.VideoCapture(index, backend)

        if config.CAMERA.requested_width:
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.CAMERA.requested_width)
        if config.CAMERA.requested_height:
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.CAMERA.requested_height)

        self._start_time: Optional[float] = None

    def is_opened(self) -> bool:
        return self._cap.isOpened()

    def read(self) -> Tuple[bool, Optional["cv2.Mat"], float]:
        ret, frame = self._cap.read()
        now = time.time()
        if self._start_time is None:
            self._start_time = now
        elapsed = now - self._start_time
        return ret, frame, elapsed

    def release(self) -> None:
        self._cap.release()

    def describe(self) -> str:
        return "live camera"


class FileVideoSource:
    is_live = False

    def __init__(self, path: str):
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Video file not found: '{path}'")

        self._cap = cv2.VideoCapture(path)
        fps = self._cap.get(cv2.CAP_PROP_FPS)
        self._fps = fps if fps and fps > 1 else 30.0
        self._frame_index = 0
        self._path = path

    def is_opened(self) -> bool:
        return self._cap.isOpened()

    def read(self) -> Tuple[bool, Optional["cv2.Mat"], float]:
        ret, frame = self._cap.read()
        elapsed = self._frame_index / self._fps
        if ret:
            self._frame_index += 1
        return ret, frame, elapsed

    def release(self) -> None:
        self._cap.release()

    def describe(self) -> str:
        return os.path.basename(self._path)


def _is_macos() -> bool:
    import platform
    return platform.system() == "Darwin"
