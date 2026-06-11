from __future__ import annotations

import time

import cv2
import numpy as np

from autocam_tracker.video.video_source import VideoSource


class WebcamSource(VideoSource):
    def __init__(self, camera_index: int = 0) -> None:
        self.camera_index = camera_index
        self.cap: cv2.VideoCapture | None = None
        self.frame_index = 0
        self._start = time.perf_counter()
        self.timestamp_ms = 0.0

    def open(self) -> None:
        self.cap = cv2.VideoCapture(self.camera_index)
        if not self.cap.isOpened():
            raise RuntimeError(f"Failed to open webcam index {self.camera_index}")

    def read(self) -> tuple[bool, np.ndarray | None]:
        if self.cap is None:
            return False, None
        ok, frame = self.cap.read()
        if not ok:
            return False, None
        self.frame_index += 1
        self.timestamp_ms = (time.perf_counter() - self._start) * 1000.0
        return True, frame

    def release(self) -> None:
        if self.cap is not None:
            self.cap.release()
        self.cap = None

