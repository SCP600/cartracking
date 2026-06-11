from __future__ import annotations

import time

import cv2
import numpy as np

from autocam_tracker.video.video_source import VideoSource


class ScreenRegionSource(VideoSource):
    def __init__(self, left: int, top: int, width: int, height: int, max_fps: float = 30.0) -> None:
        self.region = {
            "left": int(left),
            "top": int(top),
            "width": int(width),
            "height": int(height),
        }
        self.max_fps = max_fps
        self.frame_index = 0
        self.timestamp_ms = 0.0
        self._start = time.perf_counter()
        self._last_capture = 0.0
        self._sct = None

    def open(self) -> None:
        if self.region["width"] < 16 or self.region["height"] < 16:
            raise ValueError("Screen region is too small.")
        from mss import MSS

        self._sct = MSS()

    def read(self) -> tuple[bool, np.ndarray | None]:
        if self._sct is None:
            return False, None

        self._throttle()
        shot = self._sct.grab(self.region)
        bgra = np.asarray(shot)
        frame = cv2.cvtColor(bgra, cv2.COLOR_BGRA2BGR)
        self.frame_index += 1
        self.timestamp_ms = (time.perf_counter() - self._start) * 1000.0
        return True, frame

    def release(self) -> None:
        if self._sct is not None:
            self._sct.close()
        self._sct = None

    def _throttle(self) -> None:
        if self.max_fps <= 0:
            return
        min_interval = 1.0 / self.max_fps
        now = time.perf_counter()
        elapsed = now - self._last_capture
        if elapsed < min_interval:
            time.sleep(min_interval - elapsed)
        self._last_capture = time.perf_counter()


def capture_screen_region_once(left: int, top: int, width: int, height: int) -> np.ndarray:
    source = ScreenRegionSource(left, top, width, height, max_fps=0)
    source.open()
    try:
        ok, frame = source.read()
        if not ok or frame is None:
            raise RuntimeError("Failed to capture selected screen region.")
        return frame
    finally:
        source.release()
