from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from autocam_tracker.video.video_source import VideoSource


class VideoFileSource(VideoSource):
    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.cap: cv2.VideoCapture | None = None
        self.frame_index = 0
        self.timestamp_ms = 0.0
        self.frame_count = 0
        self.fps = 0.0

    def open(self) -> None:
        if not self.path.exists():
            raise FileNotFoundError(f"Video file not found: {self.path}")
        self.cap = cv2.VideoCapture(str(self.path))
        if not self.cap.isOpened():
            raise RuntimeError(f"Failed to open video file: {self.path}")
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        self.fps = float(self.cap.get(cv2.CAP_PROP_FPS) or 0.0)

    def read(self) -> tuple[bool, np.ndarray | None]:
        if self.cap is None:
            return False, None
        ok, frame = self.cap.read()
        if not ok:
            return False, None
        self.frame_index += 1
        self.timestamp_ms = float(self.cap.get(cv2.CAP_PROP_POS_MSEC))
        return True, frame

    def seek(self, frame_index: int) -> bool:
        if self.cap is None:
            return False
        target = max(0, min(int(frame_index), max(0, self.frame_count - 1)))
        ok = bool(self.cap.set(cv2.CAP_PROP_POS_FRAMES, target))
        if ok:
            self.frame_index = target
            self.timestamp_ms = 1000.0 * target / self.fps if self.fps > 0 else 0.0
        return ok

    def release(self) -> None:
        if self.cap is not None:
            self.cap.release()
        self.cap = None
