from __future__ import annotations

from collections import deque

from autocam_tracker.detection.detection_models import VehicleDetection


class DetectionHistory:
    def __init__(self, max_frames: int = 120) -> None:
        self.frames: deque[list[VehicleDetection]] = deque(maxlen=max_frames)

    def add(self, detections: list[VehicleDetection]) -> None:
        self.frames.append(detections)

