from __future__ import annotations

from autocam_tracker.detection.detection_models import VehicleDetection


class DetectionStore:
    def __init__(self) -> None:
        self.current: list[VehicleDetection] = []

    def update(self, detections: list[VehicleDetection]) -> None:
        self.current = detections

    def find_by_detection_id(self, detection_id: int) -> VehicleDetection | None:
        for detection in self.current:
            if detection.detection_id == detection_id:
                return detection
        return None

