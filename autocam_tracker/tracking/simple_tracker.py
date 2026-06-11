from __future__ import annotations

from autocam_tracker.detection.detection_models import VehicleDetection
from autocam_tracker.utils.geometry import center_distance


class SimpleTracker:
    def nearest(self, previous_center: tuple[int, int], detections: list[VehicleDetection], max_distance: float) -> VehicleDetection | None:
        best = None
        best_distance = max_distance
        for detection in detections:
            distance = center_distance(previous_center, detection.center)
            if distance <= best_distance:
                best = detection
                best_distance = distance
        return best

