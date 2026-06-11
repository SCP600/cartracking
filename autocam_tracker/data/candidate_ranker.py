from __future__ import annotations

from autocam_tracker.detection.detection_models import VehicleDetection


class CandidateRanker:
    def rank(self, detections: list[VehicleDetection]) -> list[VehicleDetection]:
        return sorted(detections, key=lambda item: (item.confidence, item.bbox[2] * item.bbox[3]), reverse=True)

