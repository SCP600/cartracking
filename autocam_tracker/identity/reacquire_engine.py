from __future__ import annotations

import cv2
import numpy as np

from autocam_tracker.detection.detection_models import VehicleDetection
from autocam_tracker.identity.vehicle_identity import VehicleIdentity
from autocam_tracker.utils.geometry import clamp_bbox, center_distance


class ReacquireEngine:
    def __init__(self, min_score: float = 0.65, margin: float = 0.12, confirm_frames: int = 3) -> None:
        self.min_score = min_score
        self.margin = margin
        self.confirm_frames = confirm_frames
        self._pending_track_id: int | None = None
        self._pending_count = 0

    def reset_pending(self) -> None:
        self._pending_track_id = None
        self._pending_count = 0

    def color_signature(self, frame: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray | None:
        x, y, w, h = clamp_bbox(bbox, frame.shape[1], frame.shape[0])
        if w <= 1 or h <= 1:
            return None
        crop = frame[y : y + h, x : x + w]
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1], None, [24, 16], [0, 180, 0, 256])
        cv2.normalize(hist, hist, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
        return hist.flatten()

    def choose(
        self,
        identity: VehicleIdentity,
        detections: list[VehicleDetection],
        frame: np.ndarray,
    ) -> tuple[VehicleDetection | None, float]:
        if not detections:
            self.reset_pending()
            return None, 0.0

        scored = [(self._score(identity, detection, frame), detection) for detection in detections]
        scored.sort(key=lambda item: item[0], reverse=True)
        best_score, best = scored[0]
        second_score = scored[1][0] if len(scored) > 1 else 0.0

        if best_score < self.min_score or best_score - second_score < self.margin:
            self.reset_pending()
            return None, best_score

        pending_key = best.local_track_id if best.local_track_id >= 0 else best.detection_id
        if pending_key == self._pending_track_id:
            self._pending_count += 1
        else:
            self._pending_track_id = pending_key
            self._pending_count = 1

        if self._pending_count >= self.confirm_frames:
            self.reset_pending()
            return best, best_score
        return None, best_score

    def _score(self, identity: VehicleIdentity, detection: VehicleDetection, frame: np.ndarray) -> float:
        local_tracker_match = 1.0 if detection.local_track_id >= 0 and detection.local_track_id == identity.last_local_track_id else 0.0
        color = self._color_similarity(identity, detection, frame)
        size = self._size_similarity(identity.last_bbox, detection.bbox)
        motion = self._motion_similarity(identity.last_center, detection.center, frame.shape[1], frame.shape[0])
        confidence = max(0.0, min(1.0, detection.confidence))
        return (
            0.40 * local_tracker_match
            + 0.25 * color
            + 0.15 * size
            + 0.10 * motion
            + 0.10 * confidence
        )

    def _color_similarity(self, identity: VehicleIdentity, detection: VehicleDetection, frame: np.ndarray) -> float:
        if identity.color_signature is None:
            return 0.0
        signature = self.color_signature(frame, detection.bbox)
        if signature is None:
            return 0.0
        score = cv2.compareHist(identity.color_signature.astype("float32"), signature.astype("float32"), cv2.HISTCMP_CORREL)
        return float(max(0.0, min(1.0, score)))

    def _size_similarity(self, a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
        area_a = max(1, a[2] * a[3])
        area_b = max(1, b[2] * b[3])
        ratio = min(area_a, area_b) / max(area_a, area_b)
        aspect_a = a[2] / max(a[3], 1)
        aspect_b = b[2] / max(b[3], 1)
        aspect = min(aspect_a, aspect_b) / max(aspect_a, aspect_b)
        return float(0.7 * ratio + 0.3 * aspect)

    def _motion_similarity(self, previous: tuple[int, int], current: tuple[int, int], frame_w: int, frame_h: int) -> float:
        diagonal = max(1.0, (frame_w**2 + frame_h**2) ** 0.5)
        distance = center_distance(previous, current)
        return float(max(0.0, 1.0 - distance / (0.6 * diagonal)))

