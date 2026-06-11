from __future__ import annotations

from collections import OrderedDict

import cv2
import numpy as np

from autocam_tracker.detection.detection_models import RecognizedVehicleSummary, VehicleDetection
from autocam_tracker.utils.geometry import center_distance


class RecognizedVehicleRegistry:
    def __init__(self, max_items: int = 80) -> None:
        self.max_items = max_items
        self._items: OrderedDict[str, RecognizedVehicleSummary] = OrderedDict()
        self._next_registry_id = 1
        self.max_merge_gap_frames = 90
        self.merge_min_score = 0.58

    def update(self, detections: list[VehicleDetection], selected_global_vehicle_id: int, tracking_status: str) -> None:
        seen_this_frame: set[str] = set()
        for detection in detections:
            key = self._key(detection)
            signature = self._color_signature(detection)
            key, match_score = self._resolve_key(detection, key, signature)
            seen_this_frame.add(key)
            if key not in self._items:
                self._items[key] = RecognizedVehicleSummary(
                    registry_id=key,
                    local_track_id=detection.local_track_id,
                    global_vehicle_id=detection.global_vehicle_id,
                    camera_id=detection.camera_id,
                    shot_id=detection.shot_id,
                    first_frame_index=detection.frame_index,
                    last_frame_index=detection.frame_index,
                    last_seen_ms=detection.timestamp_ms,
                    label=detection.label,
                    color_signature=signature,
                )
            item = self._items[key]
            item.local_track_id = detection.local_track_id
            if detection.global_vehicle_id >= 0:
                item.global_vehicle_id = detection.global_vehicle_id
            item.camera_id = detection.camera_id
            item.shot_id = detection.shot_id
            item.last_frame_index = detection.frame_index
            item.last_seen_ms = detection.timestamp_ms
            item.label = detection.label
            item.confidence = detection.confidence
            item.bbox = detection.bbox
            item.seen_frames += 1
            item.match_score = match_score
            item.selected = (
                selected_global_vehicle_id >= 0
                and item.global_vehicle_id == selected_global_vehicle_id
            )
            item.status = tracking_status if item.selected else "Seen"
            if detection.thumbnail is not None:
                item.thumbnail = detection.thumbnail
            if signature is not None:
                item.color_signature = self._blend_signature(item.color_signature, signature)
            if detection.local_track_id >= 0 and detection.local_track_id not in item.local_track_aliases:
                item.local_track_aliases.append(detection.local_track_id)
                item.local_track_aliases = item.local_track_aliases[-8:]
            detection.global_vehicle_id = item.global_vehicle_id
            self._items.move_to_end(key)

        for key, item in self._items.items():
            if key not in seen_this_frame and not item.selected:
                item.status = "NotVisible"

        while len(self._items) > self.max_items:
            self._items.popitem(last=False)

    def summaries(self) -> list[RecognizedVehicleSummary]:
        items = list(self._items.values())
        items.sort(
            key=lambda item: (
                item.selected,
                item.global_vehicle_id >= 0,
                item.last_frame_index,
            ),
            reverse=True,
        )
        return items

    def clear(self) -> None:
        self._items.clear()
        self._next_registry_id = 1

    def _key(self, detection: VehicleDetection) -> str:
        if detection.global_vehicle_id >= 0:
            return f"G{detection.global_vehicle_id}"
        return ""

    def _resolve_key(
        self,
        detection: VehicleDetection,
        preferred_key: str,
        signature: np.ndarray | None,
    ) -> tuple[str, float]:
        if preferred_key:
            return preferred_key, 1.0

        track_key = self._find_by_track_alias(detection)
        if track_key is not None:
            return track_key, 1.0

        match_key, score = self._find_appearance_match(detection, signature)
        if match_key is not None:
            return match_key, score

        key = f"R{self._next_registry_id}"
        self._next_registry_id += 1
        return key, 0.0

    def _find_by_track_alias(self, detection: VehicleDetection) -> str | None:
        if detection.local_track_id < 0:
            return None
        for key, item in self._items.items():
            if item.shot_id == detection.shot_id and detection.local_track_id in item.local_track_aliases:
                return key
        return None

    def _find_appearance_match(
        self,
        detection: VehicleDetection,
        signature: np.ndarray | None,
    ) -> tuple[str | None, float]:
        best_key = None
        best_score = 0.0
        for key, item in self._items.items():
            if item.global_vehicle_id >= 0:
                continue
            if item.shot_id != detection.shot_id:
                continue
            if detection.frame_index - item.last_frame_index > self.max_merge_gap_frames:
                continue
            score = self._merge_score(item, detection, signature)
            if score > best_score:
                best_key = key
                best_score = score
        if best_key is not None and best_score >= self.merge_min_score:
            return best_key, best_score
        return None, best_score

    def _merge_score(
        self,
        item: RecognizedVehicleSummary,
        detection: VehicleDetection,
        signature: np.ndarray | None,
    ) -> float:
        color = self._color_similarity(item.color_signature, signature)
        size = self._size_similarity(item.bbox, detection.bbox)
        motion = self._motion_similarity(item, detection)
        confidence = max(0.0, min(1.0, detection.confidence))
        return 0.50 * color + 0.22 * size + 0.18 * motion + 0.10 * confidence

    def _color_signature(self, detection: VehicleDetection) -> np.ndarray | None:
        if detection.thumbnail is None:
            return None
        hsv = cv2.cvtColor(detection.thumbnail, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1], None, [24, 16], [0, 180, 0, 256])
        cv2.normalize(hist, hist, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
        return hist.flatten().astype("float32")

    def _blend_signature(self, old: np.ndarray | None, new: np.ndarray) -> np.ndarray:
        if old is None:
            return new
        blended = old.astype("float32") * 0.82 + new.astype("float32") * 0.18
        norm = np.linalg.norm(blended)
        if norm > 0:
            blended = blended / norm
        return blended.astype("float32")

    def _color_similarity(self, a: np.ndarray | None, b: np.ndarray | None) -> float:
        if a is None or b is None:
            return 0.0
        score = cv2.compareHist(a.astype("float32"), b.astype("float32"), cv2.HISTCMP_CORREL)
        return float(max(0.0, min(1.0, score)))

    def _size_similarity(self, a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
        area_a = max(1, a[2] * a[3])
        area_b = max(1, b[2] * b[3])
        area = min(area_a, area_b) / max(area_a, area_b)
        aspect_a = a[2] / max(1, a[3])
        aspect_b = b[2] / max(1, b[3])
        aspect = min(aspect_a, aspect_b) / max(aspect_a, aspect_b)
        return float(0.65 * area + 0.35 * aspect)

    def _motion_similarity(self, item: RecognizedVehicleSummary, detection: VehicleDetection) -> float:
        item_center = (item.bbox[0] + item.bbox[2] // 2, item.bbox[1] + item.bbox[3] // 2)
        distance = center_distance(item_center, detection.center)
        reference = max(80.0, max(item.bbox[2], item.bbox[3], detection.bbox[2], detection.bbox[3]) * 3.5)
        return float(max(0.0, 1.0 - distance / reference))
