from __future__ import annotations

from collections import OrderedDict

import cv2
import numpy as np

from autocam_tracker.detection.detection_models import RecognizedVehicleSummary, VehicleDetection
from autocam_tracker.utils.geometry import center_distance


class RecognizedVehicleRegistry:
    def __init__(
        self,
        max_items: int = 80,
        reid_memory_size: int = 24,
        reid_match_threshold: float = 0.82,
        reid_cross_shot_threshold: float = 0.86,
        reid_margin: float = 0.04,
        reid_duplicate_similarity: float = 0.985,
    ) -> None:
        self.max_items = max_items
        self._items: OrderedDict[str, RecognizedVehicleSummary] = OrderedDict()
        self._next_registry_id = 1
        self._next_global_vehicle_id = 1
        self.max_merge_gap_frames = 90
        self.merge_min_score = 0.58
        self.reid_memory_size = max(0, reid_memory_size)
        self.reid_match_threshold = reid_match_threshold
        self.reid_cross_shot_threshold = reid_cross_shot_threshold
        self.reid_margin = reid_margin
        self.reid_duplicate_similarity = reid_duplicate_similarity

    def apply_known_ids(self, detections: list[VehicleDetection]) -> None:
        assigned_keys: set[str] = set()
        for detection in detections:
            if detection.global_vehicle_id >= 0:
                self._reserve_global_id(detection.global_vehicle_id)
                assigned_keys.add(f"G{detection.global_vehicle_id}")
                continue
            signature = self._color_signature(detection)
            key = self._find_by_track_alias(detection)
            if key is None or key in assigned_keys:
                key, _ = self._find_appearance_match(detection, signature, exclude_keys=assigned_keys)
            if key is None or key in assigned_keys:
                continue
            item = self._items.get(key)
            if item is not None and item.global_vehicle_id >= 0:
                detection.global_vehicle_id = item.global_vehicle_id
                assigned_keys.add(key)

    def update(self, detections: list[VehicleDetection], selected_global_vehicle_id: int, tracking_status: str) -> None:
        seen_this_frame: set[str] = set()
        for detection in detections:
            key = self._key(detection)
            signature = self._color_signature(detection)
            key, match_score = self._resolve_key(detection, key, signature, exclude_keys=seen_this_frame)
            seen_this_frame.add(key)
            if key not in self._items:
                self._items[key] = RecognizedVehicleSummary(
                    registry_id=key,
                    local_track_id=detection.local_track_id,
                    global_vehicle_id=self._global_id_from_key(key),
                    camera_id=detection.camera_id,
                    shot_id=detection.shot_id,
                    first_frame_index=detection.frame_index,
                    last_frame_index=detection.frame_index,
                    last_seen_ms=detection.timestamp_ms,
                    label=detection.label,
                    color_signature=signature,
                )
            item = self._items[key]
            shot_changed = item.shot_id != detection.shot_id
            if item.global_vehicle_id >= 0:
                detection.global_vehicle_id = item.global_vehicle_id
            item.local_track_id = detection.local_track_id
            if detection.global_vehicle_id >= 0:
                item.global_vehicle_id = detection.global_vehicle_id
                self._reserve_global_id(item.global_vehicle_id)
            item.camera_id = detection.camera_id
            if shot_changed:
                item.local_track_aliases = []
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
            self._update_reid_memory(item, detection)
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
        self._next_global_vehicle_id = 1

    def reset_runtime_state(self) -> None:
        for item in self._items.values():
            item.selected = False
            item.status = "NotVisible"
            item.match_score = 0.0

    def _key(self, detection: VehicleDetection) -> str:
        if detection.global_vehicle_id >= 0:
            return f"G{detection.global_vehicle_id}"
        return ""

    def _resolve_key(
        self,
        detection: VehicleDetection,
        preferred_key: str,
        signature: np.ndarray | None,
        exclude_keys: set[str] | None = None,
    ) -> tuple[str, float]:
        if exclude_keys is None:
            exclude_keys = set()

        if preferred_key and preferred_key not in exclude_keys:
            return preferred_key, 1.0

        track_key = self._find_by_track_alias(detection)
        if track_key is not None and track_key not in exclude_keys:
            return track_key, 1.0

        match_key, score = self._find_appearance_match(detection, signature, exclude_keys)
        if match_key is not None:
            return match_key, score

        new_key = self._new_global_key()
        while new_key in exclude_keys:
            new_key = self._new_global_key()
        return new_key, 0.0

    def local_track_aliases_for_global(self, global_vehicle_id: int, shot_id: int | None = None) -> list[int]:
        for item in self._items.values():
            if item.global_vehicle_id == global_vehicle_id:
                if shot_id is not None and item.shot_id != shot_id:
                    return []
                aliases = list(item.local_track_aliases)
                if item.local_track_id >= 0 and item.local_track_id not in aliases:
                    aliases.append(item.local_track_id)
                return aliases
        return []

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
        exclude_keys: set[str] | None = None,
    ) -> tuple[str | None, float]:
        if exclude_keys is None:
            exclude_keys = set()
        reid_key, reid_score = self._find_reid_match(detection, exclude_keys)
        best_key = reid_key
        best_score = reid_score if reid_key is not None else 0.0
        for key, item in self._items.items():
            if key in exclude_keys:
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
        return None, max(best_score, reid_score)

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
        reid = self._reid_similarity(item, detection)
        if reid > 0.0:
            detection.reid_score = max(detection.reid_score, reid)
            return 0.42 * reid + 0.30 * color + 0.13 * size + 0.08 * motion + 0.07 * confidence
        return 0.50 * color + 0.22 * size + 0.18 * motion + 0.10 * confidence

    def _find_reid_match(self, detection: VehicleDetection, exclude_keys: set[str] | None = None) -> tuple[str | None, float]:
        if exclude_keys is None:
            exclude_keys = set()
        feature = self._normalized_reid_feature(detection)
        if feature is None:
            return None, 0.0

        scored: list[tuple[float, float, str]] = []
        for key, item in self._items.items():
            if key in exclude_keys:
                continue
            if not item.reid_features:
                continue
            if item.shot_id == detection.shot_id and item.last_frame_index == detection.frame_index:
                continue
            score = self._feature_memory_similarity(item.reid_features, feature)
            threshold = (
                self.reid_match_threshold
                if item.shot_id == detection.shot_id
                else self.reid_cross_shot_threshold
            )
            scored.append((score, threshold, key))

        if not scored:
            return None, 0.0

        scored.sort(key=lambda item: item[0], reverse=True)
        best_score, threshold, best_key = scored[0]
        second_score = scored[1][0] if len(scored) > 1 else 0.0
        detection.reid_score = max(detection.reid_score, best_score)
        if best_score < threshold or best_score - second_score < self.reid_margin:
            return None, best_score
        detection.reid_matched = True
        return best_key, best_score

    def _new_global_key(self) -> str:
        while f"G{self._next_global_vehicle_id}" in self._items:
            self._next_global_vehicle_id += 1
        key = f"G{self._next_global_vehicle_id}"
        self._next_registry_id = max(self._next_registry_id, self._next_global_vehicle_id + 1)
        self._next_global_vehicle_id += 1
        return key

    def _global_id_from_key(self, key: str) -> int:
        if key.startswith("G") and key[1:].isdigit():
            global_id = int(key[1:])
            self._reserve_global_id(global_id)
            return global_id
        global_id = self._next_global_vehicle_id
        self._reserve_global_id(global_id)
        return global_id

    def _reserve_global_id(self, global_vehicle_id: int) -> None:
        if global_vehicle_id >= self._next_global_vehicle_id:
            self._next_global_vehicle_id = global_vehicle_id + 1
        self._next_registry_id = max(self._next_registry_id, self._next_global_vehicle_id)

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

    def _update_reid_memory(self, item: RecognizedVehicleSummary, detection: VehicleDetection) -> None:
        feature = self._normalized_reid_feature(detection)
        if feature is None or self.reid_memory_size <= 0:
            item.reid_feature_count = len(item.reid_features)
            return

        if not item.reid_features:
            item.reid_features.append(feature)
            item.reid_feature_count = len(item.reid_features)
            return

        similarities = np.array([self._cosine_similarity(feature, existing) for existing in item.reid_features], dtype=np.float32)
        if float(similarities.max()) >= self.reid_duplicate_similarity:
            item.reid_feature_count = len(item.reid_features)
            return

        if len(item.reid_features) < self.reid_memory_size:
            item.reid_features.append(feature)
            item.reid_feature_count = len(item.reid_features)
            return

        replace_index = self._most_redundant_feature_index(item.reid_features)
        if replace_index >= 0 and float(similarities.max()) < self._feature_redundancy(item.reid_features, replace_index):
            item.reid_features[replace_index] = feature
        item.reid_feature_count = len(item.reid_features)

    def _most_redundant_feature_index(self, features: list[np.ndarray]) -> int:
        if len(features) <= 1:
            return 0 if features else -1
        redundancies = [self._feature_redundancy(features, index) for index in range(len(features))]
        return int(np.argmax(np.array(redundancies, dtype=np.float32)))

    def _feature_redundancy(self, features: list[np.ndarray], index: int) -> float:
        if len(features) <= 1:
            return 1.0
        target = features[index]
        return max(self._cosine_similarity(target, other) for other_index, other in enumerate(features) if other_index != index)

    def _reid_similarity(self, item: RecognizedVehicleSummary, detection: VehicleDetection) -> float:
        feature = self._normalized_reid_feature(detection)
        if feature is None or not item.reid_features:
            return 0.0
        return self._feature_memory_similarity(item.reid_features, feature)

    def _feature_memory_similarity(self, features: list[np.ndarray], feature: np.ndarray) -> float:
        if not features:
            return 0.0
        return max(self._cosine_similarity(feature, existing) for existing in features)

    def _normalized_reid_feature(self, detection: VehicleDetection) -> np.ndarray | None:
        if detection.reid_feature is None:
            return None
        feature = np.asarray(detection.reid_feature, dtype=np.float32).reshape(-1)
        norm = float(np.linalg.norm(feature))
        if norm <= 0.0:
            return None
        feature = (feature / norm).astype("float32")
        detection.reid_feature = feature
        return feature

    def _cosine_similarity(self, left: np.ndarray, right: np.ndarray) -> float:
        if left.shape != right.shape:
            return 0.0
        return float(np.clip(np.dot(left, right), -1.0, 1.0))
