from __future__ import annotations

import numpy as np

from autocam_tracker.detection.detection_models import VehicleDetection
from autocam_tracker.identity.reacquire_engine import ReacquireEngine
from autocam_tracker.identity.vehicle_identity import VehicleIdentity


class GlobalIdentityManager:
    def __init__(self) -> None:
        self.next_global_vehicle_id = 1
        self.selected_identity: VehicleIdentity | None = None
        self.reacquire = ReacquireEngine()
        self.status = "Detecting"
        self.last_reacquire_score = 0.0
        self.target_lost_timeout_frames = 120
        self.lost_to_searching_frames = 5

    @property
    def selected_global_vehicle_id(self) -> int:
        return self.selected_identity.global_vehicle_id if self.selected_identity else -1

    @property
    def selected_local_track_id(self) -> int:
        return self.selected_identity.last_local_track_id if self.selected_identity else -1

    @property
    def selected_detection_id(self) -> int:
        return self.selected_identity.last_detection_id if self.selected_identity else -1

    @property
    def lost_frames(self) -> int:
        return self.selected_identity.lost_frames if self.selected_identity else 0

    def reset(self) -> None:
        self.selected_identity = None
        self.status = "Detecting"
        self.last_reacquire_score = 0.0
        self.reacquire.reset_pending()

    def handle_camera_cut(self, shot_id: int) -> None:
        if self.selected_identity is None:
            return
        self.selected_identity.shot_id = shot_id
        self.selected_identity.last_local_track_id = -1
        self.selected_identity.status = "CameraCut"
        self.status = "CameraCut"
        self.reacquire.reset_pending()

    def update(
        self,
        detections: list[VehicleDetection],
        selected_detection_id: tuple[int, int] | None,
        frame: np.ndarray,
        timestamp_ms: float,
        camera_id: int,
        shot_id: int,
    ) -> VehicleDetection | None:
        if selected_detection_id is not None:
            detection_id, local_track_id = selected_detection_id
            selected = self._find_selection(detections, detection_id, local_track_id)
            if selected is not None:
                self._select(selected, frame, timestamp_ms, camera_id, shot_id)

        if self.selected_identity is None:
            self.status = "Detecting"
            return None

        target = self._find_target_by_track(detections)
        if target is None:
            target, score = self.reacquire.choose(self.selected_identity, detections, frame)
            self.last_reacquire_score = score

        if target is not None:
            self._update_identity_from_detection(target, frame, timestamp_ms, camera_id, shot_id)
            target.selected = True
            self.status = "Tracking"
            self.selected_identity.status = self.status
            self._mark_selected(detections, target)
            return target

        self.selected_identity.lost_frames += 1
        if self.selected_identity.lost_frames > self.target_lost_timeout_frames:
            self.status = "TargetLost"
        elif self.selected_identity.lost_frames >= self.lost_to_searching_frames:
            self.status = "SearchingTarget"
        else:
            self.status = "Tracking"
        self.selected_identity.status = self.status
        self._mark_global_id(detections)
        return None

    def _select(self, detection: VehicleDetection, frame: np.ndarray, timestamp_ms: float, camera_id: int, shot_id: int) -> None:
        global_vehicle_id = detection.global_vehicle_id
        if global_vehicle_id < 0:
            global_vehicle_id = self.next_global_vehicle_id
        self.next_global_vehicle_id = max(self.next_global_vehicle_id, global_vehicle_id + 1)
        identity = VehicleIdentity(
            global_vehicle_id=global_vehicle_id,
            created_at_ms=timestamp_ms,
            label=detection.label,
            camera_id=camera_id,
            shot_id=shot_id,
        )
        self.selected_identity = identity
        self._update_identity_from_detection(detection, frame, timestamp_ms, camera_id, shot_id)
        self.status = "TargetSelected"
        self.reacquire.reset_pending()

    def _update_identity_from_detection(
        self,
        detection: VehicleDetection,
        frame: np.ndarray,
        timestamp_ms: float,
        camera_id: int,
        shot_id: int,
    ) -> None:
        if self.selected_identity is None:
            return
        identity = self.selected_identity
        identity.last_seen_ms = timestamp_ms
        identity.camera_id = camera_id
        identity.shot_id = shot_id
        identity.last_detection_id = detection.detection_id
        identity.last_local_track_id = detection.local_track_id
        identity.last_bbox = detection.bbox
        identity.last_center = detection.center
        identity.lost_frames = 0
        identity.color_signature = self.reacquire.color_signature(frame, detection.bbox)
        if detection.thumbnail is not None and len(identity.thumbnails) < 12:
            identity.thumbnails.append(detection.thumbnail)
        detection.global_vehicle_id = identity.global_vehicle_id

    def _find_detection(self, detections: list[VehicleDetection], detection_id: int) -> VehicleDetection | None:
        for detection in detections:
            if detection.detection_id == detection_id:
                return detection
        return None

    def _find_selection(
        self,
        detections: list[VehicleDetection],
        detection_id: int,
        local_track_id: int,
    ) -> VehicleDetection | None:
        if local_track_id >= 0:
            for detection in detections:
                if detection.local_track_id == local_track_id:
                    return detection
        return self._find_detection(detections, detection_id)

    def _find_target_by_track(self, detections: list[VehicleDetection]) -> VehicleDetection | None:
        if self.selected_identity is None:
            return None
        track_id = self.selected_identity.last_local_track_id
        if track_id < 0:
            return None
        for detection in detections:
            if detection.local_track_id == track_id:
                return detection
        return None

    def _mark_selected(self, detections: list[VehicleDetection], selected: VehicleDetection) -> None:
        self._mark_global_id(detections)
        if self.selected_identity is None:
            return
        for detection in detections:
            detection.selected = detection is selected
            if detection.selected:
                detection.global_vehicle_id = self.selected_identity.global_vehicle_id

    def _mark_global_id(self, detections: list[VehicleDetection]) -> None:
        if self.selected_identity is None:
            return
        for detection in detections:
            if detection.local_track_id >= 0 and detection.local_track_id == self.selected_identity.last_local_track_id:
                detection.global_vehicle_id = self.selected_identity.global_vehicle_id
