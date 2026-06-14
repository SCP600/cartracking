from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np

from autocam_tracker.detection.detection_models import VehicleDetection


class YOLO26Detector:
    def __init__(
        self,
        model_path: Path,
        conf: float = 0.15,
        imgsz: int = 960,
        vehicle_class_ids: Iterable[int] | None = (2, 3, 5, 7),
        device: int | str | None = None,
    ) -> None:
        self.model_path = Path(model_path)
        self.conf = conf
        self.imgsz = imgsz
        self.vehicle_class_ids = list(vehicle_class_ids) if vehicle_class_ids is not None else None
        self.device = device
        self._model = None

    def track(
        self,
        frame: np.ndarray,
        tracker_config: Path,
        camera_id: int,
        shot_id: int,
        frame_index: int,
        timestamp_ms: float,
    ) -> list[VehicleDetection]:
        model = self._get_model()
        track_options = {
            "persist": True,
            "tracker": str(tracker_config),
            "conf": self.conf,
            "imgsz": self.imgsz,
            "verbose": False,
        }
        if self.vehicle_class_ids is not None:
            track_options["classes"] = self.vehicle_class_ids
        if self.device is not None:
            track_options["device"] = self.device

        results = model.track(frame, **track_options)
        if not results:
            return []
        return self._to_detections(
            result=results[0],
            camera_id=camera_id,
            shot_id=shot_id,
            frame_index=frame_index,
            timestamp_ms=timestamp_ms,
        )

    def reset_tracking(self) -> None:
        # Recreate the model on scene cuts to clear tracker-local state.
        self._model = None

    def _get_model(self):
        if self._model is None:
            from ultralytics import YOLO

            self._model = YOLO(str(self.model_path))
        return self._model

    def _to_detections(self, result, camera_id: int, shot_id: int, frame_index: int, timestamp_ms: float) -> list[VehicleDetection]:
        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            return []

        xyxy = boxes.xyxy.cpu().numpy()
        confidences = boxes.conf.cpu().numpy() if boxes.conf is not None else np.zeros(len(xyxy))
        classes = boxes.cls.cpu().numpy().astype(int) if boxes.cls is not None else np.zeros(len(xyxy), dtype=int)
        track_ids = boxes.id.cpu().numpy().astype(int) if boxes.id is not None else np.full(len(xyxy), -1, dtype=int)
        names = result.names or {}

        detections: list[VehicleDetection] = []
        for index, (box, confidence, class_id, track_id) in enumerate(zip(xyxy, confidences, classes, track_ids)):
            x1, y1, x2, y2 = box
            x = max(0, int(round(x1)))
            y = max(0, int(round(y1)))
            w = max(0, int(round(x2 - x1)))
            h = max(0, int(round(y2 - y1)))
            label = names.get(int(class_id), "vehicle")
            detections.append(
                VehicleDetection(
                    detection_id=index,
                    local_track_id=int(track_id),
                    camera_id=camera_id,
                    shot_id=shot_id,
                    frame_index=frame_index,
                    timestamp_ms=timestamp_ms,
                    label=label,
                    confidence=float(confidence),
                    bbox=(x, y, w, h),
                    center=(x + w // 2, y + h // 2),
                )
            )
        return detections
