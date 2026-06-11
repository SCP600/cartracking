from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from autocam_tracker.detection.detection_models import VehicleDetection


@dataclass
class CropResult:
    frame: np.ndarray
    crop_x: int
    crop_y: int
    crop_w: int
    crop_h: int
    zoom_ratio: float
    zoom_error: float
    error_x: float
    error_y: float
    normalized_error_x: float
    normalized_error_y: float


class CropController:
    def __init__(self, max_zoom: float = 2.5, smoothing: float = 0.18, dead_zone: float = 0.04) -> None:
        self.max_zoom = max_zoom
        self.smoothing = smoothing
        self.dead_zone = dead_zone
        self.crop_center: tuple[float, float] | None = None
        self.zoom_ratio = 1.0

    def reset(self) -> None:
        self.crop_center = None
        self.zoom_ratio = 1.0

    def crop(self, frame: np.ndarray, target: VehicleDetection | None) -> CropResult:
        frame_h, frame_w = frame.shape[:2]
        frame_center = (frame_w / 2.0, frame_h / 2.0)

        if self.crop_center is None:
            self.crop_center = frame_center

        if target is not None:
            target_center = (float(target.center[0]), float(target.center[1]))
            target_area_ratio = (target.bbox[2] * target.bbox[3]) / max(frame_w * frame_h, 1)
            desired_zoom = self._desired_zoom(target_area_ratio)
            desired_center = target_center
        else:
            target_center = self.crop_center
            desired_zoom = 1.0
            desired_center = frame_center

        error_x = target_center[0] - frame_center[0]
        error_y = target_center[1] - frame_center[1]
        normalized_error_x = error_x / max(frame_w / 2.0, 1.0)
        normalized_error_y = error_y / max(frame_h / 2.0, 1.0)

        if abs(normalized_error_x) < self.dead_zone:
            desired_center = (self.crop_center[0], desired_center[1])
        if abs(normalized_error_y) < self.dead_zone:
            desired_center = (desired_center[0], self.crop_center[1])

        self.crop_center = (
            self.crop_center[0] + (desired_center[0] - self.crop_center[0]) * self.smoothing,
            self.crop_center[1] + (desired_center[1] - self.crop_center[1]) * self.smoothing,
        )
        self.zoom_ratio += (desired_zoom - self.zoom_ratio) * 0.08
        self.zoom_ratio = max(1.0, min(self.max_zoom, self.zoom_ratio))

        crop_w = int(frame_w / self.zoom_ratio)
        crop_h = int(frame_h / self.zoom_ratio)
        crop_x = int(round(self.crop_center[0] - crop_w / 2))
        crop_y = int(round(self.crop_center[1] - crop_h / 2))
        crop_x = max(0, min(frame_w - crop_w, crop_x))
        crop_y = max(0, min(frame_h - crop_h, crop_y))

        crop = frame[crop_y : crop_y + crop_h, crop_x : crop_x + crop_w]
        output = cv2.resize(crop, (frame_w, frame_h), interpolation=cv2.INTER_LINEAR)
        return CropResult(
            frame=output,
            crop_x=crop_x,
            crop_y=crop_y,
            crop_w=crop_w,
            crop_h=crop_h,
            zoom_ratio=self.zoom_ratio,
            zoom_error=desired_zoom - self.zoom_ratio,
            error_x=error_x,
            error_y=error_y,
            normalized_error_x=normalized_error_x,
            normalized_error_y=normalized_error_y,
        )

    def _desired_zoom(self, target_area_ratio: float) -> float:
        desired_ratio = 0.08
        if target_area_ratio <= 0:
            return 1.0
        zoom = (desired_ratio / target_area_ratio) ** 0.5
        return max(1.0, min(self.max_zoom, zoom))

