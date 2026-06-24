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
    def __init__(
        self,
        max_zoom: float = 2.5,
        smoothing: float = 0.18,
        dead_zone: float = 0.04,
        max_center_speed: float = 0.02,
        max_center_acceleration: float = 0.0025,
        max_zoom_speed: float = 0.012,
        lost_zoom_hold_frames: int = 45,
        lost_motion_decay: float = 0.94,
    ) -> None:
        self.max_zoom = max(1.0, max_zoom)
        self.smoothing = smoothing
        self.dead_zone = dead_zone
        self.max_center_speed = max(0.0, max_center_speed)
        self.max_center_acceleration = max(0.0, max_center_acceleration)
        self.max_zoom_speed = max(0.0, max_zoom_speed)
        self.lost_zoom_hold_frames = max(0, lost_zoom_hold_frames)
        self.lost_motion_decay = min(max(lost_motion_decay, 0.0), 1.0)
        self.crop_center: tuple[float, float] | None = None
        self.zoom_ratio = 1.0
        self.center_velocity = (0.0, 0.0)
        self.zoom_velocity = 0.0
        self.target_velocity = (0.0, 0.0)
        self.target_acceleration = (0.0, 0.0)
        self.last_target_center: tuple[float, float] | None = None
        self.target_key: int | None = None
        self.lost_frames = 0

    def reset(self) -> None:
        self.crop_center = None
        self.zoom_ratio = 1.0
        self.reset_motion()

    def reset_motion(self) -> None:
        self.center_velocity = (0.0, 0.0)
        self.zoom_velocity = 0.0
        self.target_velocity = (0.0, 0.0)
        self.target_acceleration = (0.0, 0.0)
        self.last_target_center = None
        self.target_key = None
        self.lost_frames = 0

    def crop(self, frame: np.ndarray, target: VehicleDetection | None) -> CropResult:
        frame_h, frame_w = frame.shape[:2]
        frame_center = (frame_w / 2.0, frame_h / 2.0)

        if self.crop_center is None:
            self.crop_center = frame_center

        if target is not None:
            target_center = (float(target.center[0]), float(target.center[1]))
            self._observe_target(target, target_center)
            target_area_ratio = (target.bbox[2] * target.bbox[3]) / max(frame_w * frame_h, 1)
            desired_zoom = self._desired_zoom(target_area_ratio)
            desired_center = (
                target_center[0] + self.target_velocity[0] * 2.0 + self.target_acceleration[0] * 2.0,
                target_center[1] + self.target_velocity[1] * 2.0 + self.target_acceleration[1] * 2.0,
            )
            self._move_toward(desired_center, frame_w, frame_h)
        else:
            target_center = self.crop_center
            self.lost_frames += 1
            self._coast(frame_w, frame_h)
            if self.lost_frames <= self.lost_zoom_hold_frames:
                self.zoom_velocity *= 0.5
                desired_zoom = self.zoom_ratio
            else:
                desired_zoom = 1.0

        error_x = target_center[0] - frame_center[0]
        error_y = target_center[1] - frame_center[1]
        normalized_error_x = error_x / max(frame_w / 2.0, 1.0)
        normalized_error_y = error_y / max(frame_h / 2.0, 1.0)

        self._update_zoom(desired_zoom)

        crop_w = int(frame_w / self.zoom_ratio)
        crop_h = int(frame_h / self.zoom_ratio)
        self._clamp_crop_center(frame_w, frame_h, crop_w, crop_h)
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

    def _observe_target(self, target: VehicleDetection, target_center: tuple[float, float]) -> None:
        target_key = target.local_track_id if target.local_track_id >= 0 else target.detection_id
        gap_frames = self.lost_frames + 1
        self.lost_frames = 0

        if self.target_key != target_key or self.last_target_center is None:
            self.target_key = target_key
            self.last_target_center = target_center
            self.target_velocity = (0.0, 0.0)
            self.target_acceleration = (0.0, 0.0)
            self.center_velocity = (0.0, 0.0)
            return

        measured_velocity = (
            (target_center[0] - self.last_target_center[0]) / gap_frames,
            (target_center[1] - self.last_target_center[1]) / gap_frames,
        )
        previous_velocity = self.target_velocity
        velocity_smoothing = 0.35
        self.target_velocity = (
            previous_velocity[0] + (measured_velocity[0] - previous_velocity[0]) * velocity_smoothing,
            previous_velocity[1] + (measured_velocity[1] - previous_velocity[1]) * velocity_smoothing,
        )
        measured_acceleration = (
            (self.target_velocity[0] - previous_velocity[0]) / gap_frames,
            (self.target_velocity[1] - previous_velocity[1]) / gap_frames,
        )
        acceleration_smoothing = 0.25
        self.target_acceleration = (
            self.target_acceleration[0]
            + (measured_acceleration[0] - self.target_acceleration[0]) * acceleration_smoothing,
            self.target_acceleration[1]
            + (measured_acceleration[1] - self.target_acceleration[1]) * acceleration_smoothing,
        )
        self.last_target_center = target_center

    def _move_toward(self, desired_center: tuple[float, float], frame_w: int, frame_h: int) -> None:
        if self.crop_center is None:
            return

        error_x = desired_center[0] - self.crop_center[0]
        error_y = desired_center[1] - self.crop_center[1]
        if abs(error_x) / max(frame_w / 2.0, 1.0) < self.dead_zone:
            error_x = 0.0
        if abs(error_y) / max(frame_h / 2.0, 1.0) < self.dead_zone:
            error_y = 0.0

        desired_velocity = (
            self.target_velocity[0] + self.target_acceleration[0] * 2.0 + error_x * self.smoothing,
            self.target_velocity[1] + self.target_acceleration[1] * 2.0 + error_y * self.smoothing,
        )
        max_speed_x = frame_w * self.max_center_speed
        max_speed_y = frame_h * self.max_center_speed
        desired_velocity = (
            self._clamp(desired_velocity[0], -max_speed_x, max_speed_x),
            self._clamp(desired_velocity[1], -max_speed_y, max_speed_y),
        )

        max_acceleration_x = frame_w * self.max_center_acceleration
        max_acceleration_y = frame_h * self.max_center_acceleration
        velocity_response = 0.35
        velocity_step = (
            self._clamp(
                (desired_velocity[0] - self.center_velocity[0]) * velocity_response,
                -max_acceleration_x,
                max_acceleration_x,
            ),
            self._clamp(
                (desired_velocity[1] - self.center_velocity[1]) * velocity_response,
                -max_acceleration_y,
                max_acceleration_y,
            ),
        )
        self.center_velocity = (
            self.center_velocity[0] + velocity_step[0],
            self.center_velocity[1] + velocity_step[1],
        )
        self.crop_center = (
            self.crop_center[0] + self.center_velocity[0],
            self.crop_center[1] + self.center_velocity[1],
        )

    def _coast(self, frame_w: int, frame_h: int) -> None:
        if self.crop_center is None:
            return

        self.center_velocity = (
            (self.center_velocity[0] + self.target_acceleration[0]) * self.lost_motion_decay,
            (self.center_velocity[1] + self.target_acceleration[1]) * self.lost_motion_decay,
        )
        self.target_velocity = (
            (self.target_velocity[0] + self.target_acceleration[0]) * self.lost_motion_decay,
            (self.target_velocity[1] + self.target_acceleration[1]) * self.lost_motion_decay,
        )
        self.target_acceleration = (
            self.target_acceleration[0] * 0.85,
            self.target_acceleration[1] * 0.85,
        )

        max_speed_x = frame_w * self.max_center_speed
        max_speed_y = frame_h * self.max_center_speed
        self.center_velocity = (
            self._clamp(self.center_velocity[0], -max_speed_x, max_speed_x),
            self._clamp(self.center_velocity[1], -max_speed_y, max_speed_y),
        )
        self.crop_center = (
            self.crop_center[0] + self.center_velocity[0],
            self.crop_center[1] + self.center_velocity[1],
        )

    def _update_zoom(self, desired_zoom: float) -> None:
        desired_zoom = self._clamp(desired_zoom, 1.0, self.max_zoom)
        desired_velocity = self._clamp(
            (desired_zoom - self.zoom_ratio) * 0.08,
            -self.max_zoom_speed,
            self.max_zoom_speed,
        )
        self.zoom_velocity += (desired_velocity - self.zoom_velocity) * 0.25

        previous_zoom = self.zoom_ratio
        self.zoom_ratio = self._clamp(self.zoom_ratio + self.zoom_velocity, 1.0, self.max_zoom)
        if self.zoom_ratio in (1.0, self.max_zoom):
            self.zoom_velocity = 0.0
        elif (desired_zoom - previous_zoom) * (desired_zoom - self.zoom_ratio) < 0:
            self.zoom_ratio = desired_zoom
            self.zoom_velocity = 0.0

    def _clamp_crop_center(self, frame_w: int, frame_h: int, crop_w: int, crop_h: int) -> None:
        if self.crop_center is None:
            return

        min_x = crop_w / 2.0
        max_x = frame_w - min_x
        min_y = crop_h / 2.0
        max_y = frame_h - min_y
        center_x = self._clamp(self.crop_center[0], min_x, max_x)
        center_y = self._clamp(self.crop_center[1], min_y, max_y)

        velocity_x, velocity_y = self.center_velocity
        if center_x != self.crop_center[0]:
            velocity_x = 0.0
        if center_y != self.crop_center[1]:
            velocity_y = 0.0
        self.crop_center = (center_x, center_y)
        self.center_velocity = (velocity_x, velocity_y)

    def _desired_zoom(self, target_area_ratio: float) -> float:
        desired_ratio = 0.15  # Aim for 1/8 to 1/4 of the view
        if target_area_ratio <= 0:
            return 1.0
        zoom = (desired_ratio / target_area_ratio) ** 0.5
        return max(1.0, min(self.max_zoom, zoom))

    @staticmethod
    def _clamp(value: float, minimum: float, maximum: float) -> float:
        return max(minimum, min(maximum, value))

