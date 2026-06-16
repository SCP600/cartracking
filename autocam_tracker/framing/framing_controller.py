from __future__ import annotations

import numpy as np

from autocam_tracker.detection.detection_models import FrameData, RecognizedVehicleSummary, VehicleDetection
from autocam_tracker.framing.crop_controller import CropResult


class FramingController:
    def build_frame_data(
        self,
        raw_frame: np.ndarray,
        detection_frame: np.ndarray,
        cropped_frame: np.ndarray,
        detections: list[VehicleDetection],
        camera_id: int,
        shot_id: int,
        frame_index: int,
        total_frame_count: int,
        timestamp_ms: float,
        tracking_status: str,
        selected_global_vehicle_id: int,
        selected_local_track_id: int,
        selected_detection_id: int,
        camera_cut_detected: bool,
        fps: float,
        inference_time_ms: float,
        tracking_time_ms: float,
        reframe_time_ms: float,
        crop_result: CropResult,
        lost_frames: int,
        candidate_count: int,
        reacquire_score: float,
        recognized_vehicles: list[RecognizedVehicleSummary] | None = None,
    ) -> FrameData:
        return FrameData(
            camera_id=camera_id,
            shot_id=shot_id,
            frame_index=frame_index,
            total_frame_count=total_frame_count,
            timestamp_ms=timestamp_ms,
            raw_frame=raw_frame,
            detection_frame=detection_frame,
            cropped_frame=cropped_frame,
            detections=detections,
            selected_global_vehicle_id=selected_global_vehicle_id,
            selected_local_track_id=selected_local_track_id,
            selected_detection_id=selected_detection_id,
            tracking_status=tracking_status,
            camera_cut_detected=camera_cut_detected,
            fps=fps,
            inference_time_ms=inference_time_ms,
            tracking_time_ms=tracking_time_ms,
            reframe_time_ms=reframe_time_ms,
            error_x=crop_result.error_x,
            error_y=crop_result.error_y,
            normalized_error_x=crop_result.normalized_error_x,
            normalized_error_y=crop_result.normalized_error_y,
            crop_x=crop_result.crop_x,
            crop_y=crop_result.crop_y,
            crop_w=crop_result.crop_w,
            crop_h=crop_result.crop_h,
            zoom_ratio=crop_result.zoom_ratio,
            zoom_error=crop_result.zoom_error,
            lost_frames=lost_frames,
            candidate_count=candidate_count,
            reacquire_score=reacquire_score,
            recognized_vehicles=recognized_vehicles or [],
        )
