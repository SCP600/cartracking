from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class VehicleDetection:
    detection_id: int = -1
    local_track_id: int = -1
    global_vehicle_id: int = -1

    camera_id: int = 0
    shot_id: int = 0
    frame_index: int = 0
    total_frame_count: int = 0
    timestamp_ms: float = 0.0

    label: str = "car"
    confidence: float = 0.0

    bbox: tuple[int, int, int, int] = (0, 0, 0, 0)
    center: tuple[int, int] = (0, 0)
    thumbnail: Optional[np.ndarray] = None

    selected: bool = False
    active: bool = True
    lost: bool = False

    color_signature: Optional[np.ndarray] = None
    appearance_score: float = 0.0

    reid_score: float = 0.0
    reid_matched: bool = False


@dataclass
class FrameData:
    camera_id: int = 0
    shot_id: int = 0
    frame_index: int = 0
    timestamp_ms: float = 0.0

    raw_frame: Optional[np.ndarray] = None
    detection_frame: Optional[np.ndarray] = None
    cropped_frame: Optional[np.ndarray] = None

    detections: list[VehicleDetection] = field(default_factory=list)

    selected_global_vehicle_id: int = -1
    selected_local_track_id: int = -1
    selected_detection_id: int = -1

    tracking_status: str = "Idle"
    camera_cut_detected: bool = False

    fps: float = 0.0
    inference_time_ms: float = 0.0
    tracking_time_ms: float = 0.0
    reframe_time_ms: float = 0.0

    error_x: float = 0.0
    error_y: float = 0.0
    normalized_error_x: float = 0.0
    normalized_error_y: float = 0.0

    crop_x: int = 0
    crop_y: int = 0
    crop_w: int = 0
    crop_h: int = 0
    zoom_ratio: float = 1.0
    zoom_error: float = 0.0

    lost_frames: int = 0
    candidate_count: int = 0
    reacquire_score: float = 0.0

    recognized_vehicles: list["RecognizedVehicleSummary"] = field(default_factory=list)


@dataclass
class RecognizedVehicleSummary:
    registry_id: str
    local_track_id: int = -1
    global_vehicle_id: int = -1
    camera_id: int = 0
    shot_id: int = 0
    first_frame_index: int = 0
    last_frame_index: int = 0
    last_seen_ms: float = 0.0
    label: str = "car"
    confidence: float = 0.0
    bbox: tuple[int, int, int, int] = (0, 0, 0, 0)
    seen_frames: int = 0
    selected: bool = False
    status: str = "Seen"
    thumbnail: Optional[np.ndarray] = None
    color_signature: Optional[np.ndarray] = None
    local_track_aliases: list[int] = field(default_factory=list)
    match_score: float = 0.0
