from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class VehicleIdentity:
    global_vehicle_id: int
    created_at_ms: float
    last_seen_ms: float = 0.0

    label: str = "car"
    camera_id: int = 0
    shot_id: int = 0

    last_detection_id: int = -1
    last_local_track_id: int = -1
    last_bbox: tuple[int, int, int, int] = (0, 0, 0, 0)
    last_center: tuple[int, int] = (0, 0)

    thumbnails: list[np.ndarray] = field(default_factory=list)
    color_signature: Optional[np.ndarray] = None

    lost_frames: int = 0
    status: str = "Tracking"

    reid_score: float = 0.0
    reid_matched: bool = False

