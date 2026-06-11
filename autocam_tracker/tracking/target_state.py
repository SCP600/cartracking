from __future__ import annotations

from dataclasses import dataclass


@dataclass
class TargetState:
    selected_global_vehicle_id: int = -1
    selected_local_track_id: int = -1
    status: str = "Idle"
    lost_frames: int = 0

