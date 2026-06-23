from __future__ import annotations

class GlobalIdentityManager:
    def __init__(self) -> None:
        self.selected_global_vehicle_id: int = -1
        self.selected_local_track_id: int = -1
        self.status = "Detecting"

    def reset(self) -> None:
        self.selected_global_vehicle_id = -1
        self.selected_local_track_id = -1
        self.status = "Detecting"

    def handle_camera_cut(self, shot_id: int) -> None:
        self.selected_local_track_id = -1
        if self.selected_global_vehicle_id != -1:
            self.status = "CameraCut"

    def set_target(self, global_id: int, local_track_id: int):
        self.selected_global_vehicle_id = global_id
        self.selected_local_track_id = local_track_id
        if self.selected_global_vehicle_id != -1:
            self.status = "Tracking" if local_track_id >= 0 else "SearchingTarget"
        else:
            self.status = "Detecting"
