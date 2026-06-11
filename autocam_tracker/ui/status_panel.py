from __future__ import annotations

import tkinter as tk
from tkinter import ttk

from autocam_tracker.detection.detection_models import FrameData


class StatusPanel(ttk.LabelFrame):
    def __init__(self, master) -> None:
        super().__init__(master, text="Status")
        self.labels: dict[str, tk.StringVar] = {}
        keys = [
            "tracking_status",
            "selected_global_vehicle_id",
            "selected_local_track_id",
            "fps",
            "detections",
            "shot_id",
            "lost_frames",
            "reacquire_score",
            "crop",
            "zoom_ratio",
            "error",
        ]
        for row, key in enumerate(keys):
            ttk.Label(self, text=key).grid(row=row, column=0, sticky="w", padx=4, pady=2)
            var = tk.StringVar(value="-")
            self.labels[key] = var
            ttk.Label(self, textvariable=var, width=28).grid(row=row, column=1, sticky="w", padx=4, pady=2)

    def update_status(self, frame_data: FrameData) -> None:
        self.labels["tracking_status"].set(frame_data.tracking_status)
        self.labels["selected_global_vehicle_id"].set(str(frame_data.selected_global_vehicle_id))
        self.labels["selected_local_track_id"].set(str(frame_data.selected_local_track_id))
        self.labels["fps"].set(f"{frame_data.fps:.1f}")
        self.labels["detections"].set(str(len(frame_data.detections)))
        self.labels["shot_id"].set(str(frame_data.shot_id))
        self.labels["lost_frames"].set(str(frame_data.lost_frames))
        self.labels["reacquire_score"].set(f"{frame_data.reacquire_score:.2f}")
        self.labels["crop"].set(f"{frame_data.crop_x},{frame_data.crop_y} {frame_data.crop_w}x{frame_data.crop_h}")
        self.labels["zoom_ratio"].set(f"{frame_data.zoom_ratio:.2f}")
        self.labels["error"].set(f"{frame_data.error_x:.0f}, {frame_data.error_y:.0f}")

