from __future__ import annotations

import tkinter as tk
from tkinter import ttk

import cv2
from PIL import Image, ImageTk

from autocam_tracker.detection.detection_models import VehicleDetection


class VehicleListPanel(ttk.LabelFrame):
    def __init__(self, master, on_select) -> None:
        super().__init__(master, text="Vehicles")
        self.on_select = on_select
        self.rows = ttk.Frame(self)
        self.rows.pack(fill="both", expand=True)
        self._image_refs: list[ImageTk.PhotoImage] = []

    def update_detections(self, detections: list[VehicleDetection]) -> None:
        for child in self.rows.winfo_children():
            child.destroy()
        self._image_refs = []

        if not detections:
            ttk.Label(self.rows, text="No vehicles").pack(anchor="w", padx=4, pady=4)
            return

        for detection in detections[:12]:
            row = ttk.Frame(self.rows)
            row.pack(fill="x", padx=4, pady=3)
            image = self._thumbnail_image(detection)
            if image is not None:
                self._image_refs.append(image)
                ttk.Label(row, image=image).pack(side="left")
            text = self._label_text(detection)
            button = ttk.Button(
                row,
                text=text,
                command=lambda did=detection.detection_id, lid=detection.local_track_id: self.on_select(did, lid),
            )
            button.pack(side="left", fill="x", expand=True, padx=6)

    def _thumbnail_image(self, detection: VehicleDetection) -> ImageTk.PhotoImage | None:
        if detection.thumbnail is None:
            return None
        rgb = cv2.cvtColor(detection.thumbnail, cv2.COLOR_BGR2RGB)
        return ImageTk.PhotoImage(Image.fromarray(rgb))

    def _label_text(self, detection: VehicleDetection) -> str:
        prefix = "*" if detection.selected else " "
        local_id = detection.local_track_id if detection.local_track_id >= 0 else "-"
        global_id = detection.global_vehicle_id if detection.global_vehicle_id >= 0 else "-"
        reid = f" R{detection.reid_score:.2f}" if detection.reid_score > 0.0 else ""
        return f"{prefix} D{detection.detection_id} L{local_id} G{global_id} {detection.confidence:.2f}{reid}"
