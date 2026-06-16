from __future__ import annotations

from tkinter import ttk

import cv2
from PIL import Image, ImageTk

from autocam_tracker.detection.detection_models import RecognizedVehicleSummary


class RecognizedVehicleListPanel(ttk.LabelFrame):
    def __init__(self, master) -> None:
        super().__init__(master, text="Recognized / Anchored")
        self.rows = ttk.Frame(self)
        self.rows.pack(fill="both", expand=True)
        self._image_refs: list[ImageTk.PhotoImage] = []

    def update_vehicles(self, vehicles: list[RecognizedVehicleSummary]) -> None:
        for child in self.rows.winfo_children():
            child.destroy()
        self._image_refs = []

        if not vehicles:
            ttk.Label(self.rows, text="No recognized vehicles").pack(anchor="w", padx=4, pady=4)
            return

        for vehicle in vehicles[:16]:
            row = ttk.Frame(self.rows)
            row.pack(fill="x", padx=4, pady=3)
            image = self._thumbnail_image(vehicle)
            if image is not None:
                self._image_refs.append(image)
                ttk.Label(row, image=image).pack(side="left")
            text = self._label_text(vehicle)
            ttk.Label(row, text=text, width=48).pack(side="left", fill="x", expand=True, padx=6)

    def _thumbnail_image(self, vehicle: RecognizedVehicleSummary) -> ImageTk.PhotoImage | None:
        if vehicle.thumbnail is None:
            return None
        rgb = cv2.cvtColor(vehicle.thumbnail, cv2.COLOR_BGR2RGB)
        return ImageTk.PhotoImage(Image.fromarray(rgb))

    def _label_text(self, vehicle: RecognizedVehicleSummary) -> str:
        prefix = "*" if vehicle.selected else " "
        local_id = vehicle.local_track_id if vehicle.local_track_id >= 0 else "-"
        global_id = vehicle.global_vehicle_id if vehicle.global_vehicle_id >= 0 else "-"
        aliases = ",".join(str(item) for item in vehicle.local_track_aliases[-4:])
        return (
            f"{prefix} {vehicle.registry_id} "
            f"G{global_id} L{local_id} shot{vehicle.shot_id} "
            f"{vehicle.status} seen{vehicle.seen_frames} conf{vehicle.confidence:.2f} "
            f"mem{vehicle.reid_feature_count} match{vehicle.match_score:.2f} aliases[{aliases}]"
        )
