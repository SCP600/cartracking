from __future__ import annotations

from tkinter import ttk

import cv2
from PIL import Image, ImageTk

from autocam_tracker.detection.detection_models import RecognizedVehicleSummary


class GlobalIdentityListPanel(ttk.LabelFrame):
    def __init__(self, master, on_select) -> None:
        super().__init__(master, text="GID Anchors")
        self.on_select = on_select
        self.rows = ttk.Frame(self)
        self.rows.pack(fill="both", expand=True)
        self._image_refs: list[ImageTk.PhotoImage] = []

    def update_vehicles(self, vehicles: list[RecognizedVehicleSummary]) -> None:
        for child in self.rows.winfo_children():
            child.destroy()
        self._image_refs = []

        gid_vehicles = [vehicle for vehicle in vehicles if vehicle.global_vehicle_id >= 0]
        if not gid_vehicles:
            ttk.Label(self.rows, text="No GID vehicles").pack(anchor="w", padx=4, pady=4)
            return

        for vehicle in gid_vehicles[:20]:
            row = ttk.Frame(self.rows)
            row.pack(fill="x", padx=4, pady=3)
            image = self._thumbnail_image(vehicle)
            if image is not None:
                self._image_refs.append(image)
                ttk.Label(row, image=image).pack(side="left")

            button = ttk.Button(
                row,
                text=self._label_text(vehicle),
                command=lambda gid=vehicle.global_vehicle_id, lid=vehicle.local_track_id: self.on_select(gid, lid),
            )
            button.pack(side="left", fill="x", expand=True, padx=6)

    def _thumbnail_image(self, vehicle: RecognizedVehicleSummary) -> ImageTk.PhotoImage | None:
        if vehicle.thumbnail is None:
            return None
        rgb = cv2.cvtColor(vehicle.thumbnail, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb)
        image.thumbnail((96, 54), Image.Resampling.LANCZOS)
        return ImageTk.PhotoImage(image)

    def _label_text(self, vehicle: RecognizedVehicleSummary) -> str:
        prefix = "*" if vehicle.selected else " "
        local_id = vehicle.local_track_id if vehicle.local_track_id >= 0 else "-"
        aliases = ",".join(str(item) for item in vehicle.local_track_aliases[-3:])
        return (
            f"{prefix} G{vehicle.global_vehicle_id}  L{local_id}  "
            f"{vehicle.status}  seen{vehicle.seen_frames}  "
            f"conf{vehicle.confidence:.2f}  mem{vehicle.reid_feature_count}  "
            f"match{vehicle.match_score:.2f}  aliases[{aliases}]"
        )
