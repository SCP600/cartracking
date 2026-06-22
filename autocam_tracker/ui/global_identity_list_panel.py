from __future__ import annotations

import tkinter as tk
from tkinter import ttk

import cv2
from PIL import Image, ImageTk

from autocam_tracker.detection.detection_models import RecognizedVehicleSummary


class GlobalIdentityListPanel(ttk.LabelFrame):
    def __init__(self, master, on_select) -> None:
        super().__init__(master, text="GID Anchors")
        self.on_select = on_select

        self.canvas = tk.Canvas(self, highlightthickness=0)
        self.scrollbar = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.rows = ttk.Frame(self.canvas)

        self.rows.bind(
            "<Configure>",
            lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        )

        self.canvas_window = self.canvas.create_window((0, 0), window=self.rows, anchor="nw")
        self.canvas.bind("<Configure>", self._on_canvas_configure)
        self.canvas.configure(yscrollcommand=self.scrollbar.set)

        self.canvas.pack(side="left", fill="both", expand=True)
        self.scrollbar.pack(side="right", fill="y")

        self.canvas.bind("<Enter>", self._bind_mousewheel)
        self.canvas.bind("<Leave>", self._unbind_mousewheel)

        self._image_refs: list[ImageTk.PhotoImage] = []

    def _on_canvas_configure(self, event) -> None:
        self.canvas.itemconfig(self.canvas_window, width=event.width)

    def _bind_mousewheel(self, event) -> None:
        self.canvas.bind_all("<MouseWheel>", self._on_mousewheel)

    def _unbind_mousewheel(self, event) -> None:
        self.canvas.unbind_all("<MouseWheel>")

    def _on_mousewheel(self, event) -> None:
        self.canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

    def update_vehicles(self, vehicles: list[RecognizedVehicleSummary]) -> None:
        scroll_pos = self.canvas.yview()

        for child in self.rows.winfo_children():
            child.destroy()
        self._image_refs = []

        gid_vehicles = [vehicle for vehicle in vehicles if vehicle.global_vehicle_id >= 0]
        if not gid_vehicles:
            ttk.Label(self.rows, text="No GID vehicles").pack(anchor="w", padx=4, pady=4)
            self.canvas.yview_moveto(0.0)
            return

        for vehicle in gid_vehicles:
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

        self.rows.update_idletasks()
        self.canvas.yview_moveto(scroll_pos[0])

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
