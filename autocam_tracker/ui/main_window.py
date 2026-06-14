from __future__ import annotations

from pathlib import Path
import tkinter as tk
from tkinter import messagebox, ttk

from autocam_tracker.app.app_controller import AppController
from autocam_tracker.app.app_state import SourceConfig
from autocam_tracker.ui.control_panel import ControlPanel
from autocam_tracker.ui.live_view_panel import LiveViewPanel
from autocam_tracker.ui.recognized_vehicle_list_panel import RecognizedVehicleListPanel
from autocam_tracker.ui.status_panel import StatusPanel
from autocam_tracker.ui.vehicle_list_panel import VehicleListPanel
from autocam_tracker.video.screen_region_source import capture_screen_region_once


class MainWindow:
    def __init__(self) -> None:
        self.root = tk.Tk()
        self.root.title("AutoCamTracker V1")
        self.root.geometry("1380x860")
        self.project_root = Path(__file__).resolve().parents[2]
        self.controller = AppController(self.project_root)
        self._build()

    def run(self) -> None:
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self.root.after(50, self._poll)
        self.root.mainloop()

    def _build(self) -> None:
        self.control_panel = ControlPanel(
            self.root,
            on_start=self._start,
            on_stop=self.controller.stop,
            on_reset=self.controller.reset_target,
            on_source_preview=self._preview_source,
            default_tracker=self.controller.config.tracker,
        )
        self.control_panel.pack(fill="x", padx=8, pady=8)

        main = ttk.Frame(self.root)
        main.pack(fill="both", expand=True, padx=8, pady=4)

        views = ttk.Frame(main)
        views.pack(side="top", fill="both", expand=True)

        self.raw_view = LiveViewPanel(views, "Raw / Detection View", (660, 380))
        self.raw_view.pack(side="left", fill="both", expand=True, padx=(0, 4))
        self.crop_view = LiveViewPanel(views, "Cropped / Output View", (660, 380))
        self.crop_view.pack(side="left", fill="both", expand=True, padx=(4, 0))

        bottom = ttk.Frame(main)
        bottom.pack(side="top", fill="both", expand=True, pady=(8, 0))

        lists = ttk.Notebook(bottom)
        lists.pack(side="left", fill="both", expand=True, padx=(0, 4))

        self.vehicle_list = VehicleListPanel(lists, on_select=self.controller.select_detection)
        self.recognized_list = RecognizedVehicleListPanel(lists)
        lists.add(self.vehicle_list, text="Current Detections")
        lists.add(self.recognized_list, text="Recognized")

        self.status_panel = StatusPanel(bottom)
        self.status_panel.pack(side="left", fill="y", padx=(4, 0))

    def _start(self, source: SourceConfig, tracker_name: str) -> None:
        try:
            self.controller.start(source, tracker_name)
        except Exception as exc:
            messagebox.showerror("Start failed", str(exc))

    def _preview_source(self, source: SourceConfig) -> None:
        if source.kind != "screen":
            return
        try:
            left, top, width, height = self.controller._parse_screen_region(source.value)
            frame = capture_screen_region_once(left, top, width, height)
        except Exception as exc:
            messagebox.showerror("Preview failed", str(exc))
            return
        self.raw_view.update_frame(frame)
        self.crop_view.update_frame(frame)

    def _poll(self) -> None:
        frame_data = self.controller.poll_frame()
        if frame_data is not None:
            self.raw_view.update_frame(frame_data.detection_frame)
            self.crop_view.update_frame(frame_data.cropped_frame)
            self.vehicle_list.update_detections(frame_data.detections)
            self.recognized_list.update_vehicles(frame_data.recognized_vehicles)
            self.status_panel.update_status(frame_data)
        self.root.after(50, self._poll)

    def _on_close(self) -> None:
        self.controller.stop()
        self.root.destroy()
