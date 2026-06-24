from __future__ import annotations

from pathlib import Path
import tkinter as tk
from tkinter import messagebox, ttk

from autocam_tracker.app.app_controller import AppController
from autocam_tracker.app.app_state import SourceConfig
from autocam_tracker.ui.control_panel import ControlPanel
from autocam_tracker.ui.anchor_db_panel import AnchorDBPanel
from autocam_tracker.ui.live_view_panel import LiveViewPanel
from autocam_tracker.ui.status_panel import StatusPanel
from autocam_tracker.ui.timeline_panel import TimelinePanel
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
        
        # Unbind space from buttons so it exclusively acts as a global play/pause toggle
        self.root.unbind_class("TButton", "<space>")
        self.root.unbind_class("Button", "<space>")
        self.root.bind_all("<space>", lambda event: self._toggle_pause())
        self.root.after(50, self._poll)
        self.root.mainloop()

    def _build(self) -> None:
        self.control_panel = ControlPanel(
            self.root,
            on_start=self._start,
            on_toggle_pause=self._toggle_pause,
            on_restart_video=lambda: self.controller.seek_frame(0),
            on_reset=self.controller.reset_target,
            on_reset_cropped=self.controller.reset_cropped,
            on_reset_db=self.controller.reset_anchor_db,
            on_source_preview=self._preview_source,
            default_tracker=self.controller.config.tracker,
            on_speed_change=self.controller.set_playback_speed,
        )
        self.control_panel.pack(fill="x", padx=8, pady=8)

        main = ttk.Frame(self.root)
        main.pack(fill="both", expand=True, padx=8, pady=4)

        views = ttk.Frame(main)
        views.pack(side="top", fill="both", expand=True)

        self.raw_view = LiveViewPanel(views, "Raw / Detection View", (660, 380), on_click=self._on_raw_view_click, on_right_click=self._on_raw_view_right_click)
        self.raw_view.pack(side="left", fill="both", expand=True, padx=(0, 4))
        self.crop_view = LiveViewPanel(views, "Cropped / Output View", (660, 380))
        self.crop_view.pack(side="left", fill="both", expand=True, padx=(4, 0))

        self.timeline_panel = TimelinePanel(main, on_seek=self.controller.seek_frame)
        self.timeline_panel.pack(side="top", fill="x", pady=(6, 0))

        bottom = ttk.Frame(main)
        bottom.pack(side="top", fill="both", expand=True, pady=(8, 0))

        self.anchor_db_panel = AnchorDBPanel(bottom, on_track_gid=self.controller.track_global_vehicle)
        self.anchor_db_panel.pack(side="left", fill="both", expand=True, padx=(0, 4))

        self.status_panel = StatusPanel(bottom)
        self.status_panel.pack(side="left", fill="y", padx=(4, 0))

    def _start(self, source: SourceConfig, tracker_name: str) -> None:
        try:
            self.controller.start(source, tracker_name)
            self.control_panel.update_pause_button(False)
        except Exception as exc:
            messagebox.showerror("Start failed", str(exc))

    def _toggle_pause(self) -> None:
        is_paused = self.controller.toggle_pause()
        self.control_panel.update_pause_button(is_paused)

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
            self.last_frame_data = frame_data
            self.raw_view.update_frame(frame_data.detection_frame)
            self.crop_view.update_frame(frame_data.cropped_frame)
            if hasattr(frame_data, "anchor_db_state"):
                self.anchor_db_panel.update_db_view(frame_data.anchor_db_state)
            self.status_panel.update_status(frame_data)
            self.timeline_panel.update_timeline(frame_data)
        self.root.after(50, self._poll)

    def _on_raw_view_click(self, event) -> None:
        if not hasattr(self, "last_frame_data") or self.last_frame_data is None:
            return
            
        is_shift = (event.state & 0x0001) != 0
        is_ctrl = (event.state & 0x0004) != 0
        x, y = event.x, event.y
            
        frame_data = self.last_frame_data
        detections = frame_data.detections
        if not detections:
            return
            
        orig_h, orig_w = frame_data.detection_frame.shape[:2]
        
        scale = min(self.raw_view.image_size[0] / orig_w, self.raw_view.image_size[1] / orig_h)
        scaled_w = int(orig_w * scale)
        scaled_h = int(orig_h * scale)
        
        label_w = self.raw_view.label.winfo_width()
        label_h = self.raw_view.label.winfo_height()
        
        offset_x = (label_w - scaled_w) // 2
        offset_y = (label_h - scaled_h) // 2
        
        if offset_x <= x <= offset_x + scaled_w and offset_y <= y <= offset_y + scaled_h:
            orig_x = (x - offset_x) / scale
            orig_y = (y - offset_y) / scale
            
            for d in detections:
                bx, by, bw, bh = d.bbox
                if bx <= orig_x <= bx + bw and by <= orig_y <= by + bh:
                    if is_shift:
                        self.controller.select_global_vehicle(global_vehicle_id=-1, local_track_id=d.local_track_id)
                    elif is_ctrl:
                        gid = self.controller.worker.identity_manager.selected_global_vehicle_id if self.controller.worker else -1
                        if gid == -1:
                            gid = self.anchor_db_panel.get_selected_gid()
                        if gid != -1:
                            self.controller.add_feature_to_gid(gid, d.local_track_id)
                        else:
                            self.controller.select_detection(detection_id=d.detection_id, local_track_id=d.local_track_id)
                    else:
                        self.controller.select_detection(detection_id=d.detection_id, local_track_id=d.local_track_id)
                    break

    def _on_raw_view_right_click(self, event) -> None:
        if not hasattr(self, "last_frame_data") or self.last_frame_data is None:
            return
            
        frame_data = self.last_frame_data
        detections = frame_data.detections
        if not detections:
            return
            
        orig_h, orig_w = frame_data.detection_frame.shape[:2]
        
        scale = min(self.raw_view.image_size[0] / orig_w, self.raw_view.image_size[1] / orig_h)
        scaled_w = int(orig_w * scale)
        scaled_h = int(orig_h * scale)
        
        label_w = self.raw_view.label.winfo_width()
        label_h = self.raw_view.label.winfo_height()
        
        offset_x = (label_w - scaled_w) // 2
        offset_y = (label_h - scaled_h) // 2
        x, y = event.x, event.y
        
        if offset_x <= x <= offset_x + scaled_w and offset_y <= y <= offset_y + scaled_h:
            orig_x = (x - offset_x) / scale
            orig_y = (y - offset_y) / scale
            
            for d in detections:
                bx, by, bw, bh = d.bbox
                if bx <= orig_x <= bx + bw and by <= orig_y <= by + bh:
                    menu = tk.Menu(self.root, tearoff=0)
                    menu.add_command(label="[+] Bind as New Target", command=lambda local_id=d.local_track_id: self.controller.select_global_vehicle(-1, local_id))
                    menu.add_separator()
                    
                    active_gids = []
                    if hasattr(frame_data, "anchor_db_state"):
                        active_gids = list(frame_data.anchor_db_state.keys())
                        
                    for gid in active_gids:
                        menu.add_command(label=f"Add feature to G{gid}", command=lambda gid=gid, local_id=d.local_track_id: self.controller.add_feature_to_gid(gid, local_id))
                        
                    menu.post(event.x_root, event.y_root)
                    break

    def _on_close(self) -> None:
        self.controller.stop()
        self.root.destroy()
