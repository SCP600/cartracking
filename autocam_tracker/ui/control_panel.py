from __future__ import annotations

from pathlib import Path
import tkinter as tk
from tkinter import filedialog, ttk

from autocam_tracker.app.app_state import SourceConfig
from autocam_tracker.ui.screen_region_selector import ScreenRegionSelector


class ControlPanel(ttk.Frame):
    def __init__(
        self,
        master,
        on_start,
        on_toggle_pause,
        on_restart_video,
        on_reset,
        on_reset_cropped,
        on_reset_db,
        on_source_preview=None,
        default_tracker: str = "botsort_reid",
        on_speed_change=None,
    ) -> None:
        super().__init__(master)
        self.on_start = on_start
        self.on_toggle_pause = on_toggle_pause
        self.on_restart_video = on_restart_video
        self.on_reset = on_reset
        self.on_reset_cropped = on_reset_cropped
        self.on_reset_db = on_reset_db
        self.on_source_preview = on_source_preview
        self.source = SourceConfig(kind="webcam", value="0")
        self.source_text = tk.StringVar(value="Webcam 0")
        self.tracker_values = ("botsort", "botsort_reid_default", "botsort_reid_custom", "bytetrack")
        tracker_value = default_tracker if default_tracker in self.tracker_values else "botsort_reid_custom"
        self.tracker = tk.StringVar(value=tracker_value)
        self.on_speed_change = on_speed_change
        self.speed_values = ("0.25x", "0.5x", "1.0x", "1.5x", "2.0x", "2.5x", "3.0x")
        self.speed = tk.StringVar(value="1.0x")
        self._build()

    def _build(self) -> None:
        ttk.Button(self, text="Open Video", command=self._open_file).pack(side="left", padx=4)
        ttk.Button(self, text="Use Webcam", command=self._use_webcam).pack(side="left", padx=4)
        ttk.Button(self, text="Select Region", command=self._select_region).pack(side="left", padx=4)
        ttk.Label(self, textvariable=self.source_text, width=42).pack(side="left", padx=4)
        
        ttk.Label(self, text="Speed:").pack(side="left", padx=(10, 4))
        speed_select = ttk.Combobox(
            self,
            textvariable=self.speed,
            values=self.speed_values,
            width=6,
            state="readonly",
        )
        speed_select.pack(side="left")
        speed_select.current(3)  # default 1.0x
        speed_select.bind("<<ComboboxSelected>>", self._on_speed_select)
        
        ttk.Button(self, text="Help / Controls", command=self._show_help).pack(side="left", padx=4)
        ttk.Button(self, text="Start", command=self._start).pack(side="left", padx=4)
        self.btn_pause = ttk.Button(self, text="Pause", command=self.on_toggle_pause)
        self.btn_pause.pack(side="left", padx=4)
        ttk.Button(self, text="Restart Video", command=self.on_restart_video).pack(side="left", padx=4)
        ttk.Button(self, text="Reset Target", command=self.on_reset).pack(side="left", padx=4)
        ttk.Button(self, text="Reset Cropped", command=self.on_reset_cropped).pack(side="left", padx=4)
        ttk.Button(self, text="Reset DB", command=self.on_reset_db).pack(side="left", padx=4)

    def _open_file(self) -> None:
        path = filedialog.askopenfilename(
            title="Select racing video",
            filetypes=(("Video files", "*.mp4 *.mov *.avi *.mkv"), ("All files", "*.*")),
        )
        if not path:
            return
        self.source = SourceConfig(kind="file", value=path)
        self.source_text.set(Path(path).name)

    def _use_webcam(self) -> None:
        self.source = SourceConfig(kind="webcam", value="0")
        self.source_text.set("Webcam 0")

    def _select_region(self) -> None:
        selector = ScreenRegionSelector(self.winfo_toplevel(), self._set_screen_region)
        selector.open()

    def _set_screen_region(self, region: tuple[int, int, int, int]) -> None:
        left, top, width, height = region
        self.source = SourceConfig(kind="screen", value=f"{left},{top},{width},{height}")
        self.source_text.set(f"Screen {left},{top} {width}x{height}")
        if self.on_source_preview is not None:
            self.on_source_preview(self.source)

    def _start(self) -> None:
        self.on_start(self.source, "botsort_reid_custom")

    def _on_speed_select(self, event) -> None:
        if self.on_speed_change:
            val = self.speed.get().replace("x", "")
            try:
                self.on_speed_change(float(val))
            except ValueError:
                pass

    def _show_help(self) -> None:
        from tkinter import messagebox
        help_text = (
            "🎥 AutoCamTracker Control Methods\n"
            "---------------------------------\n"
            "▶ Playback Controls:\n"
            "- [Spacebar]: Pause / Resume video playback.\n\n"
            "🖱️ Target Binding & Tracking (Raw View):\n"
            "- [Left Click]: Select target (focus camera only).\n"
            "- [Shift + Left Click]: Bind vehicle as a NEW Target (Create GID).\n"
            "- [Ctrl + Left Click]: Add vehicle feature to CURRENT Target.\n"
            "- [Right Click]: Open Context Menu for quick binding options.\n\n"
            "🗂️ Anchor DB Controls (Bottom Panel):\n"
            "- [Click GID]: Force the camera to instantly track that GID."
        )
        messagebox.showinfo("Control Methods & Hotkeys", help_text)

    def update_pause_button(self, is_paused: bool) -> None:
        if is_paused:
            self.btn_pause.config(text="Resume")
        else:
            self.btn_pause.config(text="Pause")
