from __future__ import annotations

from pathlib import Path
import tkinter as tk
from tkinter import filedialog, ttk

from autocam_tracker.app.app_state import SourceConfig
from autocam_tracker.ui.screen_region_selector import ScreenRegionSelector


class ControlPanel(ttk.Frame):
    def __init__(self, master, on_start, on_stop, on_reset, on_source_preview=None, default_tracker: str = "botsort_reid") -> None:
        super().__init__(master)
        self.on_start = on_start
        self.on_stop = on_stop
        self.on_reset = on_reset
        self.on_source_preview = on_source_preview
        self.source = SourceConfig(kind="webcam", value="0")
        self.source_text = tk.StringVar(value="Webcam 0")
        self.tracker_values = ("botsort", "botsort_reid_default", "botsort_reid_custom", "bytetrack")
        tracker_value = default_tracker if default_tracker in self.tracker_values else "botsort_reid_custom"
        self.tracker = tk.StringVar(value=tracker_value)
        self._build()

    def _build(self) -> None:
        ttk.Button(self, text="Open Video", command=self._open_file).pack(side="left", padx=4)
        ttk.Button(self, text="Use Webcam", command=self._use_webcam).pack(side="left", padx=4)
        ttk.Button(self, text="Select Region", command=self._select_region).pack(side="left", padx=4)
        ttk.Label(self, textvariable=self.source_text, width=42).pack(side="left", padx=4)
        ttk.Label(self, text="Tracker").pack(side="left", padx=(10, 4))
        tracker_select = ttk.Combobox(
            self,
            textvariable=self.tracker,
            values=self.tracker_values,
            width=22,
            state="readonly",
        )
        tracker_select.pack(side="left")
        ttk.Button(self, text="Start", command=self._start).pack(side="left", padx=4)
        ttk.Button(self, text="Stop", command=self.on_stop).pack(side="left", padx=4)
        ttk.Button(self, text="Reset Target", command=self.on_reset).pack(side="left", padx=4)

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
        self.on_start(self.source, self.tracker.get())
