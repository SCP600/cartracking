from __future__ import annotations

import tkinter as tk
from tkinter import ttk

from autocam_tracker.detection.detection_models import FrameData


class TimelinePanel(ttk.Frame):
    def __init__(self, master, on_seek) -> None:
        super().__init__(master)
        self.on_seek = on_seek
        self._dragging = False
        self._max_frame = 0
        self.position = tk.DoubleVar(value=0)
        self.label_text = tk.StringVar(value="Timeline unavailable")
        self._build()

    def _build(self) -> None:
        ttk.Label(self, text="Timeline").pack(side="left", padx=(0, 6))
        self.scale = ttk.Scale(self, from_=0, to=0, variable=self.position)
        self.scale.pack(side="left", fill="x", expand=True)
        self.scale.state(["disabled"])
        self.scale.bind("<ButtonPress-1>", self._on_press)
        self.scale.bind("<ButtonRelease-1>", self._on_release)
        ttk.Label(self, textvariable=self.label_text, width=28).pack(side="left", padx=(8, 0))

    def update_timeline(self, frame_data: FrameData) -> None:
        total = int(frame_data.total_frame_count or 0)
        if total <= 0:
            self._max_frame = 0
            self.scale.configure(to=0)
            self.scale.state(["disabled"])
            self.label_text.set("Timeline unavailable")
            return

        max_frame = max(0, total - 1)
        if max_frame != self._max_frame:
            self._max_frame = max_frame
            self.scale.configure(to=max_frame)
        self.scale.state(["!disabled"])

        current = max(0, min(int(frame_data.frame_index), max_frame))
        if not self._dragging:
            self.position.set(current)
        display_current = int(round(self.position.get())) if self._dragging else current
        self.label_text.set(f"Frame {display_current} / {max_frame}")

    def _on_press(self, _event) -> None:
        if self._max_frame > 0:
            self._dragging = True

    def _on_release(self, _event) -> None:
        if not self._dragging:
            return
        self._dragging = False
        frame_index = max(0, min(int(round(self.position.get())), self._max_frame))
        self.position.set(frame_index)
        self.on_seek(frame_index)
