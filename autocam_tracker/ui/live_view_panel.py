from __future__ import annotations

import tkinter as tk
from tkinter import ttk

import numpy as np

from autocam_tracker.utils.image_utils import bgr_to_tk_image


class LiveViewPanel(ttk.Frame):
    def __init__(self, master, title: str, image_size: tuple[int, int], on_click=None, on_right_click=None) -> None:
        super().__init__(master)
        self.image_size = image_size
        self._image_ref = None
        self.on_click = on_click
        self.on_right_click = on_right_click
        ttk.Label(self, text=title).pack(anchor="w")
        self.label = ttk.Label(self)
        self.label.pack(fill="both", expand=True)
        if self.on_click:
            self.label.bind("<Button-1>", self._handle_click)
        if self.on_right_click:
            self.label.bind("<Button-3>", self._handle_right_click)

    def _handle_click(self, event):
        if self.on_click:
            self.on_click(event)

    def _handle_right_click(self, event):
        if self.on_right_click:
            self.on_right_click(event)

    def update_frame(self, frame: np.ndarray | None) -> None:
        if frame is None:
            return
        self._image_ref = bgr_to_tk_image(frame, self.image_size)
        self.label.configure(image=self._image_ref)

