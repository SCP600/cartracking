from __future__ import annotations

import tkinter as tk
from tkinter import ttk

import numpy as np

from autocam_tracker.utils.image_utils import bgr_to_tk_image


class LiveViewPanel(ttk.Frame):
    def __init__(self, master, title: str, image_size: tuple[int, int]) -> None:
        super().__init__(master)
        self.image_size = image_size
        self._image_ref = None
        ttk.Label(self, text=title).pack(anchor="w")
        self.label = ttk.Label(self)
        self.label.pack(fill="both", expand=True)

    def update_frame(self, frame: np.ndarray | None) -> None:
        if frame is None:
            return
        self._image_ref = bgr_to_tk_image(frame, self.image_size)
        self.label.configure(image=self._image_ref)

