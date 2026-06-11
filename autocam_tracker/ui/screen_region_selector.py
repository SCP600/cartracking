from __future__ import annotations

import tkinter as tk
from typing import Callable


class ScreenRegionSelector:
    def __init__(self, master: tk.Misc, on_selected: Callable[[tuple[int, int, int, int]], None]) -> None:
        self.master = master
        self.on_selected = on_selected
        self.overlay: tk.Toplevel | None = None
        self.canvas: tk.Canvas | None = None
        self.start_x = 0
        self.start_y = 0
        self.rect_id: int | None = None

    def open(self) -> None:
        overlay = tk.Toplevel(self.master)
        overlay.title("Select screen region")
        overlay.overrideredirect(True)
        overlay.attributes("-topmost", True)
        overlay.attributes("-alpha", 0.28)
        overlay.configure(bg="black")

        screen_w = overlay.winfo_screenwidth()
        screen_h = overlay.winfo_screenheight()
        overlay.geometry(f"{screen_w}x{screen_h}+0+0")

        canvas = tk.Canvas(overlay, cursor="crosshair", bg="black", highlightthickness=0)
        canvas.pack(fill="both", expand=True)
        canvas.create_text(
            screen_w // 2,
            40,
            text="Drag to select screen region. Press Esc to cancel.",
            fill="white",
            font=("Segoe UI", 16, "bold"),
        )

        canvas.bind("<ButtonPress-1>", self._on_press)
        canvas.bind("<B1-Motion>", self._on_drag)
        canvas.bind("<ButtonRelease-1>", self._on_release)
        canvas.bind("<ButtonPress-3>", lambda _event: self._cancel())
        overlay.bind("<Escape>", lambda _event: self._cancel())
        overlay.bind_all("<Escape>", lambda _event: self._cancel())
        overlay.protocol("WM_DELETE_WINDOW", self._cancel)
        overlay.grab_set()
        overlay.lift()
        overlay.focus_force()
        canvas.focus_set()

        self.overlay = overlay
        self.canvas = canvas

    def _on_press(self, event: tk.Event) -> None:
        self.start_x = int(event.x)
        self.start_y = int(event.y)
        if self.canvas is None:
            return
        if self.rect_id is not None:
            self.canvas.delete(self.rect_id)
        self.rect_id = self.canvas.create_rectangle(
            self.start_x,
            self.start_y,
            self.start_x,
            self.start_y,
            outline="#00ff66",
            width=3,
        )

    def _on_drag(self, event: tk.Event) -> None:
        if self.canvas is None or self.rect_id is None:
            return
        self.canvas.coords(self.rect_id, self.start_x, self.start_y, event.x, event.y)

    def _on_release(self, event: tk.Event) -> None:
        if self.overlay is None:
            return

        root_x = self.overlay.winfo_rootx()
        root_y = self.overlay.winfo_rooty()
        x1 = min(self.start_x, int(event.x))
        y1 = min(self.start_y, int(event.y))
        x2 = max(self.start_x, int(event.x))
        y2 = max(self.start_y, int(event.y))
        width = x2 - x1
        height = y2 - y1

        if width >= 16 and height >= 16:
            self.on_selected((root_x + x1, root_y + y1, width, height))
        self._destroy()

    def _cancel(self) -> None:
        self._destroy()

    def _destroy(self) -> None:
        if self.overlay is not None:
            try:
                self.overlay.unbind_all("<Escape>")
                self.overlay.grab_release()
            except tk.TclError:
                pass
            self.overlay.destroy()
        self.overlay = None
        self.canvas = None
        self.rect_id = None
