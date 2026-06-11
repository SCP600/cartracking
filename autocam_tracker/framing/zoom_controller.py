from __future__ import annotations


class ZoomController:
    def clamp(self, zoom_ratio: float, min_zoom: float = 1.0, max_zoom: float = 2.5) -> float:
        return max(min_zoom, min(max_zoom, zoom_ratio))

