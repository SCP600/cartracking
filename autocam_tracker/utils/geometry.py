from __future__ import annotations


def clamp_bbox(bbox: tuple[int, int, int, int], frame_w: int, frame_h: int) -> tuple[int, int, int, int]:
    x, y, w, h = bbox
    x = max(0, min(frame_w - 1, x))
    y = max(0, min(frame_h - 1, y))
    w = max(0, min(frame_w - x, w))
    h = max(0, min(frame_h - y, h))
    return x, y, w, h


def center_distance(a: tuple[int, int], b: tuple[int, int]) -> float:
    dx = float(a[0] - b[0])
    dy = float(a[1] - b[1])
    return (dx * dx + dy * dy) ** 0.5

