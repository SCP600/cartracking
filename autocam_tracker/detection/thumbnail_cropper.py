from __future__ import annotations

import cv2
import numpy as np

from autocam_tracker.utils.geometry import clamp_bbox


class ThumbnailCropper:
    def __init__(self, size: tuple[int, int] = (96, 54)) -> None:
        self.size = size

    def crop(self, frame: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray | None:
        x, y, w, h = clamp_bbox(bbox, frame.shape[1], frame.shape[0])
        if w <= 1 or h <= 1:
            return None
        crop = frame[y : y + h, x : x + w]
        return cv2.resize(crop, self.size, interpolation=cv2.INTER_AREA)

