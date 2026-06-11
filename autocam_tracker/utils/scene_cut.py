from __future__ import annotations

import cv2
import numpy as np


class SceneCutDetector:
    def __init__(self, threshold: float = 0.62) -> None:
        self.threshold = threshold
        self.previous_hist: np.ndarray | None = None

    def update(self, frame: np.ndarray) -> bool:
        hist = self._histogram(frame)
        if self.previous_hist is None:
            self.previous_hist = hist
            return False
        correlation = cv2.compareHist(self.previous_hist, hist, cv2.HISTCMP_CORREL)
        self.previous_hist = hist
        return correlation < self.threshold

    def _histogram(self, frame: np.ndarray) -> np.ndarray:
        small = cv2.resize(frame, (160, 90), interpolation=cv2.INTER_AREA)
        hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1], None, [32, 16], [0, 180, 0, 256])
        cv2.normalize(hist, hist, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
        return hist.astype("float32")

