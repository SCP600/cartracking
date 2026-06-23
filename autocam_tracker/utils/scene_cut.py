from __future__ import annotations

import cv2
import numpy as np


class SceneCutDetector:
    def __init__(self, threshold: float = 0.62, recognition_threshold: float = 0.85) -> None:
        self.threshold = threshold
        self.recognition_threshold = recognition_threshold
        self.previous_hist: np.ndarray | None = None
        self.scene_db: list[tuple[int, np.ndarray]] = []
        self.current_shot_id: int = 1

    def update(self, frame: np.ndarray) -> tuple[bool, int]:
        hist = self._histogram(frame)
        if self.previous_hist is None:
            self.previous_hist = hist
            self.scene_db.append((self.current_shot_id, hist))
            return False, self.current_shot_id
            
        correlation = cv2.compareHist(self.previous_hist, hist, cv2.HISTCMP_CORREL)
        self.previous_hist = hist
        
        is_cut = correlation < self.threshold
        if is_cut:
            best_match_score = -1.0
            best_shot_id = -1
            for shot_id, saved_hist in self.scene_db:
                score = cv2.compareHist(saved_hist, hist, cv2.HISTCMP_CORREL)
                if score > best_match_score:
                    best_match_score = score
                    best_shot_id = shot_id
                    
            if best_match_score > self.recognition_threshold:
                self.current_shot_id = best_shot_id
            else:
                max_id = max([s[0] for s in self.scene_db]) if self.scene_db else 0
                self.current_shot_id = max_id + 1
                self.scene_db.append((self.current_shot_id, hist))
                
        return is_cut, self.current_shot_id

    def _histogram(self, frame: np.ndarray) -> np.ndarray:
        small = cv2.resize(frame, (160, 90), interpolation=cv2.INTER_AREA)
        hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1], None, [32, 16], [0, 180, 0, 256])
        cv2.normalize(hist, hist, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
        return hist.astype("float32")

