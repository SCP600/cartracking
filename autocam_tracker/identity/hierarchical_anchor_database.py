from __future__ import annotations

import numpy as np


class HierarchicalAnchorDatabase:
    def __init__(self, hysteresis_margin: float = 0.05):
        # Tree structure: global_id -> {"label": str, "shots": {shot_id: list[np.ndarray]}}
        self.db: dict[int, dict] = {}
        self.hysteresis_margin = hysteresis_margin

    def add_anchor(self, global_id: int, shot_id: int, feature: np.ndarray, label: str = ""):
        if global_id not in self.db:
            self.db[global_id] = {"label": label or f"Target {global_id}", "shots": {}}
        if shot_id not in self.db[global_id]["shots"]:
            self.db[global_id]["shots"][shot_id] = []
        self.db[global_id]["shots"][shot_id].append(feature)

    def clear(self) -> None:
        self.db.clear()

    def has_anchors_for_shot(self, global_id: int, shot_id: int) -> bool:
        if global_id not in self.db:
            return False
        return len(self.db[global_id]["shots"].get(shot_id, [])) > 0

    def match_candidates(
        self,
        shot_id: int,
        candidate_features: list[np.ndarray],
        target_global_id: int,
        current_local_track_id: int,
        local_track_ids: list[int],
    ) -> int:
        """
        Returns the best matching local_track_id for the target_global_id in the given shot.
        If no one passes, or no anchors exist, returns -1.
        Uses hysteresis_margin to prevent flickering.
        """
        if not candidate_features or not local_track_ids:
            return -1

        if target_global_id not in self.db:
            return -1

        anchors = self.db[target_global_id]["shots"].get(shot_id, [])
        if not anchors:
            # Fallback: if no anchors for current shot, gather all anchors from past shots
            anchors = []
            for s_anchors in self.db[target_global_id]["shots"].values():
                anchors.extend(s_anchors)
                
        if not anchors:
            return -1

        best_score = -1.0
        best_track_id = -1
        current_score = -1.0

        for feat, track_id in zip(candidate_features, local_track_ids):
            if feat is None:
                continue
            
            # Score against all anchors, take the max
            score = max(float(np.dot(feat, a.T)) for a in anchors)

            if track_id == current_local_track_id:
                current_score = score

            if score > best_score:
                best_score = score
                best_track_id = track_id

        if best_track_id == -1:
            return current_local_track_id

        # Hysteresis Transfer
        if current_local_track_id != -1 and current_score != -1.0:
            if best_track_id != current_local_track_id:
                if best_score > current_score + self.hysteresis_margin:
                    return best_track_id
                else:
                    return current_local_track_id

        return best_track_id

    def get_state(self) -> dict:
        state = {}
        for gid, info in self.db.items():
            shot_counts = {shot_id: len(features) for shot_id, features in info["shots"].items()}
            state[gid] = {
                "label": info["label"],
                "shots": shot_counts,
                "total_features": sum(shot_counts.values())
            }
        return state
