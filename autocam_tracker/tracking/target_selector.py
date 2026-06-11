from __future__ import annotations


class TargetSelector:
    def __init__(self) -> None:
        self.selected_detection_id: int | None = None

    def select(self, detection_id: int) -> None:
        self.selected_detection_id = detection_id

    def consume(self) -> int | None:
        value = self.selected_detection_id
        self.selected_detection_id = None
        return value

