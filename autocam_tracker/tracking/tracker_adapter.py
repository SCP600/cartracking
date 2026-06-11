from __future__ import annotations

from pathlib import Path


class TrackerAdapter:
    def __init__(self, config_path: Path) -> None:
        self.config_path = config_path

    def as_ultralytics_tracker(self) -> str:
        return str(self.config_path)

