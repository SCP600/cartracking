from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class AppConfig:
    model_path: Path
    conf: float = 0.15
    imgsz: int = 960
    camera_id: int = 0
    max_queue_size: int = 1
    display_width: int = 640
    display_height: int = 360
    crop_max_zoom: float = 2.5


@dataclass
class SourceConfig:
    kind: str = "webcam"
    value: str = "0"

