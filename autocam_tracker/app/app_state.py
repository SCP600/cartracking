from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path


@dataclass
class AppConfig:
    model_path: Path
    reid_model_path: Path | None = None
    vehicle_class_ids: list[int] | None = None
    tracker: str = "botsort_reid"
    conf: float = 0.15
    imgsz: int = 960
    device: int | str | None = None
    camera_id: int = 0
    max_queue_size: int = 1
    display_width: int = 640
    display_height: int = 360
    crop_max_zoom: float = 2.5
    gid_reid_memory_size: int = 24
    gid_reid_match_threshold: float = 0.82
    gid_reid_cross_shot_threshold: float = 0.86
    gid_reid_margin: float = 0.04
    gid_reid_duplicate_similarity: float = 0.985

    @classmethod
    def from_project_root(cls, project_root: Path) -> "AppConfig":
        project_root = Path(project_root)
        config_path = project_root / "autocam_tracker" / "config" / "default_config.json"
        raw: dict = {}
        if config_path.exists():
            with open(config_path, "r", encoding="utf-8") as file:
                raw = json.load(file)

        model_path = _resolve_optional_path(project_root, raw.get("model_path")) or project_root / "yolo11n.pt"
        reid_model_path = _resolve_optional_path(project_root, raw.get("reid_model_path"))
        return cls(
            model_path=model_path,
            reid_model_path=reid_model_path,
            vehicle_class_ids=_parse_optional_int_list(raw.get("vehicle_class_ids", [2, 3, 5, 7])),
            tracker=str(raw.get("tracker", "botsort_reid")),
            conf=float(raw.get("conf", 0.15)),
            imgsz=int(raw.get("imgsz", 960)),
            device=_parse_optional_device(raw.get("device")),
            camera_id=int(raw.get("camera_id", 0)),
            max_queue_size=int(raw.get("max_queue_size", 1)),
            display_width=int(raw.get("display_width", 640)),
            display_height=int(raw.get("display_height", 360)),
            crop_max_zoom=float(raw.get("crop_max_zoom", 2.5)),
            gid_reid_memory_size=int(raw.get("gid_reid_memory_size", 24)),
            gid_reid_match_threshold=float(raw.get("gid_reid_match_threshold", 0.82)),
            gid_reid_cross_shot_threshold=float(raw.get("gid_reid_cross_shot_threshold", 0.86)),
            gid_reid_margin=float(raw.get("gid_reid_margin", 0.04)),
            gid_reid_duplicate_similarity=float(raw.get("gid_reid_duplicate_similarity", 0.985)),
        )


@dataclass
class SourceConfig:
    kind: str = "webcam"
    value: str = "0"


def _resolve_optional_path(project_root: Path, value) -> Path | None:
    if value in (None, "", "auto"):
        return None
    path = Path(str(value))
    if not path.is_absolute():
        path = project_root / path
    return path


def _parse_optional_int_list(value) -> list[int] | None:
    if value in (None, "", "auto"):
        return None
    if isinstance(value, str):
        value = [item.strip() for item in value.split(",") if item.strip()]
    return [int(item) for item in value]


def _parse_optional_device(value) -> int | str | None:
    if value in (None, "", "auto"):
        return None
    if isinstance(value, int):
        return value
    text = str(value).strip()
    if text.isdigit():
        return int(text)
    return text

