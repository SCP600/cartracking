from __future__ import annotations

import json
from pathlib import Path
import queue
import threading
from typing import Optional

from autocam_tracker.app.app_state import AppConfig, SourceConfig
from autocam_tracker.app.pipeline_worker import PipelineWorker
from autocam_tracker.detection.yolo26_detector import YOLO26Detector
from autocam_tracker.video.screen_region_source import ScreenRegionSource
from autocam_tracker.video.video_file_source import VideoFileSource
from autocam_tracker.video.webcam_source import WebcamSource


class AppController:
    def __init__(self, project_root: Path) -> None:
        self.project_root = project_root
        self.config = AppConfig.from_project_root(project_root)
        self.frame_queue: queue.Queue = queue.Queue(maxsize=self.config.max_queue_size)
        self.worker: Optional[PipelineWorker] = None
        self.stop_event: Optional[threading.Event] = None

    def start(self, source: SourceConfig, tracker_name: str) -> None:
        self.stop()

        if source.kind == "file":
            video_source = VideoFileSource(Path(source.value))
        elif source.kind == "screen":
            left, top, width, height = self._parse_screen_region(source.value)
            video_source = ScreenRegionSource(left, top, width, height)
        else:
            try:
                camera_index = int(source.value)
            except ValueError:
                camera_index = 0
            video_source = WebcamSource(camera_index)

        tracker_config = self._tracker_config_path(tracker_name)
        detector = YOLO26Detector(
            model_path=self.config.model_path,
            conf=self.config.conf,
            imgsz=self.config.imgsz,
            vehicle_class_ids=self.config.vehicle_class_ids,
            device=self.config.device,
        )

        self.stop_event = threading.Event()
        self.worker = PipelineWorker(
            source=video_source,
            detector=detector,
            tracker_config=tracker_config,
            frame_queue=self.frame_queue,
            stop_event=self.stop_event,
            app_config=self.config,
        )
        self.worker.start()

    def stop(self) -> None:
        if self.stop_event is not None:
            self.stop_event.set()
        if self.worker is not None:
            self.worker.join(timeout=2.0)
        self.worker = None
        self.stop_event = None

    def reset_target(self) -> None:
        if self.worker is not None:
            self.worker.request_target_reset()

    def reset_cropped(self) -> None:
        if self.worker is not None:
            self.worker.request_crop_reset()

    def select_detection(self, detection_id: int, local_track_id: int = -1) -> None:
        if self.worker is not None:
            self.worker.request_target_selection(detection_id, local_track_id)

    def select_global_vehicle(self, global_vehicle_id: int, local_track_id: int = -1) -> None:
        if self.worker is not None:
            self.worker.request_target_selection(-1, local_track_id, global_vehicle_id, focus_crop=True)

    def seek_frame(self, frame_index: int) -> None:
        if self.worker is not None:
            self.worker.request_seek(frame_index)

    def poll_frame(self):
        latest = None
        while True:
            try:
                latest = self.frame_queue.get_nowait()
            except queue.Empty:
                break
        return latest

    def _tracker_config_path(self, tracker_name: str) -> Path:
        name = tracker_name.lower().strip()
        if name in {"botsort_reid", "botsort_reid_custom", "botsort_reid_self_trained"}:
            base_config = self.project_root / "autocam_tracker" / "tracking" / "custom_botsort_reid.yaml"
            if self.config.reid_model_path:
                try:
                    cfg = _load_simple_tracker_yaml(base_config)
                    cfg["model"] = self.config.reid_model_path.as_posix()
                    
                    temp_config = self.project_root / "autocam_tracker" / "tracking" / ".dynamic_botsort_reid.yaml"
                    _write_simple_tracker_yaml(temp_config, cfg)
                    return temp_config
                except Exception as e:
                    print(f"Warning: Failed to create dynamic tracker config: {e}")
            return base_config
        if name in {"botsort_reid_default", "botsort_reid_auto"}:
            return self.project_root / "autocam_tracker" / "tracking" / "custom_botsort_reid.yaml"
        if name == "bytetrack":
            return self.project_root / "autocam_tracker" / "tracking" / "custom_bytetrack.yaml"
        return self.project_root / "autocam_tracker" / "tracking" / "custom_botsort.yaml"

    def _parse_screen_region(self, value: str) -> tuple[int, int, int, int]:
        parts = [int(part.strip()) for part in value.split(",")]
        if len(parts) != 4:
            raise ValueError("Screen region must be left,top,width,height.")
        left, top, width, height = parts
        if width < 16 or height < 16:
            raise ValueError("Screen region is too small.")
        return left, top, width, height


def _load_simple_tracker_yaml(path: Path) -> dict[str, object]:
    config: dict[str, object] = {}
    with open(path, "r", encoding="utf-8") as file:
        for line in file:
            stripped = line.strip()
            if not stripped or stripped.startswith("#") or ":" not in stripped:
                continue
            key, value = stripped.split(":", 1)
            config[key.strip()] = _parse_simple_yaml_scalar(value.strip())
    return config


def _write_simple_tracker_yaml(path: Path, config: dict[str, object]) -> None:
    with open(path, "w", encoding="utf-8") as file:
        for key, value in config.items():
            file.write(f"{key}: {_format_simple_yaml_scalar(value)}\n")


def _parse_simple_yaml_scalar(value: str) -> object:
    if value.lower() == "true":
        return True
    if value.lower() == "false":
        return False
    if value.lower() in ("null", "none"):
        return None
    try:
        if "." in value:
            return float(value)
        return int(value)
    except ValueError:
        return value.strip("\"'")


def _format_simple_yaml_scalar(value: object) -> str:
    if isinstance(value, bool):
        return "True" if value else "False"
    if value is None:
        return "null"
    if isinstance(value, (int, float)):
        return str(value)
    text = str(value)
    if text == "auto":
        return text
    return json.dumps(text)
