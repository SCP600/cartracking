from __future__ import annotations

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
        self.config = AppConfig(model_path=project_root / "yolo26n.pt")
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

    def select_detection(self, detection_id: int, local_track_id: int = -1) -> None:
        if self.worker is not None:
            self.worker.request_target_selection(detection_id, local_track_id)

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
        if name == "botsort_reid":
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
