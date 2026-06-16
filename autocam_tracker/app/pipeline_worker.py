from __future__ import annotations

from pathlib import Path
import queue
import threading
import time

import cv2

from autocam_tracker.app.app_state import AppConfig
from autocam_tracker.data.detection_store import DetectionStore
from autocam_tracker.data.recognized_vehicle_registry import RecognizedVehicleRegistry
from autocam_tracker.detection.thumbnail_cropper import ThumbnailCropper
from autocam_tracker.detection.yolo26_detector import YOLO26Detector
from autocam_tracker.framing.crop_controller import CropController
from autocam_tracker.framing.framing_controller import FramingController
from autocam_tracker.identity.global_identity_manager import GlobalIdentityManager
from autocam_tracker.utils.image_utils import draw_detections
from autocam_tracker.utils.scene_cut import SceneCutDetector
from autocam_tracker.video.video_source import VideoSource


class PipelineWorker(threading.Thread):
    def __init__(
        self,
        source: VideoSource,
        detector: YOLO26Detector,
        tracker_config: Path,
        frame_queue: queue.Queue,
        stop_event: threading.Event,
        app_config: AppConfig,
    ) -> None:
        super().__init__(daemon=True)
        self.source = source
        self.detector = detector
        self.tracker_config = tracker_config
        self.frame_queue = frame_queue
        self.stop_event = stop_event
        self.app_config = app_config
        self.store = DetectionStore()
        self.recognized_registry = RecognizedVehicleRegistry()
        self.thumbnail_cropper = ThumbnailCropper()
        self.identity_manager = GlobalIdentityManager()
        self.scene_cut_detector = SceneCutDetector()
        self.framing = FramingController()
        self.cropper = CropController(max_zoom=app_config.crop_max_zoom)
        self._lock = threading.Lock()
        self._selection_request: tuple[int, int, int, bool] | None = None
        self._reset_request = False
        self._crop_reset_request = False
        self._seek_request: int | None = None
        self._crop_focus_enabled = False
        self._shot_id = 0

    def request_target_selection(
        self,
        detection_id: int,
        local_track_id: int = -1,
        global_vehicle_id: int = -1,
        focus_crop: bool = False,
    ) -> None:
        with self._lock:
            self._selection_request = (detection_id, local_track_id, global_vehicle_id, focus_crop)

    def request_target_reset(self) -> None:
        with self._lock:
            self._reset_request = True

    def request_crop_reset(self) -> None:
        with self._lock:
            self._crop_reset_request = True

    def request_seek(self, frame_index: int) -> None:
        with self._lock:
            self._seek_request = int(frame_index)

    def run(self) -> None:
        try:
            self._run_loop()
        finally:
            self.source.release()

    def _run_loop(self) -> None:
        self.source.open()
        last_tick = time.perf_counter()

        while not self.stop_event.is_set():
            selection_request, reset_request, crop_reset_request, seek_request = self._consume_requests()
            if reset_request:
                self._reset_tracking_state(clear_registry=True)
            if crop_reset_request:
                self._crop_focus_enabled = False
                self.cropper.reset()
            if seek_request is not None:
                self._seek_source(seek_request)

            ok, frame = self.source.read()
            if not ok or frame is None:
                break

            now = time.perf_counter()
            fps = 1.0 / max(now - last_tick, 1e-6)
            last_tick = now

            frame_index = self.source.frame_index
            timestamp_ms = self.source.timestamp_ms
            camera_cut = self.scene_cut_detector.update(frame)
            if camera_cut:
                self._shot_id += 1
                self.detector.reset_tracking()
                self.identity_manager.handle_camera_cut(self._shot_id)

            detect_start = time.perf_counter()
            detections = self.detector.track(
                frame=frame,
                tracker_config=self.tracker_config,
                camera_id=self.app_config.camera_id,
                shot_id=self._shot_id,
                frame_index=frame_index,
                timestamp_ms=timestamp_ms,
            )
            inference_time_ms = (time.perf_counter() - detect_start) * 1000.0

            for detection in detections:
                detection.thumbnail = self.thumbnail_cropper.crop(frame, detection.bbox)

            self.recognized_registry.apply_known_ids(detections)
            self.store.update(detections)
            resolved_selection = self._resolve_selection_request(detections, selection_request)

            track_start = time.perf_counter()
            target = self.identity_manager.update(
                detections=detections,
                selected_detection_id=resolved_selection,
                frame=frame,
                timestamp_ms=timestamp_ms,
                camera_id=self.app_config.camera_id,
                shot_id=self._shot_id,
            )
            tracking_time_ms = (time.perf_counter() - track_start) * 1000.0

            self.recognized_registry.update(
                detections=detections,
                selected_global_vehicle_id=self.identity_manager.selected_global_vehicle_id,
                tracking_status=self.identity_manager.status,
            )

            reframe_start = time.perf_counter()
            crop_target = target if self._crop_focus_enabled else None
            crop_result = self.cropper.crop(frame, crop_target)
            frame_data = self.framing.build_frame_data(
                raw_frame=frame,
                detection_frame=draw_detections(frame, detections, self.identity_manager.selected_global_vehicle_id),
                cropped_frame=crop_result.frame,
                detections=detections,
                camera_id=self.app_config.camera_id,
                shot_id=self._shot_id,
                frame_index=frame_index,
                total_frame_count=int(getattr(self.source, "frame_count", 0) or 0),
                timestamp_ms=timestamp_ms,
                tracking_status=self.identity_manager.status,
                selected_global_vehicle_id=self.identity_manager.selected_global_vehicle_id,
                selected_local_track_id=self.identity_manager.selected_local_track_id,
                selected_detection_id=self.identity_manager.selected_detection_id,
                camera_cut_detected=camera_cut,
                fps=fps,
                inference_time_ms=inference_time_ms,
                tracking_time_ms=tracking_time_ms,
                reframe_time_ms=(time.perf_counter() - reframe_start) * 1000.0,
                crop_result=crop_result,
                lost_frames=self.identity_manager.lost_frames,
                candidate_count=len(detections),
                reacquire_score=self.identity_manager.last_reacquire_score,
                recognized_vehicles=self.recognized_registry.summaries(),
            )

            self._push_latest(frame_data)

    def _consume_requests(self) -> tuple[tuple[int, int, int, bool] | None, bool, bool, int | None]:
        with self._lock:
            selection = self._selection_request
            reset = self._reset_request
            crop_reset = self._crop_reset_request
            seek = self._seek_request
            self._selection_request = None
            self._reset_request = False
            self._crop_reset_request = False
            self._seek_request = None
            return selection, reset, crop_reset, seek

    def _resolve_selection_request(
        self,
        detections: list,
        selection_request: tuple[int, int, int, bool] | None,
    ) -> tuple[int, int] | None:
        if selection_request is None:
            return None
        detection_id, local_track_id, global_vehicle_id, focus_crop = selection_request
        self._crop_focus_enabled = focus_crop
        if global_vehicle_id < 0:
            return detection_id, local_track_id

        for detection in detections:
            if detection.global_vehicle_id == global_vehicle_id:
                return detection.detection_id, detection.local_track_id

        aliases = set(self.recognized_registry.local_track_aliases_for_global(global_vehicle_id))
        if aliases:
            for detection in detections:
                if detection.local_track_id in aliases:
                    return detection.detection_id, detection.local_track_id
        if local_track_id >= 0:
            return detection_id, local_track_id
        return detection_id, local_track_id

    def _reset_tracking_state(self, clear_registry: bool) -> None:
        self.identity_manager.reset()
        if clear_registry:
            self.recognized_registry.clear()
        self.cropper.reset()
        self._crop_focus_enabled = False

    def _seek_source(self, frame_index: int) -> None:
        if not self.source.seek(frame_index):
            return
        self.detector.reset_tracking()
        self.scene_cut_detector = SceneCutDetector()
        self._shot_id += 1
        self._reset_tracking_state(clear_registry=True)

    def _push_latest(self, frame_data) -> None:
        try:
            self.frame_queue.put_nowait(frame_data)
            return
        except queue.Full:
            pass

        try:
            self.frame_queue.get_nowait()
        except queue.Empty:
            pass
        try:
            self.frame_queue.put_nowait(frame_data)
        except queue.Full:
            pass
