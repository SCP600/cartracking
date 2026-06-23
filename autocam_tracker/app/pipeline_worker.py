from __future__ import annotations

from pathlib import Path
import queue
import threading
import time

import cv2

from autocam_tracker.app.app_state import AppConfig
from autocam_tracker.data.detection_store import DetectionStore
from autocam_tracker.detection.thumbnail_cropper import ThumbnailCropper
from autocam_tracker.detection.yolo26_detector import YOLO26Detector
from autocam_tracker.framing.crop_controller import CropController
from autocam_tracker.framing.framing_controller import FramingController
from autocam_tracker.identity.global_identity_manager import GlobalIdentityManager
from autocam_tracker.identity.hierarchical_anchor_database import HierarchicalAnchorDatabase
from autocam_tracker.identity.reid_feature_extractor import RuntimeReIDFeatureExtractor
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
        self.pause_event = threading.Event()
        self._step_one_frame = False
        self.app_config = app_config
        self.store = DetectionStore()
        
        self.anchor_db = HierarchicalAnchorDatabase()
        
        self.thumbnail_cropper = ThumbnailCropper()
        self.reid_feature_extractor = RuntimeReIDFeatureExtractor(app_config.reid_model_path, device=app_config.device)
        self.identity_manager = GlobalIdentityManager()
        self.scene_cut_detector = SceneCutDetector()
        self.framing = FramingController()
        self.cropper = CropController(max_zoom=app_config.crop_max_zoom)
        self._lock = threading.Lock()
        
        self._bind_request: int | None = None
        self._select_request: int | None = None
        self._track_gid_request: int | None = None
        self._speed_request: float | None = None
        self._playback_speed: float = 1.0
        self._reset_request = False
        self._crop_reset_request = False
        self._db_reset_request = False
        self._seek_request: int | None = None
        self._crop_focus_enabled = True
        self._shot_id = 1
        
        self.next_global_id = 1

    def request_target_selection(
        self,
        detection_id: int,
        local_track_id: int = -1,
        global_vehicle_id: int = -1,
        focus_crop: bool = False,
    ) -> None:
        with self._lock:
            if detection_id == -1:
                self._bind_request = global_vehicle_id
                if local_track_id != -1:
                    self._select_request = local_track_id
            else:
                self._select_request = local_track_id

    def request_target_binding(self, global_vehicle_id: int = -1) -> None:
        with self._lock:
            self._bind_request = global_vehicle_id

    def request_target_reset(self) -> None:
        with self._lock:
            self._reset_request = True

    def toggle_pause(self) -> bool:
        if self.pause_event.is_set():
            self.pause_event.clear()
            return False
        else:
            self.pause_event.set()
            return True

    def request_playback_speed(self, speed: float) -> None:
        with self._lock:
            self._speed_request = speed

    def request_track_gid(self, gid: int) -> None:
        with self._lock:
            self._track_gid_request = gid

    def request_crop_reset(self) -> None:
        with self._lock:
            self._crop_reset_request = True

    def request_db_reset(self) -> None:
        with self._lock:
            self._db_reset_request = True

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
        ideal_time = time.perf_counter()

        while not self.stop_event.is_set():
            if self.pause_event.is_set() and not self._step_one_frame:
                bind_request, select_request, reset_request, crop_reset_request, seek_request, speed_request, track_gid_request, db_reset_request = self._consume_requests()
                if seek_request is not None:
                    self._seek_source(seek_request)
                    self._step_one_frame = True
                else:
                    with self._lock:
                        if bind_request is not None: 
                            self._bind_request = bind_request
                            self._step_one_frame = True
                        if select_request is not None: 
                            self._select_request = select_request
                            self._step_one_frame = True
                        if track_gid_request is not None: 
                            self._track_gid_request = track_gid_request
                            self._step_one_frame = True
                        if speed_request is not None: 
                            self._speed_request = speed_request
                        if reset_request: 
                            self._reset_request = True
                            self._step_one_frame = True
                        if crop_reset_request: 
                            self._crop_reset_request = True
                            self._step_one_frame = True
                        if db_reset_request: 
                            self._db_reset_request = True
                            self._step_one_frame = True
                    
                    if self._step_one_frame:
                        continue
                        
                    time.sleep(0.05)
                    ideal_time = time.perf_counter()
                    continue
            
            self._step_one_frame = False

            bind_request, select_request, reset_request, crop_reset_request, seek_request, speed_request, track_gid_request, db_reset_request = self._consume_requests()
            if speed_request is not None:
                self._playback_speed = speed_request
                ideal_time = time.perf_counter()
                
            if reset_request:
                self.identity_manager.reset()
                self.cropper.reset()
            if crop_reset_request:
                self._crop_focus_enabled = False
                self.cropper.reset()
            if db_reset_request:
                self.anchor_db.clear()
                self.identity_manager.reset()
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
            
            camera_cut, self._shot_id = self.scene_cut_detector.update(frame)
            if camera_cut:
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

            # --- Process Select Request ---
            if select_request is not None and select_request != -1:
                self.identity_manager.set_target(-1, select_request)

            if track_gid_request is not None:
                self.identity_manager.set_target(track_gid_request, -1)
                force_reid = True
            else:
                force_reid = False
                
            # --- Performance Optimization: Only extract ReID when binding or doing periodic 15-frame refresh ---
            is_periodic_refresh = frame_index % 15 == 0 and self.identity_manager.selected_global_vehicle_id != -1
            
            need_reid = force_reid
            if bind_request is not None:
                need_reid = True
            if is_periodic_refresh:
                need_reid = True
                
            if need_reid:
                reid_candidates = detections
                
                if bind_request is not None:
                    # Performance & OOM Prevention: Only extract ReID for the selected target during binding
                    reid_candidates = [d for d in detections if d.local_track_id == self.identity_manager.selected_local_track_id]
                elif is_periodic_refresh and not force_reid:
                    # Performance: Limit ReID scope if just doing periodic refresh
                    tracked_det = next((d for d in detections if d.local_track_id == self.identity_manager.selected_local_track_id), None)
                    if tracked_det:
                        tx = tracked_det.bbox[0] + tracked_det.bbox[2] / 2
                        ty = tracked_det.bbox[1] + tracked_det.bbox[3] / 2
                        radius_sq = 300 * 300
                        reid_candidates = []
                        for d in detections:
                            dx = d.bbox[0] + d.bbox[2] / 2
                            dy = d.bbox[1] + d.bbox[3] / 2
                            if (tx - dx)**2 + (ty - dy)**2 <= radius_sq:
                                reid_candidates.append(d)
                
                for detection in reid_candidates:
                    detection.thumbnail = self.thumbnail_cropper.crop(frame, detection.bbox)
                self.reid_feature_extractor.encode_detections(reid_candidates)

            self.store.update(detections)
                
            # --- Auto Select Largest if nothing tracked ---
            if self.identity_manager.selected_local_track_id == -1 and detections:
                largest = max(detections, key=lambda d: d.bbox[2] * d.bbox[3])
                self.identity_manager.set_target(self.identity_manager.selected_global_vehicle_id, largest.local_track_id)
                
            # --- Process Binding Request ---
            if bind_request is not None:
                tracked_det = next((d for d in detections if d.local_track_id == self.identity_manager.selected_local_track_id), None)
                if tracked_det and tracked_det.reid_feature is not None:
                    gid = bind_request
                    if gid == -1:
                        gid = self.next_global_id
                        self.next_global_id += 1
                    self.anchor_db.add_anchor(gid, self._shot_id, tracked_det.reid_feature, tracked_det.label)
                    self.identity_manager.set_target(gid, tracked_det.local_track_id)

            # --- 15 Frame Periodic Refresh or Forced ReID ---
            if force_reid or is_periodic_refresh:
                features = [d.reid_feature for d in detections]
                local_ids = [d.local_track_id for d in detections]
                
                best_track = self.anchor_db.match_candidates(
                    shot_id=self._shot_id,
                    candidate_features=features,
                    target_global_id=self.identity_manager.selected_global_vehicle_id,
                    current_local_track_id=self.identity_manager.selected_local_track_id,
                    local_track_ids=local_ids,
                )
                if best_track != -1:
                    self.identity_manager.set_target(self.identity_manager.selected_global_vehicle_id, best_track)
                    
                    # Auto-add anchor if this is the first time we found it in a new shot
                    if not self.anchor_db.has_anchors_for_shot(self.identity_manager.selected_global_vehicle_id, self._shot_id):
                        idx = local_ids.index(best_track)
                        feat = features[idx]
                        if feat is not None:
                            det = detections[idx]
                            self.anchor_db.add_anchor(self.identity_manager.selected_global_vehicle_id, self._shot_id, feat, det.label)

            # --- Apply markings to detections ---
            target = None
            for d in detections:
                if d.local_track_id == self.identity_manager.selected_local_track_id:
                    d.selected = True
                    if self.identity_manager.selected_global_vehicle_id != -1:
                        d.global_vehicle_id = self.identity_manager.selected_global_vehicle_id
                    target = d

            track_start = time.perf_counter()
            tracking_time_ms = (time.perf_counter() - track_start) * 1000.0

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
                selected_detection_id=-1,
                camera_cut_detected=camera_cut,
                fps=fps,
                inference_time_ms=inference_time_ms,
                tracking_time_ms=tracking_time_ms,
                reframe_time_ms=(time.perf_counter() - reframe_start) * 1000.0,
                crop_result=crop_result,
                lost_frames=0,
                candidate_count=len(detections),
                reacquire_score=0.0,
                recognized_vehicles=[],
            )
            
            # Inject anchor db state for the UI
            frame_data.anchor_db_state = self.anchor_db.get_state()

            # Pacing logic to match playback speed
            if hasattr(self.source, "fps") and self.source.fps > 0:
                frame_interval = 1.0 / (self.source.fps * self._playback_speed)
                ideal_time += frame_interval
                sleep_time = ideal_time - time.perf_counter()
                if sleep_time > 0:
                    time.sleep(sleep_time)
                else:
                    # We are lagging! Drop frames to catch up to real-time speed.
                    lag_time = -sleep_time
                    frames_to_drop = int(lag_time / frame_interval)
                    if frames_to_drop > 0:
                        # Limit dropping to max 5 frames at a time to keep tracking somewhat stable
                        frames_to_drop = min(frames_to_drop, 5)
                        for _ in range(frames_to_drop):
                            ok, _ = self.source.read()
                            if not ok:
                                break
                        ideal_time += frames_to_drop * frame_interval
                        
                    # If we are STILL lagging (hardware maxed out), reset ideal_time to prevent endless catchup
                    if ideal_time < time.perf_counter():
                        ideal_time = time.perf_counter()

            self._push_latest(frame_data)

    def _consume_requests(self) -> tuple[int | None, int | None, bool, bool, int | None, float | None, int | None, bool]:
        with self._lock:
            bind = self._bind_request
            select = self._select_request
            track_gid = self._track_gid_request
            speed = self._speed_request
            reset = self._reset_request
            crop_reset = self._crop_reset_request
            seek = self._seek_request
            db_reset = self._db_reset_request
            
            self._bind_request = None
            self._select_request = None
            self._track_gid_request = None
            self._speed_request = None
            self._reset_request = False
            self._crop_reset_request = False
            self._db_reset_request = False
            self._seek_request = None
            return bind, select, reset, crop_reset, seek, speed, track_gid, db_reset

    def _seek_source(self, frame_index: int) -> None:
        if not self.source.seek(frame_index):
            return
        self.detector.reset_tracking()
        self.scene_cut_detector = SceneCutDetector()
        self._shot_id += 1
        self.identity_manager.reset()
        self.cropper.reset()
        self._crop_focus_enabled = True

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
