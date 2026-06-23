from __future__ import annotations

import queue
import sys
import threading
import types
import unittest
from pathlib import Path

import numpy as np


def _mock_resize(image, size, interpolation=None):
    width, height = size
    channels = image.shape[2] if getattr(image, "ndim", 0) == 3 else 1
    if channels == 1:
        return np.zeros((height, width), dtype=image.dtype)
    return np.zeros((height, width, channels), dtype=image.dtype)


def _mock_calc_hist(images, channels, mask, hist_size, ranges):
    return np.ones(tuple(hist_size), dtype=np.float32)


def _mock_normalize(src, dst, alpha=0, beta=1, norm_type=None):
    dst[...] = src
    return dst


sys.modules.setdefault(
    "cv2",
    types.SimpleNamespace(
        COLOR_BGRA2BGR=0,
        COLOR_BGR2RGB=1,
        COLOR_BGR2HSV=2,
        FONT_HERSHEY_SIMPLEX=0,
        HISTCMP_CORREL=0,
        INTER_AREA=3,
        INTER_LINEAR=1,
        LINE_AA=16,
        NORM_MINMAX=0,
        calcHist=_mock_calc_hist,
        compareHist=lambda lhs, rhs, method: 1.0,
        cvtColor=lambda image, code: image,
        getTextSize=lambda text, font, scale, thickness: ((len(text) * 10, 12), 3),
        normalize=_mock_normalize,
        putText=lambda *args, **kwargs: None,
        rectangle=lambda *args, **kwargs: None,
        resize=_mock_resize,
    ),
)

from autocam_tracker.app.app_controller import AppController
from autocam_tracker.app.app_state import AppConfig
from autocam_tracker.app.pipeline_worker import PipelineWorker
from autocam_tracker.detection.detection_models import VehicleDetection


class RecordingWorker:
    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple, dict]] = []

    def request_target_selection(self, *args, **kwargs) -> None:
        self.calls.append(("request_target_selection", args, kwargs))

    def request_target_binding(self, *args, **kwargs) -> None:
        self.calls.append(("request_target_binding", args, kwargs))

    def request_track_gid(self, *args, **kwargs) -> None:
        self.calls.append(("request_track_gid", args, kwargs))


class GidBindingRequestTest(unittest.TestCase):
    def test_bind_new_gid_uses_clicked_local_track_in_one_request(self) -> None:
        controller = AppController(project_root=Path("."))
        worker = RecordingWorker()
        controller.worker = worker

        controller.select_global_vehicle(global_vehicle_id=-1, local_track_id=7)

        self.assertEqual(
            worker.calls,
            [
                (
                    "request_target_selection",
                    (),
                    {"detection_id": -1, "local_track_id": 7, "global_vehicle_id": -1},
                )
            ],
        )

    def test_add_feature_to_gid_uses_clicked_local_track_in_one_request(self) -> None:
        controller = AppController(project_root=Path("."))
        worker = RecordingWorker()
        controller.worker = worker

        controller.add_feature_to_gid(global_vehicle_id=3, local_track_id=11)

        self.assertEqual(
            worker.calls,
            [
                (
                    "request_target_selection",
                    (),
                    {"detection_id": -1, "local_track_id": 11, "global_vehicle_id": 3},
                )
            ],
        )

    def test_worker_binding_request_is_consumed_as_bind_request(self) -> None:
        worker = PipelineWorker(
            source=types.SimpleNamespace(release=lambda: None),
            detector=types.SimpleNamespace(reset_tracking=lambda: None),
            tracker_config=Path("tracker.yaml"),
            frame_queue=queue.Queue(),
            stop_event=threading.Event(),
            app_config=AppConfig(model_path=Path("model.pt"), reid_model_path=None),
        )

        worker.request_target_binding(5)
        bind, select, *_ = worker._consume_requests()

        self.assertEqual(bind, 5)
        self.assertIsNone(select)

    def test_worker_binds_new_gid_to_clicked_track_not_largest_track(self) -> None:
        class FakeSource:
            def __init__(self) -> None:
                self.frame_index = 0
                self.timestamp_ms = 0.0
                self.frame_count = 1
                self.read_count = 0

            def open(self) -> None:
                pass

            def read(self):
                if self.read_count > 0:
                    return False, None
                self.read_count += 1
                self.frame_index = 1
                return True, np.zeros((80, 120, 3), dtype=np.uint8)

            def release(self) -> None:
                pass

        class FakeDetector:
            def reset_tracking(self) -> None:
                pass

            def track(self, **kwargs):
                return [
                    VehicleDetection(
                        detection_id=0,
                        local_track_id=5,
                        bbox=(0, 0, 80, 60),
                        center=(40, 30),
                        label="large",
                    ),
                    VehicleDetection(
                        detection_id=1,
                        local_track_id=7,
                        bbox=(90, 20, 20, 20),
                        center=(100, 30),
                        label="clicked",
                    ),
                ]

        class FakeReIDExtractor:
            def encode_detections(self, detections) -> None:
                for detection in detections:
                    detection.reid_feature = np.array([1.0, 0.0], dtype=np.float32)

        frame_queue = queue.Queue()
        worker = PipelineWorker(
            source=FakeSource(),
            detector=FakeDetector(),
            tracker_config=Path("tracker.yaml"),
            frame_queue=frame_queue,
            stop_event=threading.Event(),
            app_config=AppConfig(model_path=Path("model.pt"), reid_model_path=None),
        )
        worker.reid_feature_extractor = FakeReIDExtractor()

        worker.request_target_selection(detection_id=-1, local_track_id=7, global_vehicle_id=-1)
        worker._run_loop()
        frame_data = frame_queue.get_nowait()

        self.assertEqual(frame_data.selected_global_vehicle_id, 1)
        self.assertEqual(frame_data.selected_local_track_id, 7)
        self.assertEqual(frame_data.anchor_db_state[1]["total_features"], 1)
        self.assertEqual(
            {d.local_track_id: d.global_vehicle_id for d in frame_data.detections},
            {5: -1, 7: 1},
        )


if __name__ == "__main__":
    unittest.main()
