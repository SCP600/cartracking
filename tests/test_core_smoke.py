from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np

from autocam_tracker.app.app_controller import AppController
from autocam_tracker.data.recognized_vehicle_registry import RecognizedVehicleRegistry
from autocam_tracker.detection.detection_models import VehicleDetection
from autocam_tracker.framing.crop_controller import CropController
from autocam_tracker.identity.global_identity_manager import GlobalIdentityManager


class CoreSmokeTest(unittest.TestCase):
    def test_global_identity_survives_missing_and_camera_cut(self) -> None:
        frame = np.zeros((240, 320, 3), dtype=np.uint8)
        detection = VehicleDetection(
            detection_id=0,
            local_track_id=42,
            confidence=0.9,
            bbox=(100, 80, 60, 40),
            center=(130, 100),
        )
        manager = GlobalIdentityManager()

        target = manager.update([detection], (999, 42), frame, 0.0, 0, 0)

        self.assertIsNotNone(target)
        self.assertEqual(manager.selected_global_vehicle_id, 1)
        self.assertEqual(manager.selected_local_track_id, 42)

        manager.update([], None, frame, 33.0, 0, 0)
        self.assertEqual(manager.selected_global_vehicle_id, 1)

        manager.handle_camera_cut(1)
        self.assertEqual(manager.selected_global_vehicle_id, 1)
        self.assertEqual(manager.selected_local_track_id, -1)

    def test_crop_output_keeps_frame_shape(self) -> None:
        frame = np.zeros((240, 320, 3), dtype=np.uint8)
        target = VehicleDetection(
            detection_id=0,
            local_track_id=1,
            confidence=0.8,
            bbox=(120, 90, 50, 35),
            center=(145, 107),
        )
        cropper = CropController()

        result = cropper.crop(frame, target)

        self.assertEqual(result.frame.shape, frame.shape)
        self.assertGreaterEqual(result.zoom_ratio, 1.0)

    def test_screen_region_parser(self) -> None:
        controller = AppController(project_root=Path("."))

        self.assertEqual(controller._parse_screen_region("10,20,300,200"), (10, 20, 300, 200))

        with self.assertRaises(ValueError):
            controller._parse_screen_region("10,20,8,200")

    def test_recognized_registry_keeps_anchored_summary(self) -> None:
        detection = VehicleDetection(
            detection_id=0,
            local_track_id=7,
            global_vehicle_id=1,
            shot_id=2,
            frame_index=10,
            confidence=0.8,
        )
        registry = RecognizedVehicleRegistry()

        registry.update([detection], selected_global_vehicle_id=1, tracking_status="Tracking")
        summaries = registry.summaries()

        self.assertEqual(len(summaries), 1)
        self.assertEqual(summaries[0].registry_id, "G1")
        self.assertTrue(summaries[0].selected)

    def test_recognized_registry_merges_track_fragments_by_appearance(self) -> None:
        thumb = np.zeros((54, 96, 3), dtype=np.uint8)
        thumb[:, :] = (245, 245, 245)
        first = VehicleDetection(
            detection_id=0,
            local_track_id=35,
            shot_id=0,
            frame_index=10,
            confidence=0.9,
            bbox=(100, 100, 90, 55),
            center=(145, 127),
            thumbnail=thumb,
        )
        second = VehicleDetection(
            detection_id=0,
            local_track_id=41,
            shot_id=0,
            frame_index=30,
            confidence=0.88,
            bbox=(110, 104, 86, 58),
            center=(153, 133),
            thumbnail=thumb.copy(),
        )
        registry = RecognizedVehicleRegistry()

        registry.update([first], selected_global_vehicle_id=-1, tracking_status="Detecting")
        registry.update([second], selected_global_vehicle_id=-1, tracking_status="Detecting")
        summaries = registry.summaries()

        self.assertEqual(len(summaries), 1)
        self.assertEqual(summaries[0].registry_id, "R1")
        self.assertEqual(summaries[0].local_track_aliases, [35, 41])


if __name__ == "__main__":
    unittest.main()
