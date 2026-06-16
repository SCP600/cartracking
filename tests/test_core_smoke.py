from __future__ import annotations

import json
import queue
import threading
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from autocam_tracker.app.app_state import AppConfig
from autocam_tracker.app.app_controller import AppController
from autocam_tracker.app.pipeline_worker import PipelineWorker
from autocam_tracker.data.recognized_vehicle_registry import RecognizedVehicleRegistry
from autocam_tracker.detection.detection_models import VehicleDetection
from autocam_tracker.detection.yolo26_detector import YOLO26Detector
from autocam_tracker.framing.crop_controller import CropController, CropResult
from autocam_tracker.framing.framing_controller import FramingController
from autocam_tracker.identity.global_identity_manager import GlobalIdentityManager
from autocam_tracker.identity.reid_feature_extractor import RuntimeReIDFeatureExtractor
from autocam_tracker.video.video_file_source import VideoFileSource
from scripts import (
    auto_curate_vehicle_identity_tracks,
    build_vehicle_identity_candidates,
    cluster_vehicle_identity_tracks,
    curate_vehicle_identity_dataset,
    export_fastreid_reid_onnx,
    prepare_fastreid_veri_dataset,
    prepare_reid_dataset_from_yolo,
    prepare_yolo_dataset_from_identity_crops,
    train_fastreid,
    train_yolo,
)


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

    def test_global_identity_adopts_preassigned_gid(self) -> None:
        frame = np.zeros((240, 320, 3), dtype=np.uint8)
        detection = VehicleDetection(
            detection_id=0,
            local_track_id=9,
            global_vehicle_id=12,
            confidence=0.9,
            bbox=(100, 80, 60, 40),
            center=(130, 100),
        )
        manager = GlobalIdentityManager()

        target = manager.update([detection], (0, 9), frame, 0.0, 0, 0)

        self.assertIsNotNone(target)
        self.assertEqual(manager.selected_global_vehicle_id, 12)
        self.assertEqual(detection.global_vehicle_id, 12)

    def test_global_identity_reanchors_by_registry_gid(self) -> None:
        frame = np.zeros((240, 320, 3), dtype=np.uint8)
        first = VehicleDetection(
            detection_id=0,
            local_track_id=9,
            global_vehicle_id=12,
            confidence=0.9,
            bbox=(100, 80, 60, 40),
            center=(130, 100),
        )
        manager = GlobalIdentityManager()
        manager.update([first], (0, 9), frame, 0.0, 0, 0)

        later = VehicleDetection(
            detection_id=1,
            local_track_id=31,
            global_vehicle_id=12,
            confidence=0.88,
            reid_score=0.91,
            reid_matched=True,
            bbox=(150, 90, 64, 42),
            center=(182, 111),
        )
        target = manager.update([later], None, frame, 33.0, 0, 1)

        self.assertIs(target, later)
        self.assertEqual(manager.selected_global_vehicle_id, 12)
        self.assertEqual(manager.selected_local_track_id, 31)

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

    def test_video_file_source_seek_updates_frame_position(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            import cv2

            video_path = Path(tmp) / "seek_test.avi"
            writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"MJPG"), 10.0, (64, 48))
            for index in range(6):
                frame = np.full((48, 64, 3), index * 30, dtype=np.uint8)
                writer.write(frame)
            writer.release()

            source = VideoFileSource(video_path)
            source.open()
            try:
                self.assertGreaterEqual(source.frame_count, 6)
                self.assertTrue(source.seek(3))
                ok, frame = source.read()
                self.assertTrue(ok)
                self.assertIsNotNone(frame)
                self.assertEqual(source.frame_index, 4)
            finally:
                source.release()

    def test_frame_data_carries_video_timeline_metadata(self) -> None:
        frame = np.zeros((48, 64, 3), dtype=np.uint8)
        crop_result = CropResult(
            frame=frame,
            crop_x=0,
            crop_y=0,
            crop_w=64,
            crop_h=48,
            zoom_ratio=1.0,
            zoom_error=0.0,
            error_x=0.0,
            error_y=0.0,
            normalized_error_x=0.0,
            normalized_error_y=0.0,
        )

        frame_data = FramingController().build_frame_data(
            raw_frame=frame,
            detection_frame=frame,
            cropped_frame=frame,
            detections=[],
            camera_id=0,
            shot_id=0,
            frame_index=12,
            total_frame_count=100,
            timestamp_ms=400.0,
            tracking_status="Detecting",
            selected_global_vehicle_id=-1,
            selected_local_track_id=-1,
            selected_detection_id=-1,
            camera_cut_detected=False,
            fps=30.0,
            inference_time_ms=0.0,
            tracking_time_ms=0.0,
            reframe_time_ms=0.0,
            crop_result=crop_result,
            lost_frames=0,
            candidate_count=0,
            reacquire_score=0.0,
        )

        self.assertEqual(frame_data.frame_index, 12)
        self.assertEqual(frame_data.total_frame_count, 100)

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

    def test_recognized_registry_preserves_ids_when_runtime_state_resets(self) -> None:
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

        registry.reset_runtime_state()
        summaries = registry.summaries()

        self.assertEqual(len(summaries), 1)
        self.assertEqual(summaries[0].global_vehicle_id, 1)
        self.assertFalse(summaries[0].selected)
        self.assertEqual(summaries[0].status, "NotVisible")

    def test_pipeline_seek_preserves_recognized_vehicle_registry(self) -> None:
        class FakeSource:
            def __init__(self) -> None:
                self.seek_calls: list[int] = []

            def seek(self, frame_index: int) -> bool:
                self.seek_calls.append(frame_index)
                return True

        class FakeDetector:
            def __init__(self) -> None:
                self.reset_count = 0

            def reset_tracking(self) -> None:
                self.reset_count += 1

        source = FakeSource()
        detector = FakeDetector()
        worker = PipelineWorker(
            source=source,
            detector=detector,
            tracker_config=Path("tracker.yaml"),
            frame_queue=queue.Queue(),
            stop_event=threading.Event(),
            app_config=AppConfig(model_path=Path("model.pt")),
        )
        detection = VehicleDetection(
            detection_id=0,
            local_track_id=7,
            global_vehicle_id=1,
            shot_id=0,
            frame_index=10,
            confidence=0.8,
        )
        worker.recognized_registry.update([detection], selected_global_vehicle_id=1, tracking_status="Tracking")

        worker._seek_source(42)
        summaries = worker.recognized_registry.summaries()

        self.assertEqual(source.seek_calls, [42])
        self.assertEqual(detector.reset_count, 1)
        self.assertEqual(len(summaries), 1)
        self.assertEqual(summaries[0].global_vehicle_id, 1)
        self.assertFalse(summaries[0].selected)
        self.assertEqual(summaries[0].status, "NotVisible")

    def test_recognized_registry_assigns_gid_to_unselected_vehicle(self) -> None:
        detection = VehicleDetection(
            detection_id=0,
            local_track_id=7,
            shot_id=2,
            frame_index=10,
            confidence=0.8,
        )
        registry = RecognizedVehicleRegistry()

        registry.update([detection], selected_global_vehicle_id=-1, tracking_status="Detecting")
        summaries = registry.summaries()

        self.assertEqual(len(summaries), 1)
        self.assertEqual(summaries[0].registry_id, "G1")
        self.assertEqual(summaries[0].global_vehicle_id, 1)
        self.assertEqual(detection.global_vehicle_id, 1)
        self.assertEqual(registry.local_track_aliases_for_global(1), [7])
        self.assertEqual(registry.local_track_aliases_for_global(1, shot_id=2), [7])
        self.assertEqual(registry.local_track_aliases_for_global(1, shot_id=3), [])

    def test_recognized_registry_matches_reid_memory_across_shots(self) -> None:
        registry = RecognizedVehicleRegistry(reid_cross_shot_threshold=0.86, reid_margin=0.02)
        first = VehicleDetection(
            detection_id=0,
            local_track_id=7,
            shot_id=0,
            frame_index=10,
            confidence=0.8,
            reid_feature=np.array([1.0, 0.0], dtype=np.float32),
        )
        registry.update([first], selected_global_vehicle_id=-1, tracking_status="Detecting")

        later = VehicleDetection(
            detection_id=0,
            local_track_id=3,
            shot_id=1,
            frame_index=200,
            confidence=0.85,
            reid_feature=np.array([0.99, 0.04], dtype=np.float32),
        )
        registry.apply_known_ids([later])
        registry.update([later], selected_global_vehicle_id=-1, tracking_status="Detecting")
        summaries = registry.summaries()

        self.assertEqual(later.global_vehicle_id, 1)
        self.assertTrue(later.reid_matched)
        self.assertEqual(len(summaries), 1)
        self.assertEqual(summaries[0].global_vehicle_id, 1)
        self.assertEqual(summaries[0].local_track_aliases, [3])
        self.assertGreaterEqual(summaries[0].reid_feature_count, 1)

    def test_recognized_registry_keeps_small_diverse_reid_memory(self) -> None:
        registry = RecognizedVehicleRegistry(reid_memory_size=2, reid_duplicate_similarity=0.98)
        features = [
            np.array([1.0, 0.0], dtype=np.float32),
            np.array([0.999, 0.02], dtype=np.float32),
            np.array([0.0, 1.0], dtype=np.float32),
            np.array([-1.0, 0.0], dtype=np.float32),
        ]
        for index, feature in enumerate(features):
            registry.update(
                [
                    VehicleDetection(
                        detection_id=0,
                        local_track_id=7,
                        global_vehicle_id=1,
                        shot_id=0,
                        frame_index=index,
                        confidence=0.8,
                        reid_feature=feature,
                    )
                ],
                selected_global_vehicle_id=-1,
                tracking_status="Detecting",
            )

        summary = registry.summaries()[0]
        self.assertEqual(summary.global_vehicle_id, 1)
        self.assertEqual(summary.reid_feature_count, 2)
        self.assertLessEqual(len(summary.reid_features), 2)

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
        self.assertEqual(summaries[0].registry_id, "G1")
        self.assertEqual(summaries[0].local_track_aliases, [35, 41])

    def test_runtime_reid_extractor_noops_without_model(self) -> None:
        detection = VehicleDetection(thumbnail=np.zeros((54, 96, 3), dtype=np.uint8))
        extractor = RuntimeReIDFeatureExtractor(Path("missing.torchscript"))

        extractor.encode_detections([detection])

        self.assertIsNone(detection.reid_feature)

    def test_app_config_loads_reid_model_path_from_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            project_root = Path(tmp)
            config_dir = project_root / "autocam_tracker" / "config"
            config_dir.mkdir(parents=True)
            config_path = config_dir / "default_config.json"
            config_path.write_text(
                json.dumps(
                    {
                        "model_path": "models/yolo_vehicle.pt",
                        "reid_model_path": "weights/fastreid_vehicle.onnx",
                        "vehicle_class_ids": None,
                        "tracker": "botsort_reid",
                        "conf": 0.2,
                        "imgsz": 640,
                        "device": 0,
                    }
                ),
                encoding="utf-8",
            )

            config = AppConfig.from_project_root(project_root)

            self.assertEqual(config.model_path, project_root / "models" / "yolo_vehicle.pt")
            self.assertEqual(config.reid_model_path, project_root / "weights" / "fastreid_vehicle.onnx")
            self.assertIsNone(config.vehicle_class_ids)
            self.assertEqual(config.tracker, "botsort_reid")
            self.assertEqual(config.conf, 0.2)
            self.assertEqual(config.imgsz, 640)
            self.assertEqual(config.device, 0)

    def test_detector_omits_class_filter_for_custom_vehicle_model(self) -> None:
        class FakeModel:
            def __init__(self) -> None:
                self.kwargs = {}

            def track(self, frame, **kwargs):
                self.kwargs = kwargs
                return []

        fake_model = FakeModel()
        detector = YOLO26Detector(
            model_path=Path("custom_vehicle.pt"),
            vehicle_class_ids=None,
            device=0,
        )
        detector._model = fake_model

        detections = detector.track(
            frame=np.zeros((32, 32, 3), dtype=np.uint8),
            tracker_config=Path("tracker.yaml"),
            camera_id=0,
            shot_id=0,
            frame_index=0,
            timestamp_ms=0.0,
        )

        self.assertEqual(detections, [])
        self.assertNotIn("classes", fake_model.kwargs)
        self.assertEqual(fake_model.kwargs["device"], 0)

    def test_dynamic_botsort_reid_config_uses_custom_model(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            project_root = Path(tmp)
            tracking_dir = project_root / "autocam_tracker" / "tracking"
            tracking_dir.mkdir(parents=True)
            (tracking_dir / "custom_botsort_reid.yaml").write_text(
                "tracker_type: botsort\nwith_reid: true\nmodel: auto\n",
                encoding="utf-8",
            )
            controller = AppController(project_root=project_root)
            controller.config.reid_model_path = project_root / "weights" / "fastreid_vehicle.onnx"

            tracker_config = controller._tracker_config_path("botsort_reid")

            self.assertEqual(tracker_config, tracking_dir / ".dynamic_botsort_reid.yaml")
            self.assertIn(controller.config.reid_model_path.as_posix(), tracker_config.read_text(encoding="utf-8"))

    def test_default_botsort_reid_config_keeps_auto_model(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            project_root = Path(tmp)
            tracking_dir = project_root / "autocam_tracker" / "tracking"
            tracking_dir.mkdir(parents=True)
            base_config = tracking_dir / "custom_botsort_reid.yaml"
            base_config.write_text(
                "tracker_type: botsort\nwith_reid: true\nmodel: auto\n",
                encoding="utf-8",
            )
            controller = AppController(project_root=project_root)
            controller.config.reid_model_path = project_root / "weights" / "fastreid_vehicle.onnx"

            tracker_config = controller._tracker_config_path("botsort_reid_default")

            self.assertEqual(tracker_config, base_config)
            self.assertEqual("auto", base_config.read_text(encoding="utf-8").split("model:", 1)[1].strip())

    def test_training_script_defaults_match_optimization_plan(self) -> None:
        fastreid_args = train_fastreid.build_parser().parse_args([])
        yolo_args = train_yolo.build_parser().parse_args([])

        self.assertEqual(fastreid_args.lr, 0.00035)
        self.assertEqual(fastreid_args.batch_size, 256)
        self.assertEqual(fastreid_args.epochs, 10)
        self.assertEqual(yolo_args.lr, 0.01)
        self.assertEqual(yolo_args.batch_size, 16)
        self.assertEqual(yolo_args.epochs, 20)
        self.assertEqual(yolo_args.optimizer, "SGD")

    def test_fastreid_dataset_inspection_counts_identities_and_images(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            data_dir = Path(tmp)
            (data_dir / "car_001").mkdir()
            (data_dir / "car_002").mkdir()
            (data_dir / "empty_identity").mkdir()
            (data_dir / "car_001" / "a.jpg").write_bytes(b"fake")
            (data_dir / "car_001" / "b.png").write_bytes(b"fake")
            (data_dir / "car_002" / "c.jpeg").write_bytes(b"fake")

            identity_count, image_count = train_fastreid.inspect_reid_dataset(data_dir)

            self.assertEqual(identity_count, 2)
            self.assertEqual(image_count, 3)

    def test_prepare_reid_dataset_from_yolo_writes_class_identity_crops(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image_dir = root / "train" / "images"
            label_dir = root / "train" / "labels"
            output_dir = root / "datasets" / "vehicle_reid_bootstrap"
            image_dir.mkdir(parents=True)
            label_dir.mkdir(parents=True)

            image = np.zeros((100, 120, 3), dtype=np.uint8)
            image[25:75, 30:90] = (255, 255, 255)
            image_path = image_dir / "car_001.jpg"
            import cv2

            cv2.imwrite(str(image_path), image)
            (label_dir / "car_001.txt").write_text("0 0.5 0.5 0.5 0.5\n", encoding="utf-8")
            data_yaml = root / "data.yaml"
            data_yaml.write_text(
                "names:\n- Demo Vehicle\nnc: 1\ntrain: train/images\n",
                encoding="utf-8",
            )

            summary = prepare_reid_dataset_from_yolo.prepare_reid_dataset_from_yolo(
                data_yaml=data_yaml,
                output_dir=output_dir,
                splits=["train"],
                min_box_size=8,
            )

            crop_path = output_dir / "train" / "000_Demo_Vehicle" / "car_001_00.jpg"
            self.assertEqual(summary.identity_count, 1)
            self.assertEqual(summary.crop_count, 1)
            self.assertTrue(crop_path.exists())
            self.assertTrue((output_dir / "manifest.csv").exists())

    def test_prepare_fastreid_veri_dataset_exports_expected_layout(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source_dir = root / "datasets" / "vehicle_reid_bootstrap"
            output_dir = root / "datasets" / "fastreid" / "veri"
            for split in ("train", "val"):
                identity_dir = source_dir / split / "000_Demo_Vehicle"
                identity_dir.mkdir(parents=True)
                (identity_dir / f"{split}_001.jpg").write_bytes(b"fake")
                (identity_dir / f"{split}_002.jpg").write_bytes(b"fake")

            summary = prepare_fastreid_veri_dataset.export_identity_folders_to_veri(
                source_dir=source_dir,
                output_dir=output_dir,
            )

            self.assertEqual(summary.identity_count, 1)
            self.assertEqual(summary.train_count, 2)
            self.assertEqual(summary.query_count, 1)
            self.assertEqual(summary.gallery_count, 1)
            self.assertTrue((output_dir / "image_train" / "0001_c001_000001.jpg").exists())
            self.assertTrue((output_dir / "image_query" / "0001_c001_000001.jpg").exists())
            self.assertTrue((output_dir / "image_test" / "0001_c002_000001.jpg").exists())

    def test_build_vehicle_identity_candidates_writes_track_crops(self) -> None:
        class FakeDetector:
            def reset_tracking(self) -> None:
                pass

            def track(self, frame, tracker_config, camera_id, shot_id, frame_index, timestamp_ms):
                return [
                    VehicleDetection(
                        detection_id=0,
                        local_track_id=3,
                        frame_index=frame_index,
                        timestamp_ms=timestamp_ms,
                        label="demo_car",
                        confidence=0.91,
                        bbox=(10, 12, 28, 24),
                        center=(24, 24),
                    )
                ]

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source_dir = root / "frames"
            output_dir = root / "datasets" / "vehicle_identity_candidates"
            source_dir.mkdir()
            import cv2

            for index in range(2):
                image = np.zeros((80, 100, 3), dtype=np.uint8)
                image[12:36, 10:38] = (255, 255, 255)
                cv2.imwrite(str(source_dir / f"frame_{index:03d}.jpg"), image)

            summary = build_vehicle_identity_candidates.build_vehicle_identity_candidates(
                sources=[source_dir],
                output_dir=output_dir,
                model_path=Path("fake.pt"),
                tracker_config=Path("tracker.yaml"),
                detector=FakeDetector(),
                frame_stride=1,
                min_box_size=8,
                min_crops_per_track=1,
            )

            track_dir = output_dir / "cam_000_frames" / "track_0003"
            self.assertEqual(summary.crop_count, 2)
            self.assertEqual(summary.kept_track_count, 1)
            self.assertEqual(len(list(track_dir.glob("*.jpg"))), 2)
            self.assertTrue((output_dir / "manifest.csv").exists())
            self.assertTrue((output_dir / "track_summary.csv").exists())
            self.assertTrue((output_dir / "dataset_summary.json").exists())

    def test_curate_vehicle_identity_dataset_applies_mapping(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            candidates_dir = root / "datasets" / "vehicle_identity_candidates"
            track_dir = candidates_dir / "cam_000_main" / "track_0007"
            output_dir = root / "datasets" / "vehicle_identity"
            track_dir.mkdir(parents=True)
            (track_dir / "frame_000001.jpg").write_bytes(b"fake")
            (candidates_dir / "track_summary.csv").write_text(
                "camera_name,track_id,crop_count,first_frame_index,last_frame_index,mean_confidence,label,representative_crop,keep\n"
                "cam_000_main,7,1,1,1,0.95,demo,frame_000001.jpg,True\n",
                encoding="utf-8",
            )
            mapping_csv = candidates_dir / "identity_mapping_template.csv"

            row_count = curate_vehicle_identity_dataset.init_identity_mapping(candidates_dir, mapping_csv)
            self.assertEqual(row_count, 1)

            mapping_csv.write_text(
                "include,identity_id,split,camera_name,track_id,crop_count,first_frame_index,last_frame_index,mean_confidence,representative_crop,note\n"
                "1,vehicle_0001,train,cam_000_main,7,1,1,1,0.95,frame_000001.jpg,\n",
                encoding="utf-8",
            )
            summary = curate_vehicle_identity_dataset.curate_identity_dataset(
                candidates_dir=candidates_dir,
                mapping_csv=mapping_csv,
                output_dir=output_dir,
            )

            self.assertEqual(summary.identity_count, 1)
            self.assertEqual(summary.crop_count, 1)
            self.assertTrue((output_dir / "train" / "vehicle_0001" / "cam_000_main_track_0007_00001.jpg").exists())
            self.assertTrue((output_dir / "manifest.csv").exists())

    def test_auto_curate_vehicle_identity_tracks_selects_stable_tracks(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            candidates_root = root / "datasets" / "vehicle_identity_candidates_batch"
            segment_dir = candidates_root / "seg_000000_000060"
            track_dir = segment_dir / "cam_000_video" / "track_0003"
            short_track_dir = segment_dir / "cam_000_video" / "track_0004"
            output_dir = root / "datasets" / "vehicle_identity_auto"
            track_dir.mkdir(parents=True)
            short_track_dir.mkdir(parents=True)
            for index in range(6):
                (track_dir / f"frame_{index:06d}.jpg").write_bytes(b"fake")
            (short_track_dir / "frame_000000.jpg").write_bytes(b"fake")
            (segment_dir / "track_summary.csv").write_text(
                "camera_name,track_id,crop_count,first_frame_index,last_frame_index,mean_confidence,label,representative_crop,keep\n"
                f"cam_000_video,3,6,0,5,0.60,vehicle,{track_dir / 'frame_000000.jpg'},True\n"
                f"cam_000_video,4,1,0,0,0.90,vehicle,{short_track_dir / 'frame_000000.jpg'},True\n",
                encoding="utf-8",
            )

            summary = auto_curate_vehicle_identity_tracks.auto_curate_vehicle_identity_tracks(
                candidates_root=candidates_root,
                output_dir=output_dir,
                min_crops=5,
                min_mean_confidence=0.45,
                max_crops_per_identity=6,
                val_ratio=0.33,
            )

            self.assertEqual(summary.identity_count, 1)
            self.assertEqual(summary.train_count, 4)
            self.assertEqual(summary.val_count, 2)
            self.assertEqual(len(list((output_dir / "train" / "vehicle_0001").glob("*.jpg"))), 4)
            self.assertEqual(len(list((output_dir / "val" / "vehicle_0001").glob("*.jpg"))), 2)

    def test_cluster_vehicle_identity_tracks_merges_similar_track_embeddings(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            candidates_root = root / "datasets" / "vehicle_identity_candidates_batch"
            output_dir = root / "datasets" / "vehicle_identity_global"

            from PIL import Image

            rows = []
            for segment, track_id, frame_start, color in (
                ("seg_000000_000060", 1, 0, (255, 0, 0)),
                ("seg_000120_000180", 2, 120, (250, 0, 0)),
                ("seg_000240_000300", 3, 240, (0, 0, 255)),
            ):
                track_dir = candidates_root / segment / "cam_000_video" / f"track_{track_id:04d}"
                track_dir.mkdir(parents=True)
                for index in range(5):
                    Image.new("RGB", (32, 20), color).save(track_dir / f"frame_{frame_start + index:06d}.jpg")
                representative_crop = track_dir / f"frame_{frame_start:06d}.jpg"
                rows.append(
                    f"cam_000_video,{track_id},5,{frame_start},{frame_start + 4},0.80,vehicle,{representative_crop},True\n"
                )

            for segment in ("seg_000000_000060", "seg_000120_000180", "seg_000240_000300"):
                summary_path = candidates_root / segment / "track_summary.csv"
                segment_rows = [
                    row
                    for row in rows
                    if f"track_{int(row.split(',')[1]):04d}" in row and segment in row
                ]
                summary_path.write_text(
                    "camera_name,track_id,crop_count,first_frame_index,last_frame_index,mean_confidence,label,representative_crop,keep\n"
                    + "".join(segment_rows),
                    encoding="utf-8",
                )

            def fake_extractor(paths: list[Path]) -> np.ndarray:
                features = []
                for path in paths:
                    if "track_0001" in str(path) or "track_0002" in str(path):
                        features.append(np.array([1.0, 0.0], dtype="float32"))
                    else:
                        features.append(np.array([0.0, 1.0], dtype="float32"))
                return np.stack(features)

            manual_merge_csv = root / "manual_global_identity.csv"
            manual_merge_csv.write_text(
                "include,global_identity_id,segment,camera_name,track_id,note\n"
                "1,reviewed_vehicle_0001,seg_000000_000060,cam_000_video,1,\n"
                "1,reviewed_vehicle_0001,seg_000120_000180,cam_000_video,2,\n",
                encoding="utf-8",
            )
            summary = cluster_vehicle_identity_tracks.cluster_vehicle_identity_tracks(
                candidates_root=candidates_root,
                output_dir=output_dir,
                model_path=None,
                similarity_threshold=1.10,
                min_crops=2,
                min_mean_confidence=0.45,
                max_crops_per_track=3,
                max_crops_per_global_identity=5,
                feature_crops_per_track=2,
                val_ratio=0.33,
                manual_merge_csv=manual_merge_csv,
                feature_extractor=fake_extractor,
            )

            self.assertEqual(summary.selected_track_count, 3)
            self.assertEqual(summary.global_identity_count, 2)
            self.assertEqual(summary.crop_count, 8)
            self.assertEqual(summary.train_count, 5)
            self.assertEqual(summary.val_count, 3)
            self.assertEqual(summary.max_crops_per_global_identity, 5)
            self.assertTrue((output_dir / "train" / "reviewed_vehicle_0001").exists())
            cluster_rows = (output_dir / "cluster_summary.csv").read_text(encoding="utf-8")
            self.assertIn("reviewed_vehicle_0001", cluster_rows)
            self.assertIn("global_vehicle_0001", cluster_rows)

    def test_prepare_yolo_dataset_from_identity_crops_writes_full_box_labels(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source_dir = root / "datasets" / "vehicle_identity"
            output_dir = root / "datasets" / "yolo_vehicle"
            for split in ("train", "val"):
                identity_dir = source_dir / split / "vehicle_0001"
                identity_dir.mkdir(parents=True)
                (identity_dir / f"{split}_001.jpg").write_bytes(b"fake")

            summary = prepare_yolo_dataset_from_identity_crops.prepare_yolo_dataset_from_identity_crops(
                source_dir=source_dir,
                output_dir=output_dir,
            )

            self.assertEqual(summary.train_count, 1)
            self.assertEqual(summary.val_count, 1)
            self.assertTrue((output_dir / "data.yaml").exists())
            label_path = output_dir / "train" / "labels" / "vehicle_0001_train_001.txt"
            self.assertEqual(label_path.read_text(encoding="utf-8").strip(), "0 0.5 0.5 1.0 1.0")

    def test_fastreid_onnx_wrapper_scales_and_normalizes_embeddings(self) -> None:
        class FakeReID(torch.nn.Module):
            def forward(self, images):
                return torch.stack((images.mean(dim=(1, 2, 3)), torch.ones(images.shape[0])), dim=1)

        wrapper = export_fastreid_reid_onnx.UltralyticsReIDWrapper(FakeReID())
        output = wrapper(torch.ones(2, 3, 4, 4))

        self.assertEqual(tuple(output.shape), (2, 2))
        np.testing.assert_allclose(output.norm(dim=1).detach().numpy(), np.ones(2), rtol=1e-5)


if __name__ == "__main__":
    unittest.main()
