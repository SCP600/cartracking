from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable, Iterator

import cv2

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from autocam_tracker.app.app_controller import AppController
from autocam_tracker.app.app_state import AppConfig
from autocam_tracker.detection.detection_models import VehicleDetection
from autocam_tracker.detection.yolo26_detector import YOLO26Detector


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VIDEO_SUFFIXES = {".mp4", ".mov", ".avi", ".mkv", ".m4v", ".wmv"}


@dataclass
class MediaSource:
    path: str
    kind: str
    camera_name: str


@dataclass
class TrackSummary:
    camera_name: str
    track_id: int
    crop_count: int = 0
    first_frame_index: int = -1
    last_frame_index: int = -1
    mean_confidence: float = 0.0
    label: str = "vehicle"
    representative_crop: str = ""
    keep: bool = True


@dataclass
class CandidateBuildSummary:
    output_dir: str
    model_path: str
    tracker_config: str
    source_count: int = 0
    frame_count: int = 0
    detection_count: int = 0
    crop_count: int = 0
    skipped_count: int = 0
    track_count: int = 0
    kept_track_count: int = 0
    removed_track_count: int = 0
    sources: list[MediaSource] = field(default_factory=list)
    note: str = (
        "Tracks are candidate same-vehicle groups inside each source. "
        "Manually merge same vehicles across cameras or clips before FastReID identity training."
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build vehicle identity candidate crops from videos or image folders")
    parser.add_argument("sources", nargs="+", help="Video files, image files, or image folders")
    parser.add_argument("--project-root", type=str, default=".", help="Project root containing autocam_tracker/config")
    parser.add_argument("--output-dir", type=str, default="datasets/vehicle_identity_candidates")
    parser.add_argument("--model", type=str, default=None, help="YOLO model path. Defaults to default_config.json")
    parser.add_argument("--tracker", type=str, default=None, help="Tracker name. Defaults to default_config.json")
    parser.add_argument("--tracker-config", type=str, default=None, help="Explicit Ultralytics tracker YAML")
    parser.add_argument("--conf", type=float, default=None, help="Detection confidence. Defaults to default_config.json")
    parser.add_argument("--imgsz", type=int, default=None, help="YOLO image size. Defaults to default_config.json")
    parser.add_argument("--device", type=str, default=None, help="Device override, e.g. 0, cuda:0, or cpu")
    parser.add_argument("--frame-stride", type=int, default=5, help="Process every Nth frame/image")
    parser.add_argument("--save-stride", type=int, default=1, help="Save crops every N processed source frames")
    parser.add_argument("--start-frame", type=int, default=0, help="Start reading each source at this frame index")
    parser.add_argument("--end-frame", type=int, default=0, help="Stop before this frame index. 0 means no limit")
    parser.add_argument("--max-frames", type=int, default=0, help="0 means process every sampled frame")
    parser.add_argument("--padding", type=float, default=0.08, help="BBox padding ratio before saving crops")
    parser.add_argument("--min-box-size", type=int, default=24, help="Skip boxes smaller than this width/height")
    parser.add_argument("--max-crops-per-track", type=int, default=80, help="0 means keep every crop")
    parser.add_argument("--min-crops-per-track", type=int, default=3, help="Remove candidate tracks with fewer crops")
    parser.add_argument("--jpeg-quality", type=int, default=92)
    parser.add_argument("--include-untracked", action="store_true", help="Keep detections without tracker IDs")
    parser.add_argument("--clear-output", action="store_true", help="Remove the output folder before writing")
    parser.add_argument("--dry-run", action="store_true", help="List sources and write no crops")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    project_root = Path(args.project_root).resolve()
    app_config = AppConfig.from_project_root(project_root)
    model_path = _resolve_path(project_root, args.model) if args.model else app_config.model_path
    tracker_name = args.tracker or app_config.tracker
    tracker_config = _resolve_tracker_config(project_root, tracker_name, args.tracker_config)
    device = _parse_device(args.device) if args.device is not None else app_config.device

    detector = YOLO26Detector(
        model_path=model_path,
        conf=args.conf if args.conf is not None else app_config.conf,
        imgsz=args.imgsz if args.imgsz is not None else app_config.imgsz,
        vehicle_class_ids=app_config.vehicle_class_ids,
        device=device,
    )

    summary = build_vehicle_identity_candidates(
        sources=[Path(source) for source in args.sources],
        output_dir=Path(args.output_dir),
        model_path=model_path,
        tracker_config=tracker_config,
        detector=detector,
        frame_stride=args.frame_stride,
        save_stride=args.save_stride,
        start_frame=args.start_frame,
        end_frame=args.end_frame,
        max_frames=args.max_frames,
        padding=args.padding,
        min_box_size=args.min_box_size,
        max_crops_per_track=args.max_crops_per_track,
        min_crops_per_track=args.min_crops_per_track,
        include_untracked=args.include_untracked,
        jpeg_quality=args.jpeg_quality,
        clear_output=args.clear_output,
        dry_run=args.dry_run,
    )

    print(json.dumps(asdict(summary), indent=2))
    return 0 if summary.source_count > 0 and (args.dry_run or summary.crop_count > 0) else 1


def build_vehicle_identity_candidates(
    sources: list[Path],
    output_dir: Path,
    model_path: Path,
    tracker_config: Path,
    detector: YOLO26Detector,
    frame_stride: int = 5,
    save_stride: int = 1,
    start_frame: int = 0,
    end_frame: int = 0,
    max_frames: int = 0,
    padding: float = 0.08,
    min_box_size: int = 24,
    max_crops_per_track: int = 80,
    min_crops_per_track: int = 3,
    include_untracked: bool = False,
    jpeg_quality: int = 92,
    clear_output: bool = False,
    dry_run: bool = False,
) -> CandidateBuildSummary:
    output_dir = Path(output_dir)
    media_sources = discover_sources(sources)
    summary = CandidateBuildSummary(
        output_dir=str(output_dir),
        model_path=str(model_path),
        tracker_config=str(tracker_config),
        source_count=len(media_sources),
        sources=media_sources,
    )

    if dry_run:
        return summary

    if clear_output and output_dir.exists():
        _remove_output_dir(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest_rows: list[dict[str, object]] = []
    track_stats: dict[tuple[str, int], TrackSummary] = {}
    track_conf_sums: dict[tuple[str, int], float] = {}
    track_saved_counts: dict[tuple[str, int], int] = {}

    for source in media_sources:
        detector.reset_tracking()
        processed_frames = 0
        previous_frame_shape: tuple[int, int] | None = None
        track_id_offset = 0
        reset_segment = 0
        for frame_index, timestamp_ms, frame in iter_frames(
            source,
            frame_stride=max(1, frame_stride),
            max_frames=max_frames,
            start_frame=max(0, start_frame),
            end_frame=max(0, end_frame),
        ):
            frame_shape = frame.shape[:2]
            if previous_frame_shape is not None and frame_shape != previous_frame_shape:
                detector.reset_tracking()
                reset_segment += 1
                track_id_offset = reset_segment * 100000
            previous_frame_shape = frame_shape
            processed_frames += 1
            summary.frame_count += 1
            detections = detector.track(
                frame=frame,
                tracker_config=tracker_config,
                camera_id=0,
                shot_id=0,
                frame_index=frame_index,
                timestamp_ms=timestamp_ms,
            )
            summary.detection_count += len(detections)
            for detection in detections:
                if detection.local_track_id < 0 and not include_untracked:
                    summary.skipped_count += 1
                    continue

                raw_track_id = detection.local_track_id if detection.local_track_id >= 0 else 900000 + detection.detection_id
                track_id = track_id_offset + raw_track_id
                track_key = (source.camera_name, track_id)
                saved_count = track_saved_counts.get(track_key, 0)
                if save_stride > 1 and frame_index % save_stride != 0:
                    continue
                if max_crops_per_track > 0 and saved_count >= max_crops_per_track:
                    summary.skipped_count += 1
                    continue

                crop_box = padded_bbox(detection.bbox, frame.shape[1], frame.shape[0], padding)
                x, y, w, h = crop_box
                if w < min_box_size or h < min_box_size:
                    summary.skipped_count += 1
                    continue

                track_dir = output_dir / source.camera_name / f"track_{track_id:04d}"
                track_dir.mkdir(parents=True, exist_ok=True)
                crop_path = track_dir / f"frame_{frame_index:06d}_det_{detection.detection_id:02d}_conf_{detection.confidence:.3f}.jpg"
                crop = frame[y : y + h, x : x + w]
                if crop.size == 0:
                    summary.skipped_count += 1
                    continue
                cv2.imwrite(str(crop_path), crop, [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)])

                track_saved_counts[track_key] = saved_count + 1
                summary.crop_count += 1
                stats = track_stats.setdefault(
                    track_key,
                    TrackSummary(
                        camera_name=source.camera_name,
                        track_id=track_id,
                        first_frame_index=frame_index,
                        representative_crop=str(crop_path),
                    ),
                )
                stats.crop_count += 1
                stats.last_frame_index = frame_index
                stats.label = detection.label
                track_conf_sums[track_key] = track_conf_sums.get(track_key, 0.0) + detection.confidence

                manifest_rows.append(
                    {
                        "camera_name": source.camera_name,
                        "source_path": source.path,
                        "source_kind": source.kind,
                        "frame_index": frame_index,
                        "timestamp_ms": f"{timestamp_ms:.3f}",
                        "track_id": track_id,
                        "local_track_id": detection.local_track_id,
                        "detection_id": detection.detection_id,
                        "label": detection.label,
                        "confidence": f"{detection.confidence:.6f}",
                        "bbox_x": detection.bbox[0],
                        "bbox_y": detection.bbox[1],
                        "bbox_w": detection.bbox[2],
                        "bbox_h": detection.bbox[3],
                        "crop_x": x,
                        "crop_y": y,
                        "crop_w": w,
                        "crop_h": h,
                        "crop_path": str(crop_path),
                    }
                )
        print(f"[INFO] {source.camera_name}: processed {processed_frames} frames")

    for track_key, stats in track_stats.items():
        stats.mean_confidence = track_conf_sums[track_key] / max(1, stats.crop_count)
        if stats.crop_count < max(1, min_crops_per_track):
            stats.keep = False

    kept_track_keys = {track_key for track_key, stats in track_stats.items() if stats.keep}
    filtered_rows = [row for row in manifest_rows if (str(row["camera_name"]), int(row["track_id"])) in kept_track_keys]
    _remove_rejected_track_dirs(output_dir, track_stats)

    summary.track_count = len(track_stats)
    summary.kept_track_count = len(kept_track_keys)
    summary.removed_track_count = summary.track_count - summary.kept_track_count
    summary.crop_count = len(filtered_rows)
    summary.skipped_count += len(manifest_rows) - len(filtered_rows)

    _write_manifest(output_dir / "manifest.csv", filtered_rows)
    _write_track_summaries(output_dir / "track_summary.csv", list(track_stats.values()))
    (output_dir / "dataset_summary.json").write_text(json.dumps(asdict(summary), indent=2), encoding="utf-8")
    _write_readme(output_dir / "README.md", summary)
    return summary


def discover_sources(paths: Iterable[Path]) -> list[MediaSource]:
    sources: list[MediaSource] = []
    for path in paths:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Source not found: {path}")
        if path.is_dir():
            image_paths = sorted(item for item in path.iterdir() if item.suffix.lower() in IMAGE_SUFFIXES)
            video_paths = sorted(item for item in path.iterdir() if item.suffix.lower() in VIDEO_SUFFIXES)
            if image_paths:
                sources.append(_media_source(path, "image_dir", len(sources)))
            for video_path in video_paths:
                sources.append(_media_source(video_path, "video", len(sources)))
            if not image_paths and not video_paths:
                raise FileNotFoundError(f"No supported images or videos found in: {path}")
            continue
        suffix = path.suffix.lower()
        if suffix in VIDEO_SUFFIXES:
            sources.append(_media_source(path, "video", len(sources)))
        elif suffix in IMAGE_SUFFIXES:
            sources.append(_media_source(path, "image", len(sources)))
        else:
            raise ValueError(f"Unsupported source type: {path}")
    return sources


def iter_frames(
    source: MediaSource,
    frame_stride: int,
    max_frames: int = 0,
    start_frame: int = 0,
    end_frame: int = 0,
) -> Iterator[tuple[int, float, object]]:
    path = Path(source.path)
    emitted = 0
    if source.kind == "video":
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video: {path}")
        frame_index = max(0, start_frame)
        if frame_index:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        try:
            while True:
                if end_frame > 0 and frame_index >= end_frame:
                    break
                ok, frame = cap.read()
                if not ok:
                    break
                timestamp_ms = float(cap.get(cv2.CAP_PROP_POS_MSEC))
                if frame_index % frame_stride == 0:
                    yield frame_index, timestamp_ms, frame
                    emitted += 1
                    if max_frames > 0 and emitted >= max_frames:
                        break
                frame_index += 1
        finally:
            cap.release()
        return

    if source.kind == "image_dir":
        image_paths = sorted(item for item in path.iterdir() if item.suffix.lower() in IMAGE_SUFFIXES)
    else:
        image_paths = [path]
    for frame_index, image_path in enumerate(image_paths):
        if frame_index < start_frame:
            continue
        if end_frame > 0 and frame_index >= end_frame:
            break
        if frame_index % frame_stride != 0:
            continue
        frame = cv2.imread(str(image_path))
        if frame is None:
            continue
        yield frame_index, 0.0, frame
        emitted += 1
        if max_frames > 0 and emitted >= max_frames:
            break


def padded_bbox(bbox: tuple[int, int, int, int], frame_w: int, frame_h: int, padding: float) -> tuple[int, int, int, int]:
    x, y, w, h = bbox
    pad_x = int(round(w * padding))
    pad_y = int(round(h * padding))
    x1 = max(0, x - pad_x)
    y1 = max(0, y - pad_y)
    x2 = min(frame_w, x + w + pad_x)
    y2 = min(frame_h, y + h + pad_y)
    return x1, y1, max(0, x2 - x1), max(0, y2 - y1)


def _media_source(path: Path, kind: str, index: int) -> MediaSource:
    return MediaSource(path=str(path), kind=kind, camera_name=f"cam_{index:03d}_{_safe_name(path.stem)}")


def _safe_name(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9]+", "_", value).strip("_")
    return re.sub(r"_+", "_", cleaned) or "source"


def _resolve_path(project_root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else project_root / path


def _resolve_tracker_config(project_root: Path, tracker_name: str, tracker_config: str | None) -> Path:
    if tracker_config:
        return _resolve_path(project_root, tracker_config)
    controller = AppController(project_root=project_root)
    return controller._tracker_config_path(tracker_name)


def _parse_device(value: str) -> int | str:
    text = value.strip()
    return int(text) if text.isdigit() else text


def _write_manifest(path: Path, rows: list[dict[str, object]]) -> None:
    fieldnames = [
        "camera_name",
        "source_path",
        "source_kind",
        "frame_index",
        "timestamp_ms",
        "track_id",
        "local_track_id",
        "detection_id",
        "label",
        "confidence",
        "bbox_x",
        "bbox_y",
        "bbox_w",
        "bbox_h",
        "crop_x",
        "crop_y",
        "crop_w",
        "crop_h",
        "crop_path",
    ]
    with open(path, "w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_track_summaries(path: Path, tracks: list[TrackSummary]) -> None:
    fieldnames = list(asdict(TrackSummary(camera_name="", track_id=0)).keys())
    with open(path, "w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for track in sorted(tracks, key=lambda item: (item.camera_name, item.track_id)):
            writer.writerow(asdict(track))


def _write_readme(path: Path, summary: CandidateBuildSummary) -> None:
    path.write_text(
        "\n".join(
            [
                "# Vehicle Identity Candidates",
                "",
                "This folder contains tracker-generated candidate vehicle tracks.",
                "",
                "Review the crop folders manually before training FastReID with true identities:",
                "",
                "1. Open each `cam_*/track_*` folder.",
                "2. Remove bad crops, false positives, and track fragments that changed vehicle.",
                "3. Copy or merge folders that are the same real vehicle into `datasets/vehicle_identity/train/vehicle_XXXX/`.",
                "4. Convert that curated identity folder dataset with `scripts/prepare_fastreid_veri_dataset.py`.",
                "",
                f"Sources: {summary.source_count}",
                f"Kept tracks: {summary.kept_track_count}",
                f"Crops: {summary.crop_count}",
                "",
            ]
        ),
        encoding="utf-8",
    )


def _remove_rejected_track_dirs(output_dir: Path, track_stats: dict[tuple[str, int], TrackSummary]) -> None:
    for stats in track_stats.values():
        if stats.keep:
            continue
        track_dir = output_dir / stats.camera_name / f"track_{stats.track_id:04d}"
        if track_dir.exists():
            shutil.rmtree(track_dir)


def _remove_output_dir(output_dir: Path) -> None:
    resolved = output_dir.resolve()
    if "datasets" not in resolved.parts:
        raise ValueError(f"Refusing to clear output outside a datasets folder: {resolved}")
    shutil.rmtree(resolved)


if __name__ == "__main__":
    raise SystemExit(main())
