from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
from PIL import Image, ImageDraw, ImageFont


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
FALSE_VALUES = {"", "0", "false", "no", "n", "skip"}
TRUE_VALUES = {"1", "true", "yes", "y", "keep"}
FeatureExtractor = Callable[[list[Path]], np.ndarray]


@dataclass
class TrackCandidate:
    segment: str
    segment_dir: str
    camera_name: str
    track_id: int
    crop_count: int
    first_frame_index: int
    last_frame_index: int
    mean_confidence: float
    label: str
    representative_crop: str
    crop_paths: list[str]


@dataclass
class GlobalTrackAssignment:
    global_identity_id: str
    cluster_index: int
    segment: str
    camera_name: str
    track_id: int
    crop_count: int
    selected_crop_count: int
    train_count: int
    val_count: int
    mean_confidence: float
    label: str
    max_cluster_similarity: float
    representative_crop: str


@dataclass
class GlobalIdentitySummary:
    candidates_root: str
    output_dir: str
    model_path: str
    manual_merge_csv: str = ""
    candidate_track_count: int = 0
    selected_track_count: int = 0
    global_identity_count: int = 0
    crop_count: int = 0
    train_count: int = 0
    val_count: int = 0
    similarity_threshold: float = 0.82
    min_crops: int = 5
    min_mean_confidence: float = 0.45
    max_crops_per_track: int = 40
    feature_crops_per_track: int = 8
    assignments: list[GlobalTrackAssignment] = field(default_factory=list)
    note: str = (
        "Tracks with similar ReID embeddings are merged into global vehicle identities. "
        "Review the contact sheet and cluster_summary.csv before using the dataset as final supervision."
    )


class UnionFind:
    def __init__(self, size: int) -> None:
        self.parent = list(range(size))

    def find(self, item: int) -> int:
        while self.parent[item] != item:
            self.parent[item] = self.parent[self.parent[item]]
            item = self.parent[item]
        return item

    def union(self, left: int, right: int) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root != right_root:
            self.parent[right_root] = left_root


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Cluster candidate vehicle tracks into global identity folders")
    parser.add_argument("--candidates-root", type=str, default="datasets/vehicle_identity_candidates_videoplayback_batch")
    parser.add_argument("--output-dir", type=str, default="datasets/vehicle_identity_videoplayback_global")
    parser.add_argument("--model", type=str, default="weights/fastreid_videoplayback_auto/fastreid_vehicle_reid.torchscript")
    parser.add_argument("--similarity-threshold", type=float, default=0.82)
    parser.add_argument("--min-crops", type=int, default=5)
    parser.add_argument("--min-mean-confidence", type=float, default=0.45)
    parser.add_argument("--max-crops-per-track", type=int, default=40)
    parser.add_argument("--feature-crops-per-track", type=int, default=8)
    parser.add_argument("--val-ratio", type=float, default=0.25)
    parser.add_argument("--imgsz", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", type=str, default="auto", help="auto, cpu, cuda, or cuda:0")
    parser.add_argument("--overlap-margin", type=int, default=0, help="Do not merge overlapping tracks in the same segment/camera")
    parser.add_argument("--manual-merge-csv", type=str, default=None, help="Optional reviewed CSV: global_identity_id,segment,camera_name,track_id")
    parser.add_argument("--contact-sheet", type=str, default="runs/eval/vehicle_identity_videoplayback_global_contact_sheet.jpg")
    parser.add_argument("--clear-output", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    summary = cluster_vehicle_identity_tracks(
        candidates_root=Path(args.candidates_root),
        output_dir=Path(args.output_dir),
        model_path=Path(args.model),
        similarity_threshold=args.similarity_threshold,
        min_crops=args.min_crops,
        min_mean_confidence=args.min_mean_confidence,
        max_crops_per_track=args.max_crops_per_track,
        feature_crops_per_track=args.feature_crops_per_track,
        val_ratio=args.val_ratio,
        imgsz=args.imgsz,
        batch_size=args.batch_size,
        device=args.device,
        overlap_margin=args.overlap_margin,
        manual_merge_csv=Path(args.manual_merge_csv) if args.manual_merge_csv else None,
        contact_sheet=Path(args.contact_sheet),
        clear_output=args.clear_output,
        dry_run=args.dry_run,
    )
    print(json.dumps(asdict(summary), indent=2))
    return 0 if summary.crop_count > 0 or args.dry_run else 1


def cluster_vehicle_identity_tracks(
    candidates_root: Path,
    output_dir: Path,
    model_path: Path | None = None,
    similarity_threshold: float = 0.82,
    min_crops: int = 5,
    min_mean_confidence: float = 0.45,
    max_crops_per_track: int = 40,
    feature_crops_per_track: int = 8,
    val_ratio: float = 0.25,
    imgsz: int = 224,
    batch_size: int = 64,
    device: str = "auto",
    overlap_margin: int = 0,
    manual_merge_csv: Path | None = None,
    contact_sheet: Path | None = None,
    clear_output: bool = False,
    dry_run: bool = False,
    feature_extractor: FeatureExtractor | None = None,
) -> GlobalIdentitySummary:
    candidates_root = Path(candidates_root)
    output_dir = Path(output_dir)
    model_path = Path(model_path) if model_path else Path("")
    if not candidates_root.exists():
        raise FileNotFoundError(f"Candidates root not found: {candidates_root}")
    if feature_extractor is None:
        if not model_path.exists():
            raise FileNotFoundError(f"ReID model not found: {model_path}")
        feature_extractor = TorchScriptFeatureExtractor(model_path, imgsz=imgsz, batch_size=batch_size, device=device)

    all_tracks = load_track_candidates(candidates_root)
    selected_tracks = [
        track
        for track in all_tracks
        if track.crop_count >= min_crops
        and track.mean_confidence >= min_mean_confidence
        and len(track.crop_paths) >= min_crops
    ]

    if clear_output and output_dir.exists() and not dry_run:
        remove_output_dir(output_dir)
    if not dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)

    if not selected_tracks:
        summary = GlobalIdentitySummary(
            candidates_root=str(candidates_root),
            output_dir=str(output_dir),
            model_path=str(model_path),
            manual_merge_csv=str(manual_merge_csv or ""),
            candidate_track_count=len(all_tracks),
            min_crops=min_crops,
            min_mean_confidence=min_mean_confidence,
            max_crops_per_track=max_crops_per_track,
            feature_crops_per_track=feature_crops_per_track,
            similarity_threshold=similarity_threshold,
        )
        if not dry_run:
            write_outputs(output_dir, summary, [], [], np.zeros((0, 0), dtype=np.float32), contact_sheet)
        return summary

    embeddings = encode_track_embeddings(selected_tracks, feature_extractor, feature_crops_per_track)
    similarities = embeddings @ embeddings.T
    clusters = cluster_tracks(selected_tracks, similarities, similarity_threshold, overlap_margin)
    manual_identity_by_index = load_manual_identity_assignments(manual_merge_csv, selected_tracks) if manual_merge_csv else {}
    if manual_identity_by_index:
        clusters = apply_manual_merges(len(selected_tracks), clusters, manual_identity_by_index)
    clusters = sorted(clusters, key=lambda members: sort_key(selected_tracks[members[0]]))

    manifest_rows: list[dict[str, str]] = []
    cluster_rows: list[dict[str, str]] = []
    assignments: list[GlobalTrackAssignment] = []
    train_count = 0
    val_count = 0

    used_identity_ids: set[str] = set()
    next_auto_identity_index = 1
    for cluster_index, members in enumerate(clusters, start=1):
        global_identity_id, next_auto_identity_index = choose_global_identity_id(
            members,
            manual_identity_by_index,
            used_identity_ids,
            next_auto_identity_index,
        )
        used_identity_ids.add(global_identity_id)
        cluster_crop_paths: list[tuple[int, Path]] = []
        for member_index in members:
            track = selected_tracks[member_index]
            sampled_paths = evenly_sample([Path(path) for path in track.crop_paths], max_crops_per_track)
            cluster_crop_paths.extend((member_index, crop_path) for crop_path in sampled_paths)

        train_items, val_items = split_train_val(cluster_crop_paths, val_ratio)
        per_track_counts: dict[tuple[int, str], int] = {}
        for split, items in (("train", train_items), ("val", val_items)):
            for member_index, crop_path in items:
                track = selected_tracks[member_index]
                key = (member_index, split)
                per_track_counts[key] = per_track_counts.get(key, 0) + 1
                copy_index = per_track_counts[key]
                output_name = (
                    f"{track.segment}_{track.camera_name}_track_{track.track_id:04d}_"
                    f"{copy_index:05d}{crop_path.suffix.lower()}"
                )
                output_path = output_dir / split / global_identity_id / output_name
                if not dry_run:
                    output_path.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(crop_path, output_path)
                manifest_rows.append(
                    {
                        "global_identity_id": global_identity_id,
                        "split": split,
                        "segment": track.segment,
                        "camera_name": track.camera_name,
                        "track_id": str(track.track_id),
                        "mean_confidence": f"{track.mean_confidence:.6f}",
                        "label": track.label,
                        "source_crop": str(crop_path),
                        "output_crop": str(output_path),
                    }
                )
                if split == "train":
                    train_count += 1
                else:
                    val_count += 1

        for member_index in members:
            track = selected_tracks[member_index]
            same_cluster_scores = [
                float(similarities[member_index, other_index])
                for other_index in members
                if other_index != member_index
            ]
            selected_crop_count = sum(1 for item_member, _ in cluster_crop_paths if item_member == member_index)
            assignment = GlobalTrackAssignment(
                global_identity_id=global_identity_id,
                cluster_index=cluster_index,
                segment=track.segment,
                camera_name=track.camera_name,
                track_id=track.track_id,
                crop_count=track.crop_count,
                selected_crop_count=selected_crop_count,
                train_count=per_track_counts.get((member_index, "train"), 0),
                val_count=per_track_counts.get((member_index, "val"), 0),
                mean_confidence=track.mean_confidence,
                label=track.label,
                max_cluster_similarity=max(same_cluster_scores, default=0.0),
                representative_crop=track.representative_crop,
            )
            assignments.append(assignment)
            cluster_rows.append({key: str(value) for key, value in asdict(assignment).items()})

    summary = GlobalIdentitySummary(
        candidates_root=str(candidates_root),
        output_dir=str(output_dir),
        model_path=str(model_path),
        manual_merge_csv=str(manual_merge_csv or ""),
        candidate_track_count=len(all_tracks),
        selected_track_count=len(selected_tracks),
        global_identity_count=len(clusters),
        crop_count=len(manifest_rows),
        train_count=train_count,
        val_count=val_count,
        similarity_threshold=similarity_threshold,
        min_crops=min_crops,
        min_mean_confidence=min_mean_confidence,
        max_crops_per_track=max_crops_per_track,
        feature_crops_per_track=feature_crops_per_track,
        assignments=assignments,
    )

    if not dry_run:
        write_outputs(output_dir, summary, manifest_rows, cluster_rows, similarities, contact_sheet)
    return summary


class TorchScriptFeatureExtractor:
    def __init__(self, model_path: Path, imgsz: int, batch_size: int, device: str) -> None:
        import torch
        import torch.nn.functional as F

        self.torch = torch
        self.functional = F
        self.imgsz = imgsz
        self.batch_size = batch_size
        if device == "auto":
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        elif device == "cuda":
            self.device = torch.device("cuda:0")
        else:
            self.device = torch.device(device)
        self.model = torch.jit.load(str(model_path), map_location=self.device).eval().to(self.device)

    def __call__(self, crop_paths: list[Path]) -> np.ndarray:
        outputs = []
        for start in range(0, len(crop_paths), self.batch_size):
            batch_paths = crop_paths[start : start + self.batch_size]
            images = self.torch.stack([self._load_image_tensor(path) for path in batch_paths]).to(self.device)
            with self.torch.no_grad():
                features = self.model(images)
                features = self.functional.normalize(features, p=2, dim=1)
                outputs.append(features.detach().cpu())
        if not outputs:
            return np.zeros((0, 1), dtype=np.float32)
        return self.torch.cat(outputs, dim=0).numpy().astype("float32")

    def _load_image_tensor(self, path: Path):
        image = Image.open(path).convert("RGB").resize((self.imgsz, self.imgsz), Image.BILINEAR)
        data = self.torch.frombuffer(bytearray(image.tobytes()), dtype=self.torch.uint8)
        data = data.view(self.imgsz, self.imgsz, 3).permute(2, 0, 1).contiguous()
        return data.float().div(255.0)


def load_track_candidates(candidates_root: Path) -> list[TrackCandidate]:
    summary_paths = sorted(candidates_root.glob("seg_*/track_summary.csv"))
    if not summary_paths and (candidates_root / "track_summary.csv").exists():
        summary_paths = [candidates_root / "track_summary.csv"]

    tracks: list[TrackCandidate] = []
    for summary_path in summary_paths:
        segment_dir = summary_path.parent
        segment = segment_dir.name
        with open(summary_path, "r", encoding="utf-8", newline="") as file:
            for row in csv.DictReader(file):
                camera_name = str(row.get("camera_name", "")).strip()
                track_id = safe_int(row.get("track_id", "0"))
                track_dir = segment_dir / camera_name / f"track_{track_id:04d}"
                crop_paths = sorted(path for path in track_dir.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES) if track_dir.exists() else []
                tracks.append(
                    TrackCandidate(
                        segment=segment,
                        segment_dir=str(segment_dir),
                        camera_name=camera_name,
                        track_id=track_id,
                        crop_count=safe_int(row.get("crop_count", "0")),
                        first_frame_index=safe_int(row.get("first_frame_index", "0")),
                        last_frame_index=safe_int(row.get("last_frame_index", "0")),
                        mean_confidence=safe_float(row.get("mean_confidence", "0")),
                        label=str(row.get("label", "")),
                        representative_crop=str(row.get("representative_crop", "")),
                        crop_paths=[str(path) for path in crop_paths],
                    )
                )
    return sorted(tracks, key=sort_key)


def encode_track_embeddings(
    tracks: list[TrackCandidate],
    feature_extractor: FeatureExtractor,
    feature_crops_per_track: int,
) -> np.ndarray:
    embeddings: list[np.ndarray] = []
    for track in tracks:
        crop_paths = evenly_sample([Path(path) for path in track.crop_paths], feature_crops_per_track)
        features = feature_extractor(crop_paths)
        if features.ndim != 2 or features.shape[0] == 0:
            raise RuntimeError(f"Feature extractor returned no features for {track.segment}/{track.camera_name}/{track.track_id}")
        features = normalize_rows(features)
        track_embedding = normalize_vector(features.mean(axis=0))
        embeddings.append(track_embedding)
    return np.stack(embeddings).astype("float32")


def cluster_tracks(
    tracks: list[TrackCandidate],
    similarities: np.ndarray,
    similarity_threshold: float,
    overlap_margin: int,
) -> list[list[int]]:
    uf = UnionFind(len(tracks))
    pairs: list[tuple[float, int, int]] = []
    for left in range(len(tracks)):
        for right in range(left + 1, len(tracks)):
            score = float(similarities[left, right])
            if score >= similarity_threshold:
                pairs.append((score, left, right))
    pairs.sort(reverse=True)

    for _, left, right in pairs:
        left_root = uf.find(left)
        right_root = uf.find(right)
        if left_root == right_root:
            continue
        left_members = [index for index in range(len(tracks)) if uf.find(index) == left_root]
        right_members = [index for index in range(len(tracks)) if uf.find(index) == right_root]
        if clusters_can_merge(tracks, left_members, right_members, overlap_margin):
            uf.union(left_root, right_root)

    clusters_by_root: dict[int, list[int]] = {}
    for index in range(len(tracks)):
        clusters_by_root.setdefault(uf.find(index), []).append(index)
    return [sorted(members, key=lambda index: sort_key(tracks[index])) for members in clusters_by_root.values()]


def load_manual_identity_assignments(manual_merge_csv: Path | None, tracks: list[TrackCandidate]) -> dict[int, str]:
    if manual_merge_csv is None:
        return {}
    if not manual_merge_csv.exists():
        raise FileNotFoundError(f"Manual merge CSV not found: {manual_merge_csv}")

    track_index = {track_key(track): index for index, track in enumerate(tracks)}
    assignments: dict[int, str] = {}
    with open(manual_merge_csv, "r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        for row in reader:
            if not parse_bool(row.get("include", "1"), default=True):
                continue
            identity_id = safe_identity_id(str(row.get("global_identity_id", "")))
            if not identity_id:
                continue
            key = (
                str(row.get("segment", "")).strip(),
                str(row.get("camera_name", "")).strip(),
                safe_int(row.get("track_id", "0")),
            )
            if key in track_index:
                assignments[track_index[key]] = identity_id
    return assignments


def apply_manual_merges(
    track_count: int,
    clusters: list[list[int]],
    manual_identity_by_index: dict[int, str],
) -> list[list[int]]:
    uf = UnionFind(track_count)
    for members in clusters:
        if not members:
            continue
        first = members[0]
        for member in members[1:]:
            uf.union(first, member)

    manual_groups: dict[str, list[int]] = {}
    for index, identity_id in manual_identity_by_index.items():
        manual_groups.setdefault(identity_id, []).append(index)
    for members in manual_groups.values():
        first = members[0]
        for member in members[1:]:
            uf.union(first, member)

    clusters_by_root: dict[int, list[int]] = {}
    for index in range(track_count):
        clusters_by_root.setdefault(uf.find(index), []).append(index)
    return [members for members in clusters_by_root.values()]


def choose_global_identity_id(
    members: list[int],
    manual_identity_by_index: dict[int, str],
    used_identity_ids: set[str],
    next_auto_identity_index: int,
) -> tuple[str, int]:
    manual_ids = sorted({manual_identity_by_index[index] for index in members if index in manual_identity_by_index})
    if manual_ids:
        identity_id = manual_ids[0]
        if identity_id not in used_identity_ids:
            return identity_id, next_auto_identity_index

    while True:
        identity_id = f"global_vehicle_{next_auto_identity_index:04d}"
        next_auto_identity_index += 1
        if identity_id not in used_identity_ids:
            return identity_id, next_auto_identity_index


def clusters_can_merge(
    tracks: list[TrackCandidate],
    left_members: list[int],
    right_members: list[int],
    overlap_margin: int,
) -> bool:
    for left_index in left_members:
        for right_index in right_members:
            if tracks_overlap(tracks[left_index], tracks[right_index], overlap_margin):
                return False
    return True


def tracks_overlap(left: TrackCandidate, right: TrackCandidate, overlap_margin: int) -> bool:
    if left.segment != right.segment or left.camera_name != right.camera_name:
        return False
    left_start = left.first_frame_index - overlap_margin
    left_end = left.last_frame_index + overlap_margin
    right_start = right.first_frame_index - overlap_margin
    right_end = right.last_frame_index + overlap_margin
    return max(left_start, right_start) <= min(left_end, right_end)


def evenly_sample(paths: list[Path], max_items: int) -> list[Path]:
    if max_items <= 0 or len(paths) <= max_items:
        return paths
    if max_items == 1:
        return [paths[0]]
    step = (len(paths) - 1) / (max_items - 1)
    return [paths[round(index * step)] for index in range(max_items)]


def track_key(track: TrackCandidate) -> tuple[str, str, int]:
    return (track.segment, track.camera_name, track.track_id)


def split_train_val(items: list[tuple[int, Path]], val_ratio: float) -> tuple[list[tuple[int, Path]], list[tuple[int, Path]]]:
    if len(items) <= 1:
        return items, []
    val_count = max(1, round(len(items) * val_ratio))
    if len(items) - val_count < 1:
        val_count = len(items) - 1
    val_indices = set(round(index * (len(items) - 1) / max(1, val_count - 1)) for index in range(val_count))
    train_items = [item for index, item in enumerate(items) if index not in val_indices]
    val_items = [item for index, item in enumerate(items) if index in val_indices]
    return train_items, val_items


def normalize_rows(values: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return values / norms


def normalize_vector(value: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(value))
    if norm == 0:
        return value.astype("float32")
    return (value / norm).astype("float32")


def write_outputs(
    output_dir: Path,
    summary: GlobalIdentitySummary,
    manifest_rows: list[dict[str, str]],
    cluster_rows: list[dict[str, str]],
    similarities: np.ndarray,
    contact_sheet: Path | None,
) -> None:
    write_manifest(output_dir / "manifest.csv", manifest_rows)
    write_cluster_summary(output_dir / "cluster_summary.csv", cluster_rows)
    (output_dir / "dataset_summary.json").write_text(json.dumps(asdict(summary), indent=2), encoding="utf-8")
    np.save(output_dir / "track_similarity.npy", similarities.astype("float32"))
    write_readme(output_dir / "README.md", summary)
    if contact_sheet:
        write_contact_sheet(contact_sheet, summary.assignments)


def write_manifest(path: Path, rows: list[dict[str, str]]) -> None:
    fieldnames = [
        "global_identity_id",
        "split",
        "segment",
        "camera_name",
        "track_id",
        "mean_confidence",
        "label",
        "source_crop",
        "output_crop",
    ]
    with open(path, "w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_cluster_summary(path: Path, rows: list[dict[str, str]]) -> None:
    fieldnames = list(asdict(GlobalTrackAssignment("", 0, "", "", 0, 0, 0, 0, 0, 0.0, "", 0.0, "")).keys())
    with open(path, "w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_readme(path: Path, summary: GlobalIdentitySummary) -> None:
    path.write_text(
        "\n".join(
            [
                "# Global Vehicle Identity Dataset",
                "",
                "Candidate tracker fragments are merged with ReID embedding similarity.",
                "Review cluster_summary.csv and the contact sheet before treating clusters as final labels.",
                "",
                f"Global identities: {summary.global_identity_count}",
                f"Selected tracks: {summary.selected_track_count}",
                f"Train crops: {summary.train_count}",
                f"Val crops: {summary.val_count}",
                f"Similarity threshold: {summary.similarity_threshold}",
                "",
            ]
        ),
        encoding="utf-8",
    )


def write_contact_sheet(path: Path, assignments: list[GlobalTrackAssignment], max_items: int = 80) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    items = assignments[:max_items]
    if not items:
        Image.new("RGB", (400, 120), "white").save(path)
        return

    thumb_w, thumb_h = 190, 120
    label_h = 44
    pad = 10
    columns = 4
    rows = (len(items) + columns - 1) // columns
    sheet = Image.new("RGB", (columns * (thumb_w + pad) + pad, rows * (thumb_h + label_h + pad) + pad), "white")
    draw = ImageDraw.Draw(sheet)
    try:
        font = ImageFont.truetype("arial.ttf", 13)
    except OSError:
        font = ImageFont.load_default()

    for index, assignment in enumerate(items):
        x = pad + (index % columns) * (thumb_w + pad)
        y = pad + (index // columns) * (thumb_h + label_h + pad)
        crop_path = Path(assignment.representative_crop)
        if not crop_path.is_absolute():
            crop_path = Path.cwd() / crop_path
        try:
            image = Image.open(crop_path).convert("RGB")
            image.thumbnail((thumb_w, thumb_h), Image.LANCZOS)
            offset_x = x + (thumb_w - image.width) // 2
            offset_y = y + (thumb_h - image.height) // 2
            sheet.paste(image, (offset_x, offset_y))
        except Exception as exc:
            draw.rectangle([x, y, x + thumb_w, y + thumb_h], outline="red")
            draw.text((x + 4, y + 4), f"load failed\n{exc}", fill="red", font=font)
        draw.rectangle([x, y, x + thumb_w, y + thumb_h], outline=(210, 210, 210))
        label = (
            f"{assignment.global_identity_id} t{assignment.track_id} "
            f"n={assignment.selected_crop_count} sim={assignment.max_cluster_similarity:.2f}"
        )
        draw.text((x, y + thumb_h + 4), label, fill="black", font=font)
    sheet.save(path, quality=92)


def safe_int(value: str | int | None) -> int:
    try:
        return int(float(str(value)))
    except (TypeError, ValueError):
        return 0


def safe_float(value: str | float | None) -> float:
    try:
        return float(str(value))
    except (TypeError, ValueError):
        return 0.0


def parse_bool(value: str | None, default: bool) -> bool:
    if value is None:
        return default
    text = str(value).strip().lower()
    if text in TRUE_VALUES:
        return True
    if text in FALSE_VALUES:
        return False
    return default


def safe_identity_id(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9]+", "_", str(value).strip()).strip("_")
    return re.sub(r"_+", "_", cleaned)


def sort_key(track: TrackCandidate) -> tuple[str, str, int, int]:
    return (track.segment, track.camera_name, track.first_frame_index, track.track_id)


def remove_output_dir(output_dir: Path) -> None:
    resolved = output_dir.resolve()
    if "datasets" not in resolved.parts:
        raise ValueError(f"Refusing to clear output outside a datasets folder: {resolved}")
    shutil.rmtree(resolved)


if __name__ == "__main__":
    raise SystemExit(main())
