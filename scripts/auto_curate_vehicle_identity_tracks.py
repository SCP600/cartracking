from __future__ import annotations

import argparse
import csv
import json
import shutil
from dataclasses import asdict, dataclass, field
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass
class SelectedTrack:
    identity_id: str
    segment: str
    camera_name: str
    track_id: int
    crop_count: int
    selected_crop_count: int
    train_count: int
    val_count: int
    mean_confidence: float
    label: str
    representative_crop: str


@dataclass
class AutoCurationSummary:
    candidates_root: str
    output_dir: str
    track_count: int = 0
    selected_track_count: int = 0
    identity_count: int = 0
    crop_count: int = 0
    train_count: int = 0
    val_count: int = 0
    min_crops: int = 5
    min_mean_confidence: float = 0.45
    max_crops_per_identity: int = 40
    selected_tracks: list[SelectedTrack] = field(default_factory=list)
    note: str = (
        "Each selected tracker segment is treated as one vehicle identity. "
        "Review the contact sheet before using it as high-confidence ReID supervision."
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Auto-curate clean vehicle identity folders from candidate track batches")
    parser.add_argument("--candidates-root", type=str, default="datasets/vehicle_identity_candidates_videoplayback_batch")
    parser.add_argument("--output-dir", type=str, default="datasets/vehicle_identity_videoplayback_auto")
    parser.add_argument("--min-crops", type=int, default=5)
    parser.add_argument("--min-mean-confidence", type=float, default=0.45)
    parser.add_argument("--max-crops-per-identity", type=int, default=40)
    parser.add_argument("--val-ratio", type=float, default=0.25)
    parser.add_argument("--contact-sheet", type=str, default="runs/eval/vehicle_identity_videoplayback_auto_contact_sheet.jpg")
    parser.add_argument("--clear-output", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    summary = auto_curate_vehicle_identity_tracks(
        candidates_root=Path(args.candidates_root),
        output_dir=Path(args.output_dir),
        min_crops=args.min_crops,
        min_mean_confidence=args.min_mean_confidence,
        max_crops_per_identity=args.max_crops_per_identity,
        val_ratio=args.val_ratio,
        contact_sheet=Path(args.contact_sheet),
        clear_output=args.clear_output,
        dry_run=args.dry_run,
    )
    print(json.dumps(asdict(summary), indent=2))
    return 0 if summary.crop_count > 0 or args.dry_run else 1


def auto_curate_vehicle_identity_tracks(
    candidates_root: Path,
    output_dir: Path,
    min_crops: int = 5,
    min_mean_confidence: float = 0.45,
    max_crops_per_identity: int = 40,
    val_ratio: float = 0.25,
    contact_sheet: Path | None = None,
    clear_output: bool = False,
    dry_run: bool = False,
) -> AutoCurationSummary:
    candidates_root = Path(candidates_root)
    output_dir = Path(output_dir)
    if not candidates_root.exists():
        raise FileNotFoundError(f"Candidates root not found: {candidates_root}")
    if clear_output and output_dir.exists() and not dry_run:
        _remove_output_dir(output_dir)
    if not dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)

    all_tracks = load_track_rows(candidates_root)
    selected_rows = [
        row
        for row in all_tracks
        if int(row["crop_count"]) >= min_crops and float(row["mean_confidence"]) >= min_mean_confidence
    ]

    manifest_rows: list[dict[str, str]] = []
    selected_tracks: list[SelectedTrack] = []

    for identity_index, row in enumerate(selected_rows, start=1):
        identity_id = f"vehicle_{identity_index:04d}"
        segment_dir = Path(row["segment_dir"])
        camera_name = str(row["camera_name"])
        track_id = int(row["track_id"])
        track_dir = segment_dir / camera_name / f"track_{track_id:04d}"
        crop_paths = sorted(path for path in track_dir.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES)
        if max_crops_per_identity > 0:
            crop_paths = evenly_sample(crop_paths, max_crops_per_identity)
        train_paths, val_paths = split_train_val(crop_paths, val_ratio)

        copied = 0
        for split, paths in (("train", train_paths), ("val", val_paths)):
            for crop_index, crop_path in enumerate(paths, start=1):
                output_name = f"{row['segment']}_{camera_name}_track_{track_id:04d}_{crop_index:05d}{crop_path.suffix.lower()}"
                output_path = output_dir / split / identity_id / output_name
                if not dry_run:
                    output_path.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(crop_path, output_path)
                copied += 1
                manifest_rows.append(
                    {
                        "identity_id": identity_id,
                        "split": split,
                        "segment": str(row["segment"]),
                        "camera_name": camera_name,
                        "track_id": str(track_id),
                        "mean_confidence": str(row["mean_confidence"]),
                        "label": str(row["label"]),
                        "source_crop": str(crop_path),
                        "output_crop": str(output_path),
                    }
                )

        selected_tracks.append(
            SelectedTrack(
                identity_id=identity_id,
                segment=str(row["segment"]),
                camera_name=camera_name,
                track_id=track_id,
                crop_count=int(row["crop_count"]),
                selected_crop_count=copied,
                train_count=len(train_paths),
                val_count=len(val_paths),
                mean_confidence=float(row["mean_confidence"]),
                label=str(row["label"]),
                representative_crop=str(row["representative_crop"]),
            )
        )

    summary = AutoCurationSummary(
        candidates_root=str(candidates_root),
        output_dir=str(output_dir),
        track_count=len(all_tracks),
        selected_track_count=len(selected_tracks),
        identity_count=len(selected_tracks),
        crop_count=len(manifest_rows),
        train_count=sum(track.train_count for track in selected_tracks),
        val_count=sum(track.val_count for track in selected_tracks),
        min_crops=min_crops,
        min_mean_confidence=min_mean_confidence,
        max_crops_per_identity=max_crops_per_identity,
        selected_tracks=selected_tracks,
    )

    if not dry_run:
        write_manifest(output_dir / "manifest.csv", manifest_rows)
        write_selected_tracks(output_dir / "selected_tracks.csv", selected_tracks)
        (output_dir / "dataset_summary.json").write_text(json.dumps(asdict(summary), indent=2), encoding="utf-8")
        write_readme(output_dir / "README.md", summary)
        if contact_sheet:
            write_contact_sheet(contact_sheet, selected_tracks)
    return summary


def load_track_rows(candidates_root: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for summary_path in sorted(candidates_root.glob("seg_*/track_summary.csv")):
        segment_dir = summary_path.parent
        with open(summary_path, "r", encoding="utf-8", newline="") as file:
            for row in csv.DictReader(file):
                row["segment"] = segment_dir.name
                row["segment_dir"] = str(segment_dir)
                rows.append(row)
    return rows


def evenly_sample(paths: list[Path], max_items: int) -> list[Path]:
    if len(paths) <= max_items:
        return paths
    if max_items <= 1:
        return [paths[0]]
    step = (len(paths) - 1) / (max_items - 1)
    indices = [round(index * step) for index in range(max_items)]
    return [paths[index] for index in indices]


def split_train_val(paths: list[Path], val_ratio: float) -> tuple[list[Path], list[Path]]:
    if len(paths) <= 1:
        return paths, []
    val_count = max(1, round(len(paths) * val_ratio))
    if len(paths) - val_count < 1:
        val_count = len(paths) - 1
    val_indices = set(round(index * (len(paths) - 1) / max(1, val_count - 1)) for index in range(val_count))
    train_paths = [path for index, path in enumerate(paths) if index not in val_indices]
    val_paths = [path for index, path in enumerate(paths) if index in val_indices]
    return train_paths, val_paths


def write_manifest(path: Path, rows: list[dict[str, str]]) -> None:
    fieldnames = [
        "identity_id",
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


def write_selected_tracks(path: Path, tracks: list[SelectedTrack]) -> None:
    fieldnames = list(asdict(SelectedTrack("", "", "", 0, 0, 0, 0, 0, 0.0, "", "")).keys())
    with open(path, "w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for track in tracks:
            writer.writerow(asdict(track))


def write_readme(path: Path, summary: AutoCurationSummary) -> None:
    path.write_text(
        "\n".join(
            [
                "# Auto-Curated Vehicle Identity Dataset",
                "",
                "Each selected tracker segment is treated as one identity.",
                "Review the contact sheet and selected_tracks.csv before treating this as final identity supervision.",
                "",
                f"Identities: {summary.identity_count}",
                f"Train crops: {summary.train_count}",
                f"Val crops: {summary.val_count}",
                f"Min crops per selected track: {summary.min_crops}",
                f"Min mean confidence: {summary.min_mean_confidence}",
                "",
            ]
        ),
        encoding="utf-8",
    )


def write_contact_sheet(path: Path, tracks: list[SelectedTrack], max_tracks: int = 60) -> None:
    selected = tracks[:max_tracks]
    if not selected:
        return
    thumb_w, thumb_h = 160, 104
    pad, label_h = 10, 34
    cols = 5
    rows = (len(selected) + cols - 1) // cols
    canvas = Image.new("RGB", (pad + cols * (thumb_w + pad), pad + rows * (thumb_h + label_h + pad)), "white")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()

    for index, track in enumerate(selected):
        x = pad + (index % cols) * (thumb_w + pad)
        y = pad + (index // cols) * (thumb_h + label_h + pad)
        image = thumbnail(Path(track.representative_crop), thumb_w, thumb_h)
        canvas.paste(image, (x, y))
        label = f"{track.identity_id} {track.crop_count} {track.mean_confidence:.2f}"
        draw.text((x, y + thumb_h + 3), label, fill="black", font=font)
        draw.text((x, y + thumb_h + 15), track.label[:28], fill="black", font=font)

    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path, quality=92)


def thumbnail(path: Path, width: int, height: int) -> Image.Image:
    image = Image.open(path).convert("RGB")
    image.thumbnail((width, height), Image.BILINEAR)
    canvas = Image.new("RGB", (width, height), "white")
    canvas.paste(image, ((width - image.width) // 2, (height - image.height) // 2))
    return canvas


def _remove_output_dir(output_dir: Path) -> None:
    resolved = output_dir.resolve()
    if "datasets" not in resolved.parts:
        raise ValueError(f"Refusing to clear output outside a datasets folder: {resolved}")
    shutil.rmtree(resolved)


if __name__ == "__main__":
    raise SystemExit(main())
