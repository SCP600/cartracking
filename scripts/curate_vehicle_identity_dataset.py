from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
from dataclasses import asdict, dataclass, field
from pathlib import Path


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
FALSE_VALUES = {"", "0", "false", "no", "n", "skip"}
TRUE_VALUES = {"1", "true", "yes", "y", "keep"}


@dataclass
class CuratedIdentitySummary:
    candidates_dir: str
    output_dir: str
    mapping_csv: str
    identity_count: int = 0
    crop_count: int = 0
    skipped_track_count: int = 0
    missing_track_count: int = 0
    per_identity: dict[str, int] = field(default_factory=dict)
    note: str = "Curated identity folders are intended for FastReID training after manual review."


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Create or apply a vehicle identity mapping from candidate track folders")
    parser.add_argument("--candidates-dir", type=str, default="datasets/vehicle_identity_candidates")
    parser.add_argument("--mapping-csv", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default="datasets/vehicle_identity")
    parser.add_argument("--split", type=str, default="train", help="Default split when applying rows without a split value")
    parser.add_argument("--init-mapping", action="store_true", help="Write an editable mapping template from track_summary.csv")
    parser.add_argument("--clear-output", action="store_true")
    parser.add_argument("--no-copy", dest="copy_images", action="store_false", help="Write reports without copying images")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    candidates_dir = Path(args.candidates_dir)
    mapping_csv = Path(args.mapping_csv) if args.mapping_csv else candidates_dir / "identity_mapping_template.csv"

    if args.init_mapping:
        row_count = init_identity_mapping(candidates_dir=candidates_dir, mapping_csv=mapping_csv)
        print(f"[INFO] Mapping template rows: {row_count}")
        print(f"[INFO] Edit this file: {mapping_csv}")
        return 0 if row_count > 0 else 1

    summary = curate_identity_dataset(
        candidates_dir=candidates_dir,
        mapping_csv=mapping_csv,
        output_dir=Path(args.output_dir),
        default_split=args.split,
        clear_output=args.clear_output,
        copy_images=args.copy_images,
        dry_run=args.dry_run,
    )
    print(json.dumps(asdict(summary), indent=2))
    return 0 if summary.crop_count > 0 or args.dry_run else 1


def init_identity_mapping(candidates_dir: Path, mapping_csv: Path) -> int:
    candidates_dir = Path(candidates_dir)
    track_summary = candidates_dir / "track_summary.csv"
    if not track_summary.exists():
        raise FileNotFoundError(f"Track summary not found: {track_summary}")

    rows = []
    with open(track_summary, "r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        for row in reader:
            if not _parse_bool(row.get("keep", "true"), default=True):
                continue
            camera_name = str(row.get("camera_name", "")).strip()
            track_id = str(row.get("track_id", "")).strip()
            if not camera_name or not track_id:
                continue
            rows.append(
                {
                    "include": "",
                    "identity_id": "",
                    "split": "train",
                    "camera_name": camera_name,
                    "track_id": track_id,
                    "crop_count": row.get("crop_count", ""),
                    "first_frame_index": row.get("first_frame_index", ""),
                    "last_frame_index": row.get("last_frame_index", ""),
                    "mean_confidence": row.get("mean_confidence", ""),
                    "representative_crop": row.get("representative_crop", ""),
                    "note": "Fill include=1 and identity_id=vehicle_0001 after review.",
                }
            )

    mapping_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(mapping_csv, "w", encoding="utf-8", newline="") as file:
        fieldnames = [
            "include",
            "identity_id",
            "split",
            "camera_name",
            "track_id",
            "crop_count",
            "first_frame_index",
            "last_frame_index",
            "mean_confidence",
            "representative_crop",
            "note",
        ]
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def curate_identity_dataset(
    candidates_dir: Path,
    mapping_csv: Path,
    output_dir: Path,
    default_split: str = "train",
    clear_output: bool = False,
    copy_images: bool = True,
    dry_run: bool = False,
) -> CuratedIdentitySummary:
    candidates_dir = Path(candidates_dir)
    mapping_csv = Path(mapping_csv)
    output_dir = Path(output_dir)
    if not mapping_csv.exists():
        raise FileNotFoundError(f"Mapping CSV not found: {mapping_csv}")
    if clear_output and output_dir.exists() and not dry_run:
        _remove_output_dir(output_dir)
    if not dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)

    manifest_rows: list[dict[str, str]] = []
    per_identity: dict[str, int] = {}
    skipped_tracks = 0
    missing_tracks = 0

    with open(mapping_csv, "r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        for row in reader:
            if not _parse_bool(row.get("include", ""), default=False):
                skipped_tracks += 1
                continue
            identity_id = _safe_identity_id(row.get("identity_id", ""))
            if not identity_id:
                skipped_tracks += 1
                continue
            split = _safe_split(row.get("split", "") or default_split)
            camera_name = str(row.get("camera_name", "")).strip()
            track_id = int(str(row.get("track_id", "0")).strip())
            track_dir = candidates_dir / camera_name / f"track_{track_id:04d}"
            if not track_dir.exists():
                missing_tracks += 1
                continue

            crop_paths = sorted(path for path in track_dir.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES)
            for crop_index, crop_path in enumerate(crop_paths, start=1):
                output_name = f"{camera_name}_track_{track_id:04d}_{crop_index:05d}{crop_path.suffix.lower()}"
                output_path = output_dir / split / identity_id / output_name
                if not dry_run and copy_images:
                    output_path.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(crop_path, output_path)
                per_identity[identity_id] = per_identity.get(identity_id, 0) + 1
                manifest_rows.append(
                    {
                        "split": split,
                        "identity_id": identity_id,
                        "camera_name": camera_name,
                        "track_id": str(track_id),
                        "source_crop": str(crop_path),
                        "output_crop": str(output_path),
                    }
                )

    summary = CuratedIdentitySummary(
        candidates_dir=str(candidates_dir),
        output_dir=str(output_dir),
        mapping_csv=str(mapping_csv),
        identity_count=len(per_identity),
        crop_count=len(manifest_rows),
        skipped_track_count=skipped_tracks,
        missing_track_count=missing_tracks,
        per_identity=dict(sorted(per_identity.items())),
    )

    if not dry_run:
        _write_manifest(output_dir / "manifest.csv", manifest_rows)
        (output_dir / "dataset_summary.json").write_text(json.dumps(asdict(summary), indent=2), encoding="utf-8")
        _write_readme(output_dir / "README.md", summary)
    return summary


def _parse_bool(value: str | None, default: bool) -> bool:
    if value is None:
        return default
    text = str(value).strip().lower()
    if text in TRUE_VALUES:
        return True
    if text in FALSE_VALUES:
        return False
    return default


def _safe_identity_id(value: str) -> str:
    text = str(value).strip()
    if not text:
        return ""
    cleaned = re.sub(r"[^A-Za-z0-9]+", "_", text).strip("_")
    return re.sub(r"_+", "_", cleaned)


def _safe_split(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9]+", "_", str(value).strip()).strip("_")
    return cleaned or "train"


def _write_manifest(path: Path, rows: list[dict[str, str]]) -> None:
    fieldnames = ["split", "identity_id", "camera_name", "track_id", "source_crop", "output_crop"]
    with open(path, "w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_readme(path: Path, summary: CuratedIdentitySummary) -> None:
    path.write_text(
        "\n".join(
            [
                "# Vehicle Identity Dataset",
                "",
                "Each identity folder should contain crops of the same real vehicle.",
                "",
                "Next step:",
                "",
                "```powershell",
                "C:\\Users\\vu86e\\anaconda3\\envs\\cartracking\\python.exe scripts\\prepare_fastreid_veri_dataset.py `",
                "  --source-dir datasets\\vehicle_identity `",
                "  --output-dir datasets\\fastreid\\veri_identity `",
                "  --clear-output",
                "```",
                "",
                f"Identities: {summary.identity_count}",
                f"Crops: {summary.crop_count}",
                "",
            ]
        ),
        encoding="utf-8",
    )


def _remove_output_dir(output_dir: Path) -> None:
    resolved = output_dir.resolve()
    if "datasets" not in resolved.parts:
        raise ValueError(f"Refusing to clear output outside a datasets folder: {resolved}")
    shutil.rmtree(resolved)


if __name__ == "__main__":
    raise SystemExit(main())
