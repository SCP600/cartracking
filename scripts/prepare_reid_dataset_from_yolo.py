from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
from dataclasses import asdict, dataclass, field
from pathlib import Path

import cv2
import yaml


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass
class ReIDCropSummary:
    data_yaml: str
    output_dir: str
    splits: list[str]
    identity_count: int = 0
    crop_count: int = 0
    skipped_count: int = 0
    note: str = (
        "This bootstrap dataset uses YOLO class labels as pseudo identities. "
        "Use true same-vehicle identity labels for production ReID training."
    )
    per_identity: dict[str, int] = field(default_factory=dict)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare a bootstrap ReID crop dataset from a YOLO dataset")
    parser.add_argument("--data-yaml", type=str, default="Stanford_Car-1/data.yaml", help="YOLO data.yaml path")
    parser.add_argument("--output-dir", type=str, default="datasets/vehicle_reid_bootstrap", help="Output folder")
    parser.add_argument("--splits", type=str, default="train,val", help="Comma-separated YOLO splits to convert")
    parser.add_argument("--padding", type=float, default=0.08, help="BBox padding ratio before cropping")
    parser.add_argument("--min-box-size", type=int, default=24, help="Skip boxes smaller than this width/height")
    parser.add_argument("--max-crops-per-identity", type=int, default=0, help="0 means keep every crop")
    parser.add_argument("--clear-output", action="store_true", help="Remove the output folder before writing crops")
    parser.add_argument("--dry-run", action="store_true", help="Inspect the dataset without writing crop images")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    data_yaml = Path(args.data_yaml)
    output_dir = Path(args.output_dir)
    splits = [item.strip() for item in args.splits.split(",") if item.strip()]

    summary = prepare_reid_dataset_from_yolo(
        data_yaml=data_yaml,
        output_dir=output_dir,
        splits=splits,
        padding=args.padding,
        min_box_size=args.min_box_size,
        max_crops_per_identity=args.max_crops_per_identity,
        clear_output=args.clear_output,
        dry_run=args.dry_run,
    )

    print(f"[INFO] Identities: {summary.identity_count}")
    print(f"[INFO] Crops     : {summary.crop_count}")
    print(f"[INFO] Skipped   : {summary.skipped_count}")
    print(f"[INFO] Output    : {summary.output_dir}")
    print("[INFO] Note      : class labels are pseudo identities, not true vehicle identities.")
    return 0 if summary.crop_count > 0 else 1


def prepare_reid_dataset_from_yolo(
    data_yaml: Path,
    output_dir: Path,
    splits: list[str],
    padding: float = 0.08,
    min_box_size: int = 24,
    max_crops_per_identity: int = 0,
    clear_output: bool = False,
    dry_run: bool = False,
) -> ReIDCropSummary:
    data_yaml = Path(data_yaml)
    output_dir = Path(output_dir)
    if not data_yaml.exists():
        raise FileNotFoundError(f"YOLO data.yaml not found: {data_yaml}")

    with open(data_yaml, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    names = _parse_names(config.get("names", []))
    if not names:
        raise ValueError("data.yaml does not contain class names.")

    if clear_output and output_dir.exists() and not dry_run:
        _remove_output_dir(output_dir)

    if not dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)

    manifest_rows: list[dict[str, str]] = []
    per_identity: dict[str, int] = {}
    skipped_count = 0

    for split in splits:
        image_dir = _resolve_split_image_dir(data_yaml, config, split)
        label_dir = image_dir.parent / "labels"
        if not image_dir.exists():
            raise FileNotFoundError(f"Image directory not found for split '{split}': {image_dir}")
        if not label_dir.exists():
            raise FileNotFoundError(f"Label directory not found for split '{split}': {label_dir}")

        image_paths = sorted(path for path in image_dir.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES)
        for image_path in image_paths:
            label_path = label_dir / f"{image_path.stem}.txt"
            if not label_path.exists():
                continue
            image = cv2.imread(str(image_path))
            if image is None:
                skipped_count += 1
                continue
            image_height, image_width = image.shape[:2]
            for box_index, label in enumerate(_read_yolo_labels(label_path)):
                class_id, x_center, y_center, width, height = label
                class_name = names.get(class_id, f"class_{class_id}")
                identity_name = _identity_folder_name(class_id, class_name)
                crop_box = _normalized_xywh_to_padded_xyxy(
                    x_center=x_center,
                    y_center=y_center,
                    width=width,
                    height=height,
                    image_width=image_width,
                    image_height=image_height,
                    padding=padding,
                )
                x1, y1, x2, y2 = crop_box
                crop_width = x2 - x1
                crop_height = y2 - y1
                if crop_width < min_box_size or crop_height < min_box_size:
                    skipped_count += 1
                    continue
                if max_crops_per_identity > 0 and per_identity.get(identity_name, 0) >= max_crops_per_identity:
                    skipped_count += 1
                    continue

                per_identity[identity_name] = per_identity.get(identity_name, 0) + 1
                output_path = output_dir / split / identity_name / f"{image_path.stem}_{box_index:02d}.jpg"
                if not dry_run:
                    output_path.parent.mkdir(parents=True, exist_ok=True)
                    crop = image[y1:y2, x1:x2]
                    cv2.imwrite(str(output_path), crop)

                manifest_rows.append(
                    {
                        "split": split,
                        "identity": identity_name,
                        "class_id": str(class_id),
                        "class_name": class_name,
                        "source_image": str(image_path),
                        "crop_path": str(output_path),
                        "x1": str(x1),
                        "y1": str(y1),
                        "x2": str(x2),
                        "y2": str(y2),
                    }
                )

    summary = ReIDCropSummary(
        data_yaml=str(data_yaml),
        output_dir=str(output_dir),
        splits=splits,
        identity_count=len(per_identity),
        crop_count=len(manifest_rows),
        skipped_count=skipped_count,
        per_identity=dict(sorted(per_identity.items())),
    )

    if not dry_run:
        _write_manifest(output_dir / "manifest.csv", manifest_rows)
        with open(output_dir / "dataset_summary.json", "w", encoding="utf-8") as file:
            json.dump(asdict(summary), file, indent=2)

    return summary


def _parse_names(raw_names) -> dict[int, str]:
    if isinstance(raw_names, dict):
        return {int(key): str(value) for key, value in raw_names.items()}
    return {index: str(value) for index, value in enumerate(raw_names)}


def _resolve_split_image_dir(data_yaml: Path, config: dict, split: str) -> Path:
    config_key = _split_config_key(config, split)
    raw_value = config.get(config_key)
    if not raw_value:
        raise ValueError(f"Split '{split}' is not defined in {data_yaml}")

    candidates = []
    raw_path = Path(str(raw_value))
    if raw_path.is_absolute():
        candidates.append(raw_path)
    else:
        candidates.append((data_yaml.parent / raw_path).resolve())
        if str(raw_value).startswith("../"):
            candidates.append((data_yaml.parent / str(raw_value)[3:]).resolve())
        candidates.append((data_yaml.parent / split / "images").resolve())
        if config_key != split:
            candidates.append((data_yaml.parent / config_key / "images").resolve())

    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def _split_config_key(config: dict, split: str) -> str:
    if split in config:
        return split
    aliases = {
        "valid": "val",
        "validation": "val",
        "val": "valid",
    }
    alias = aliases.get(split)
    if alias in config:
        return alias
    return split


def _read_yolo_labels(label_path: Path) -> list[tuple[int, float, float, float, float]]:
    labels = []
    with open(label_path, "r", encoding="utf-8") as file:
        for line in file:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            labels.append((int(float(parts[0])), float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])))
    return labels


def _normalized_xywh_to_padded_xyxy(
    x_center: float,
    y_center: float,
    width: float,
    height: float,
    image_width: int,
    image_height: int,
    padding: float,
) -> tuple[int, int, int, int]:
    box_width = width * image_width
    box_height = height * image_height
    pad_x = box_width * padding
    pad_y = box_height * padding
    x1 = int(round((x_center * image_width) - (box_width / 2) - pad_x))
    y1 = int(round((y_center * image_height) - (box_height / 2) - pad_y))
    x2 = int(round((x_center * image_width) + (box_width / 2) + pad_x))
    y2 = int(round((y_center * image_height) + (box_height / 2) + pad_y))
    return (
        max(0, x1),
        max(0, y1),
        min(image_width, x2),
        min(image_height, y2),
    )


def _identity_folder_name(class_id: int, class_name: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9]+", "_", class_name).strip("_")
    cleaned = re.sub(r"_+", "_", cleaned)
    return f"{class_id:03d}_{cleaned or 'vehicle'}"


def _write_manifest(path: Path, rows: list[dict[str, str]]) -> None:
    fieldnames = ["split", "identity", "class_id", "class_name", "source_image", "crop_path", "x1", "y1", "x2", "y2"]
    with open(path, "w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _remove_output_dir(output_dir: Path) -> None:
    resolved = output_dir.resolve()
    if "datasets" not in resolved.parts:
        raise ValueError(f"Refusing to clear output outside a datasets folder: {resolved}")
    shutil.rmtree(resolved)


if __name__ == "__main__":
    raise SystemExit(main())
