from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass
class YOLOCropDatasetSummary:
    source_dir: str
    output_dir: str
    train_count: int = 0
    val_count: int = 0
    class_name: str = "vehicle"
    note: str = (
        "This dataset labels each curated crop as one full-image vehicle. "
        "Use it for conservative domain adaptation, not as a replacement for full-frame bbox labels."
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Create a one-class YOLO dataset from curated vehicle crops")
    parser.add_argument("--source-dir", type=str, default="datasets/vehicle_identity_videoplayback_auto")
    parser.add_argument("--output-dir", type=str, default="datasets/yolo_videoplayback_vehicle_crops")
    parser.add_argument("--class-name", type=str, default="vehicle")
    parser.add_argument("--train-split", type=str, default="train")
    parser.add_argument("--val-split", type=str, default="val")
    parser.add_argument("--clear-output", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    summary = prepare_yolo_dataset_from_identity_crops(
        source_dir=Path(args.source_dir),
        output_dir=Path(args.output_dir),
        class_name=args.class_name,
        train_split=args.train_split,
        val_split=args.val_split,
        clear_output=args.clear_output,
    )
    print(json.dumps(asdict(summary), indent=2))
    return 0 if summary.train_count and summary.val_count else 1


def prepare_yolo_dataset_from_identity_crops(
    source_dir: Path,
    output_dir: Path,
    class_name: str = "vehicle",
    train_split: str = "train",
    val_split: str = "val",
    clear_output: bool = False,
) -> YOLOCropDatasetSummary:
    source_dir = Path(source_dir)
    output_dir = Path(output_dir)
    if clear_output and output_dir.exists():
        _remove_output_dir(output_dir)

    train_count = export_split(source_dir / train_split, output_dir / "train")
    val_count = export_split(source_dir / val_split, output_dir / "valid")
    data_yaml = output_dir / "data.yaml"
    data_yaml.write_text(
        "\n".join(
            [
                f"path: {output_dir.resolve().as_posix()}",
                "train: train/images",
                "val: valid/images",
                "nc: 1",
                "names:",
                f"  0: {class_name}",
                "",
            ]
        ),
        encoding="utf-8",
    )
    summary = YOLOCropDatasetSummary(
        source_dir=str(source_dir),
        output_dir=str(output_dir),
        train_count=train_count,
        val_count=val_count,
        class_name=class_name,
    )
    (output_dir / "dataset_summary.json").write_text(json.dumps(asdict(summary), indent=2), encoding="utf-8")
    return summary


def export_split(source_split_dir: Path, output_split_dir: Path) -> int:
    image_dir = output_split_dir / "images"
    label_dir = output_split_dir / "labels"
    image_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)

    count = 0
    for identity_dir in sorted(path for path in source_split_dir.iterdir() if path.is_dir()):
        for image_path in sorted(path for path in identity_dir.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES):
            output_name = f"{identity_dir.name}_{image_path.name}"
            output_image = image_dir / output_name
            output_label = label_dir / f"{Path(output_name).stem}.txt"
            shutil.copy2(image_path, output_image)
            output_label.write_text("0 0.5 0.5 1.0 1.0\n", encoding="utf-8")
            count += 1
    return count


def _remove_output_dir(output_dir: Path) -> None:
    resolved = output_dir.resolve()
    if "datasets" not in resolved.parts:
        raise ValueError(f"Refusing to clear output outside a datasets folder: {resolved}")
    shutil.rmtree(resolved)


if __name__ == "__main__":
    raise SystemExit(main())
