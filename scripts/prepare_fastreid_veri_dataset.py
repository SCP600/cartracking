from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass
class VeRiExportSummary:
    source_dir: str
    output_dir: str
    identity_count: int
    train_count: int
    query_count: int
    gallery_count: int
    note: str = (
        "This exports bootstrap pseudo identities into FastReID's built-in VeRi layout. "
        "Replace with true vehicle identity data before measuring real ReID quality."
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export identity-folder crops to FastReID VeRi layout")
    parser.add_argument("--source-dir", type=str, default="datasets/vehicle_reid_bootstrap", help="Bootstrap ReID crop folder")
    parser.add_argument("--output-dir", type=str, default="datasets/fastreid/veri", help="Output VeRi dataset folder")
    parser.add_argument("--train-split", type=str, default="train", help="Source split for FastReID image_train")
    parser.add_argument("--val-split", type=str, default="val", help="Source split for FastReID image_query/image_test")
    parser.add_argument("--clear-output", action="store_true", help="Remove output folder before writing")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    summary = export_identity_folders_to_veri(
        source_dir=Path(args.source_dir),
        output_dir=Path(args.output_dir),
        train_split=args.train_split,
        val_split=args.val_split,
        clear_output=args.clear_output,
    )
    print(f"[INFO] Identities: {summary.identity_count}")
    print(f"[INFO] Train     : {summary.train_count}")
    print(f"[INFO] Query     : {summary.query_count}")
    print(f"[INFO] Gallery   : {summary.gallery_count}")
    print(f"[INFO] Output    : {summary.output_dir}")
    return 0 if summary.train_count and summary.query_count and summary.gallery_count else 1


def export_identity_folders_to_veri(
    source_dir: Path,
    output_dir: Path,
    train_split: str = "train",
    val_split: str = "val",
    clear_output: bool = False,
) -> VeRiExportSummary:
    source_dir = Path(source_dir)
    output_dir = Path(output_dir)
    train_dir = source_dir / train_split
    val_dir = source_dir / val_split
    if not train_dir.exists():
        raise FileNotFoundError(f"Train split not found: {train_dir}")
    if not val_dir.exists():
        raise FileNotFoundError(f"Validation split not found: {val_dir}")

    if clear_output and output_dir.exists():
        _remove_output_dir(output_dir)

    image_train = output_dir / "image_train"
    image_query = output_dir / "image_query"
    image_test = output_dir / "image_test"
    for directory in (image_train, image_query, image_test):
        directory.mkdir(parents=True, exist_ok=True)

    identities = sorted(path.name for path in train_dir.iterdir() if path.is_dir())
    pid_map = {identity: pid for pid, identity in enumerate(identities, start=1)}
    train_count = 0
    query_count = 0
    gallery_count = 0

    for identity in identities:
        pid = pid_map[identity]
        train_images = _identity_images(train_dir / identity)
        val_images = _identity_images(val_dir / identity)
        for index, image_path in enumerate(train_images, start=1):
            _copy_image(image_path, image_train / _veri_filename(pid, 1, index))
            train_count += 1

        if not val_images:
            continue

        query_image = val_images[0]
        _copy_image(query_image, image_query / _veri_filename(pid, 1, 1))
        query_count += 1

        gallery_images = val_images[1:] or [query_image]
        for index, image_path in enumerate(gallery_images, start=1):
            _copy_image(image_path, image_test / _veri_filename(pid, 2, index))
            gallery_count += 1

    summary = VeRiExportSummary(
        source_dir=str(source_dir),
        output_dir=str(output_dir),
        identity_count=len(pid_map),
        train_count=train_count,
        query_count=query_count,
        gallery_count=gallery_count,
    )
    with open(output_dir / "pid_map.json", "w", encoding="utf-8") as file:
        json.dump(pid_map, file, indent=2)
    with open(output_dir / "export_summary.json", "w", encoding="utf-8") as file:
        json.dump(asdict(summary), file, indent=2)
    return summary


def _identity_images(identity_dir: Path) -> list[Path]:
    if not identity_dir.exists():
        return []
    return sorted(path for path in identity_dir.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES)


def _veri_filename(pid: int, camid: int, index: int) -> str:
    return f"{pid:04d}_c{camid:03d}_{index:06d}.jpg"


def _copy_image(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def _remove_output_dir(output_dir: Path) -> None:
    resolved = output_dir.resolve()
    if "datasets" not in resolved.parts:
        raise ValueError(f"Refusing to clear output outside a datasets folder: {resolved}")
    shutil.rmtree(resolved)


if __name__ == "__main__":
    raise SystemExit(main())
