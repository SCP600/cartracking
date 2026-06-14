from __future__ import annotations

import argparse
import csv
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
FILENAME_PATTERN = re.compile(r"^(?P<pid>\d+)_c(?P<camid>\d+)_")


@dataclass
class ImageRecord:
    path: str
    pid: int
    camid: int


@dataclass
class RetrievalMetrics:
    query_count: int
    gallery_count: int
    identity_count: int
    rank1: float
    rank5: float
    rank10: float
    map: float
    mean_positive_similarity: float
    mean_negative_similarity: float
    similarity_gap: float
    mean_best_positive_similarity: float
    mean_hardest_negative_similarity: float
    mean_top1_margin: float


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate TorchScript ReID feature quality on a query/gallery dataset.")
    parser.add_argument("--model", type=str, default="weights/fastreid_bootstrap/fastreid_vehicle_reid.torchscript")
    parser.add_argument("--dataset-dir", type=str, default="datasets/fastreid/veri")
    parser.add_argument("--query-dir", type=str, default=None, help="Defaults to <dataset-dir>/image_query")
    parser.add_argument("--gallery-dir", type=str, default=None, help="Defaults to <dataset-dir>/image_test")
    parser.add_argument("--output-dir", type=str, default="runs/eval/reid_feature_eval")
    parser.add_argument("--imgsz", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", type=str, default="auto", help="auto, cpu, cuda, or cuda:0")
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--examples", type=int, default=12, help="Number of query examples in the contact sheet")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    model_path = Path(args.model).resolve()
    dataset_dir = Path(args.dataset_dir)
    query_dir = Path(args.query_dir) if args.query_dir else dataset_dir / "image_query"
    gallery_dir = Path(args.gallery_dir) if args.gallery_dir else dataset_dir / "image_test"
    output_dir = Path(args.output_dir)

    if not model_path.exists():
        raise FileNotFoundError(f"ReID model not found: {model_path}")
    if not query_dir.exists():
        raise FileNotFoundError(f"Query directory not found: {query_dir}")
    if not gallery_dir.exists():
        raise FileNotFoundError(f"Gallery directory not found: {gallery_dir}")

    device = choose_device(args.device)
    output_dir.mkdir(parents=True, exist_ok=True)

    query_records = load_records(query_dir)
    gallery_records = load_records(gallery_dir)
    if not query_records:
        raise RuntimeError(f"No query images found in {query_dir}")
    if not gallery_records:
        raise RuntimeError(f"No gallery images found in {gallery_dir}")

    model = torch.jit.load(str(model_path), map_location=device).eval().to(device)
    query_features = encode_records(model, query_records, args.imgsz, args.batch_size, device)
    gallery_features = encode_records(model, gallery_records, args.imgsz, args.batch_size, device)
    query_features = F.normalize(query_features, p=2, dim=1).cpu()
    gallery_features = F.normalize(gallery_features, p=2, dim=1).cpu()

    metrics, topk_rows = evaluate_retrieval(query_records, gallery_records, query_features, gallery_features, args.topk)
    write_outputs(output_dir, model_path, query_dir, gallery_dir, metrics, topk_rows)
    write_contact_sheet(output_dir / "topk_examples.jpg", query_records, gallery_records, topk_rows, args.examples, args.topk)

    print(json.dumps(asdict(metrics), indent=2))
    print(f"[INFO] Wrote report: {output_dir.resolve()}")
    return 0


def choose_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if requested == "cuda":
        return torch.device("cuda:0")
    return torch.device(requested)


def load_records(directory: Path) -> list[ImageRecord]:
    records: list[ImageRecord] = []
    for path in sorted(directory.iterdir()):
        if path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        match = FILENAME_PATTERN.match(path.name)
        if not match:
            continue
        records.append(ImageRecord(path=str(path), pid=int(match.group("pid")), camid=int(match.group("camid"))))
    return records


def encode_records(
    model: torch.nn.Module,
    records: list[ImageRecord],
    imgsz: int,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    outputs: list[torch.Tensor] = []
    for start in range(0, len(records), batch_size):
        batch_records = records[start : start + batch_size]
        images = torch.stack([load_image_tensor(Path(record.path), imgsz) for record in batch_records]).to(device)
        with torch.no_grad():
            outputs.append(model(images).detach().cpu())
    return torch.cat(outputs, dim=0)


def load_image_tensor(path: Path, imgsz: int) -> torch.Tensor:
    image = Image.open(path).convert("RGB").resize((imgsz, imgsz), Image.BILINEAR)
    data = torch.frombuffer(bytearray(image.tobytes()), dtype=torch.uint8)
    data = data.view(imgsz, imgsz, 3).permute(2, 0, 1).contiguous()
    return data.float().div(255.0)


def evaluate_retrieval(
    query_records: list[ImageRecord],
    gallery_records: list[ImageRecord],
    query_features: torch.Tensor,
    gallery_features: torch.Tensor,
    topk: int,
) -> tuple[RetrievalMetrics, list[dict[str, object]]]:
    similarities = query_features @ gallery_features.T
    gallery_pids = torch.tensor([record.pid for record in gallery_records], dtype=torch.long)
    query_pids = torch.tensor([record.pid for record in query_records], dtype=torch.long)

    rank_hits = {1: 0, 5: 0, 10: 0}
    average_precisions: list[float] = []
    positive_scores: list[float] = []
    negative_scores: list[float] = []
    best_positive_scores: list[float] = []
    hardest_negative_scores: list[float] = []
    top1_margins: list[float] = []
    topk_rows: list[dict[str, object]] = []

    for query_index, query_pid in enumerate(query_pids.tolist()):
        sims = similarities[query_index]
        sorted_indices = torch.argsort(sims, descending=True)
        matches = gallery_pids[sorted_indices].eq(query_pid)

        for k in rank_hits:
            if bool(matches[: min(k, len(matches))].any()):
                rank_hits[k] += 1

        positive_mask = gallery_pids.eq(query_pid)
        negative_mask = ~positive_mask
        if bool(positive_mask.any()):
            positive_values = sims[positive_mask]
            positive_scores.extend(float(value) for value in positive_values)
            best_positive = float(positive_values.max())
            best_positive_scores.append(best_positive)
        else:
            best_positive = float("nan")

        if bool(negative_mask.any()):
            negative_values = sims[negative_mask]
            negative_scores.extend(float(value) for value in negative_values)
            hardest_negative = float(negative_values.max())
            hardest_negative_scores.append(hardest_negative)
        else:
            hardest_negative = float("nan")

        if not torch.isnan(torch.tensor(best_positive)) and not torch.isnan(torch.tensor(hardest_negative)):
            top1_margins.append(best_positive - hardest_negative)

        average_precisions.append(average_precision(matches))
        top_indices = sorted_indices[:topk].tolist()
        topk_rows.append(
            {
                "query_path": query_records[query_index].path,
                "query_pid": query_pid,
                "best_positive_similarity": best_positive,
                "hardest_negative_similarity": hardest_negative,
                "top1_margin": best_positive - hardest_negative,
                "topk": [
                    {
                        "rank": rank + 1,
                        "gallery_path": gallery_records[gallery_index].path,
                        "gallery_pid": gallery_records[gallery_index].pid,
                        "similarity": float(sims[gallery_index]),
                        "match": gallery_records[gallery_index].pid == query_pid,
                    }
                    for rank, gallery_index in enumerate(top_indices)
                ],
            }
        )

    query_count = len(query_records)
    identity_count = len(set(record.pid for record in query_records) | set(record.pid for record in gallery_records))
    mean_positive = mean(positive_scores)
    mean_negative = mean(negative_scores)
    metrics = RetrievalMetrics(
        query_count=query_count,
        gallery_count=len(gallery_records),
        identity_count=identity_count,
        rank1=100.0 * rank_hits[1] / query_count,
        rank5=100.0 * rank_hits[5] / query_count,
        rank10=100.0 * rank_hits[10] / query_count,
        map=100.0 * mean(average_precisions),
        mean_positive_similarity=mean_positive,
        mean_negative_similarity=mean_negative,
        similarity_gap=mean_positive - mean_negative,
        mean_best_positive_similarity=mean(best_positive_scores),
        mean_hardest_negative_similarity=mean(hardest_negative_scores),
        mean_top1_margin=mean(top1_margins),
    )
    return metrics, topk_rows


def average_precision(sorted_matches: torch.Tensor) -> float:
    positives = sorted_matches.float()
    positive_count = float(positives.sum())
    if positive_count == 0:
        return 0.0
    precision_at_k = torch.cumsum(positives, dim=0) / torch.arange(1, len(positives) + 1, dtype=torch.float32)
    return float((precision_at_k * positives).sum() / positive_count)


def mean(values: list[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def write_outputs(
    output_dir: Path,
    model_path: Path,
    query_dir: Path,
    gallery_dir: Path,
    metrics: RetrievalMetrics,
    topk_rows: list[dict[str, object]],
) -> None:
    report = {
        "model": str(model_path),
        "query_dir": str(query_dir),
        "gallery_dir": str(gallery_dir),
        "metrics": asdict(metrics),
        "note": (
            "These metrics evaluate the current bootstrap pseudo-identity dataset. "
            "They are useful for comparing feature extractors, but not a real cross-camera vehicle ReID score."
        ),
    }
    (output_dir / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    with open(output_dir / "topk.csv", "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["query_pid", "query_path", "rank", "gallery_pid", "match", "similarity", "gallery_path"])
        for row in topk_rows:
            for item in row["topk"]:
                writer.writerow(
                    [
                        row["query_pid"],
                        row["query_path"],
                        item["rank"],
                        item["gallery_pid"],
                        item["match"],
                        item["similarity"],
                        item["gallery_path"],
                    ]
                )


def write_contact_sheet(
    output_path: Path,
    query_records: list[ImageRecord],
    gallery_records: list[ImageRecord],
    topk_rows: list[dict[str, object]],
    example_count: int,
    topk: int,
) -> None:
    if not topk_rows:
        return
    failures = [row for row in topk_rows if not row["topk"][0]["match"]]
    successes = [row for row in topk_rows if row["topk"][0]["match"]]
    selected_rows = (failures + successes)[:example_count]

    thumb = 112
    label_h = 28
    pad = 10
    cols = 1 + topk
    rows = len(selected_rows)
    width = pad + cols * (thumb + pad)
    height = pad + rows * (thumb + label_h + pad)
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()

    for row_index, row in enumerate(selected_rows):
        y = pad + row_index * (thumb + label_h + pad)
        query_image = thumbnail(Path(row["query_path"]), thumb)
        paste_with_border(canvas, query_image, pad, y, "blue")
        draw.text((pad, y + thumb + 3), f"Q pid {row['query_pid']}", fill="black", font=font)
        for item_index, item in enumerate(row["topk"][:topk]):
            x = pad + (item_index + 1) * (thumb + pad)
            gallery_image = thumbnail(Path(item["gallery_path"]), thumb)
            border = "green" if item["match"] else "red"
            paste_with_border(canvas, gallery_image, x, y, border)
            label = f"{item['rank']} pid {item['gallery_pid']} {item['similarity']:.2f}"
            draw.text((x, y + thumb + 3), label, fill="black", font=font)
    canvas.save(output_path, quality=92)


def thumbnail(path: Path, size: int) -> Image.Image:
    image = Image.open(path).convert("RGB")
    image.thumbnail((size, size), Image.BILINEAR)
    canvas = Image.new("RGB", (size, size), "white")
    x = (size - image.width) // 2
    y = (size - image.height) // 2
    canvas.paste(image, (x, y))
    return canvas


def paste_with_border(canvas: Image.Image, image: Image.Image, x: int, y: int, color: str) -> None:
    draw = ImageDraw.Draw(canvas)
    draw.rectangle((x - 2, y - 2, x + image.width + 1, y + image.height + 1), outline=color, width=3)
    canvas.paste(image, (x, y))


if __name__ == "__main__":
    raise SystemExit(main())
