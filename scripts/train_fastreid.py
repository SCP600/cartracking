import argparse
import json
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

# 這是針對 FastReID 的微調訓練鷹架 (Training Scaffold)。
# 根據 CVGIP 2024 論文 "Cross-Camera Multi-Target Vehicle Tracking"，
# 以下超參數能顯著提升車輛 ReID 的準確率。


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass
class FastReIDTrainingPlan:
    data_dir: str
    output_dir: str
    learning_rate: float
    batch_size: int
    physical_batch_size: int
    accumulate_grad: int
    epochs: int
    identity_count: int
    image_count: int
    export_format: str
    config_file: str | None = None
    fastreid_root: str | None = None
    fastreid_datasets_root: str | None = None
    device: str = "cuda:0"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train FastReID with paper's optimal hyperparameters")
    parser.add_argument("--data_dir", type=str, default="./datasets/vehicle_reid", help="Path to your ReID dataset")
    parser.add_argument("--output_dir", type=str, default="./weights/fastreid_finetuned", help="Where to save the model")
    
    # 論文最佳超參數
    parser.add_argument("--lr", type=float, default=0.00035, help="Learning rate (Optimal: 0.00035)")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size (Optimal: 256). Very important for accuracy.")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs (Optimal: 10)")
    parser.add_argument("--accumulate_grad", type=int, default=1, help="If VRAM is insufficient for BS=256, set this to e.g. 4 (64x4=256)")
    parser.add_argument("--export-format", choices=("pt", "onnx"), default="onnx", help="Expected export format for Ultralytics BoT-SORT")
    parser.add_argument("--fastreid-root", type=str, default=None, help="Path to a local FastReID checkout")
    parser.add_argument("--config-file", type=str, default=None, help="FastReID config file to use with --run")
    parser.add_argument("--fastreid-datasets-root", type=str, default=None, help="Path containing FastReID datasets such as veri/")
    parser.add_argument("--device", type=str, default="cuda:0", help="FastReID MODEL.DEVICE, e.g. cuda:0 or cpu")
    parser.add_argument("--num-gpus", type=int, default=1, help="Number of GPUs for FastReID training")
    parser.add_argument("--run", action="store_true", help="Run FastReID training instead of only writing the training plan")
    parser.add_argument("--dry-run", action="store_true", help="Validate inputs and write the training plan without running training")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    data_path = Path(args.data_dir)
    output_path = Path(args.output_dir)

    print("=== FastReID Fine-tuning Setup ===")
    print(f"Dataset Path : {data_path}")
    print(f"Learning Rate: {args.lr}")
    print(f"Batch Size   : {args.batch_size}")
    print(f"Accumulate   : {args.accumulate_grad}")
    print(f"Epochs       : {args.epochs}")
    print("==================================")
    
    if not data_path.exists():
        print(f"[!] Warning: Dataset directory '{data_path}' not found.")
        print("    Please place your vehicle ReID dataset here before training.")
        print("    Expected simple layout: datasets/vehicle_reid/<vehicle_identity>/*.jpg")
        return 1

    identity_count, image_count = inspect_reid_dataset(data_path)
    if identity_count == 0 or image_count == 0:
        print("[!] No ReID identities/images were found.")
        print("    Expected simple layout: datasets/vehicle_reid/<vehicle_identity>/*.jpg")
        return 1

    output_path.mkdir(parents=True, exist_ok=True)
    plan = build_training_plan(args, data_path, output_path, identity_count, image_count)
    plan_path = output_path / "training_plan.json"
    with open(plan_path, "w", encoding="utf-8") as file:
        json.dump(asdict(plan), file, indent=2)
    
    print(f"\n[INFO] Found {identity_count} identities and {image_count} images.")
    print(f"[INFO] Training plan written to {plan_path}")
    print(f"[INFO] Target export format: {args.export_format}")

    if args.dry_run or not args.run:
        print("[INFO] Dry run completed. Use --run with --fastreid-root and --config-file to start FastReID training.")
        return 0

    return run_fastreid_training(args, output_path)


def inspect_reid_dataset(data_path: Path) -> tuple[int, int]:
    identity_dirs = [path for path in data_path.iterdir() if path.is_dir()]
    image_count = 0
    valid_identity_count = 0
    for identity_dir in identity_dirs:
        images = [path for path in identity_dir.rglob("*") if path.suffix.lower() in IMAGE_SUFFIXES]
        if images:
            valid_identity_count += 1
            image_count += len(images)
    return valid_identity_count, image_count


def build_training_plan(args: argparse.Namespace, data_path: Path, output_path: Path, identity_count: int, image_count: int) -> FastReIDTrainingPlan:
    accumulate_grad = max(1, args.accumulate_grad)
    physical_batch_size = max(1, args.batch_size // accumulate_grad)
    return FastReIDTrainingPlan(
        data_dir=str(data_path),
        output_dir=str(output_path),
        learning_rate=args.lr,
        batch_size=args.batch_size,
        physical_batch_size=physical_batch_size,
        accumulate_grad=accumulate_grad,
        epochs=args.epochs,
        identity_count=identity_count,
        image_count=image_count,
        export_format=args.export_format,
        config_file=args.config_file,
        fastreid_root=args.fastreid_root,
        fastreid_datasets_root=args.fastreid_datasets_root,
        device=args.device,
    )


def run_fastreid_training(args: argparse.Namespace, output_path: Path) -> int:
    if not args.fastreid_root:
        print("[錯誤] --run 需要指定 --fastreid-root。")
        return 1
    if not args.config_file:
        print("[錯誤] --run 需要指定 --config-file，讓 FastReID 知道資料集與模型架構。")
        return 1

    fastreid_root = Path(args.fastreid_root).resolve()
    train_script = fastreid_root / "tools" / "train_net.py"
    config_file = Path(args.config_file).resolve()
    output_path = output_path.resolve()
    if not train_script.exists():
        print(f"[錯誤] 找不到 FastReID 訓練入口：{train_script}")
        return 1
    if not config_file.exists():
        print(f"[錯誤] 找不到 FastReID config：{config_file}")
        return 1

    command = [
        sys.executable,
        str(train_script),
        "--config-file",
        str(config_file),
        "--num-gpus",
        str(args.num_gpus),
        "OUTPUT_DIR",
        str(output_path),
        "SOLVER.BASE_LR",
        str(args.lr),
        "SOLVER.IMS_PER_BATCH",
        str(max(1, args.batch_size // max(1, args.accumulate_grad))),
        "SOLVER.MAX_EPOCH",
        str(args.epochs),
        "MODEL.DEVICE",
        str(args.device),
    ]
    print("\n[+] Starting FastReID training...")
    print(" ".join(command))
    env = None
    if args.fastreid_datasets_root:
        import os

        env = os.environ.copy()
        env["FASTREID_DATASETS"] = str(Path(args.fastreid_datasets_root).resolve())
    completed = subprocess.run(command, cwd=fastreid_root, env=env, check=False)
    if completed.returncode != 0:
        print(f"[錯誤] FastReID training failed with exit code {completed.returncode}.")
        return completed.returncode

    print("\n[INFO] FastReID training completed.")
    print(f"[INFO] Export the final weights to {args.export_format} and set reid_model_path in default_config.json.")
    return 0

if __name__ == "__main__":
    sys.exit(main())
