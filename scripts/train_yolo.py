import argparse
import sys
from pathlib import Path

# 這是針對 YOLOv7 (或同系列 YOLO) 的微調訓練鷹架 (Training Scaffold)。
# 根據 CVGIP 2024 論文 "Cross-Camera Multi-Target Vehicle Tracking"，
# 以下超參數為他們測試出的 YOLO 模型設定。


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train YOLO detector with paper's hyperparameters")
    parser.add_argument("--model", type=str, default="yolo26n.pt", help="Pretrained YOLO model path")
    parser.add_argument("--data", type=str, default="data.yaml", help="Path to your dataset YAML file")
    
    # 論文最佳超參數
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate (Optimal: 0.01)")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size (Optimal: 16)")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs (Optimal: 20)")
    parser.add_argument("--imgsz", type=int, default=960, help="Training image size")
    parser.add_argument("--optimizer", type=str, default="SGD", help="Optimizer to use. Use SGD so lr0 is respected.")
    parser.add_argument("--momentum", type=float, default=0.937, help="SGD momentum")
    parser.add_argument("--project", type=str, default="runs/train", help="Training output project directory")
    parser.add_argument("--name", type=str, default="yolo_vehicle_detector", help="Training run name")
    parser.add_argument("--device", type=str, default=None, help="Training device, for example 0, cpu, or cuda")
    parser.add_argument("--dry-run", action="store_true", help="Print the training plan without starting training")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    project_path = Path(args.project)
    if not project_path.is_absolute():
        project_path = Path.cwd() / project_path

    print("=== YOLO Fine-tuning Setup ===")
    print(f"Base Model   : {args.model}")
    print(f"Dataset YAML : {args.data}")
    print(f"Learning Rate: {args.lr}")
    print(f"Batch Size   : {args.batch_size}")
    print(f"Epochs       : {args.epochs}")
    print(f"Image Size   : {args.imgsz}")
    print(f"Optimizer    : {args.optimizer}")
    print(f"Output       : {project_path / args.name}")
    print("==============================")

    data_path = Path(args.data)
    if not data_path.exists():
        print(f"[!] Warning: Dataset YAML '{data_path}' not found.")
        print("    Please create a data.yaml configuring your YOLO dataset before training.")
        return 1

    if args.dry_run:
        print("\n[INFO] Dry run completed. No training was started.")
        return 0

    print("\n[+] Initializing YOLO model...")
    from ultralytics import YOLO

    model = YOLO(args.model)

    print("\n[+] Starting training...")
    # 執行 Ultralytics 內建的訓練流程
    train_kwargs = {
        "data": args.data,
        "epochs": args.epochs,
        "batch": args.batch_size,
        "lr0": args.lr,
        "optimizer": args.optimizer,
        "momentum": args.momentum,
        "imgsz": args.imgsz,
        "project": str(project_path),
        "name": args.name,
    }
    if args.device is not None:
        train_kwargs["device"] = args.device

    model.train(
        **train_kwargs
    )
    
    print("\n[INFO] YOLO training completed!")
    print(f"[INFO] The fine-tuned weights are saved in {project_path / args.name / 'weights' / 'best.pt'}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
