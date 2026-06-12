import argparse
from pathlib import Path
from ultralytics import YOLO

# 這是針對 YOLOv7 (或同系列 YOLO) 的微調訓練鷹架 (Training Scaffold)。
# 根據 CVGIP 2024 論文 "Cross-Camera Multi-Target Vehicle Tracking"，
# 以下超參數為他們測試出的 YOLO 模型設定。

def main():
    parser = argparse.ArgumentParser(description="Train YOLO detector with paper's hyperparameters")
    parser.add_argument("--model", type=str, default="yolo26n.pt", help="Pretrained YOLO model path")
    parser.add_argument("--data", type=str, default="data.yaml", help="Path to your dataset YAML file")
    
    # 論文最佳超參數
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate (Optimal: 0.01)")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size (Optimal: 16)")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs (Optimal: 20)")

    args = parser.parse_args()

    print("=== YOLO Fine-tuning Setup ===")
    print(f"Base Model   : {args.model}")
    print(f"Dataset YAML : {args.data}")
    print(f"Learning Rate: {args.lr}")
    print(f"Batch Size   : {args.batch_size}")
    print(f"Epochs       : {args.epochs}")
    print("==============================")

    data_path = Path(args.data)
    if not data_path.exists():
        print(f"[!] Warning: Dataset YAML '{data_path}' not found.")
        print("    Please create a data.yaml configuring your YOLO dataset before training.")
        return

    print("\n[+] Initializing YOLO model...")
    model = YOLO(args.model)

    print("\n[+] Starting training...")
    # 執行 Ultralytics 內建的訓練流程
    results = model.train(
        data=args.data,
        epochs=args.epochs,
        batch=args.batch_size,
        lr0=args.lr,
        project="runs/train",
        name="yolo_vehicle_detector"
    )
    
    print("\n[INFO] YOLO training completed!")
    print(f"[INFO] The fine-tuned weights are saved in runs/train/yolo_vehicle_detector/weights/best.pt")

if __name__ == "__main__":
    main()
