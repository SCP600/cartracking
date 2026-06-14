import argparse
import sys

from roboflow import Roboflow


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Download the vehicle dataset from Roboflow")
    parser.add_argument("--workspace", default="yarok077-gmail-com", help="Roboflow workspace slug")
    parser.add_argument("--project", default="stanford_car-yaayi", help="Roboflow project slug")
    parser.add_argument("--version", type=int, default=1, help="Roboflow dataset version")
    parser.add_argument("--format", default="yolov8", help="Download format, for example yolov8")
    parser.add_argument("--api-key-env", default="ROBOFLOW_API_KEY", help="Environment variable that stores the API key")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    # =========================================================================
    # 請在專案根目錄建立 .env，並填入您在 Roboflow 網站上取得的真實 API Key
    # 取得網址: https://universe.roboflow.com/yarok077-gmail-com/stanford_car-yaayi
    # 點擊 "Download Dataset" -> 選擇 YOLOv8 -> "Show Download Code" 即可看到
    # =========================================================================
    import os
    from dotenv import load_dotenv
    
    load_dotenv()
    api_key = os.environ.get(args.api_key_env)
    
    if not api_key or api_key == "YOUR_API_KEY":
        print(f"[錯誤] 您尚未設定 {args.api_key_env}！")
        print("請在專案根目錄建立一個 .env 檔案，內容為：")
        print(f"{args.api_key_env}=您的真實金鑰")
        return 1

    print("[INFO] 開始連接 Roboflow 並下載資料集...")
    
    try:
        rf = Roboflow(api_key=api_key)
        project = rf.workspace(args.workspace).project(args.project)
        version = project.version(args.version)
        
        dataset = version.download(args.format)
        print(f"\n[INFO] 下載成功！資料集已儲存於: {dataset.location}")
        return 0
    except Exception as e:
        print(f"\n[錯誤] 下載失敗，請檢查 API Key 或版本號是否正確。錯誤訊息: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
