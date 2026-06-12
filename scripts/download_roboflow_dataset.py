import sys
from roboflow import Roboflow

def main():
    # =========================================================================
    # 請將下方的 "YOUR_API_KEY" 替換成您在 Roboflow 網站上取得的真實 API Key
    # 取得網址: https://universe.roboflow.com/yarok077-gmail-com/stanford_car-yaayi
    # 點擊 "Download Dataset" -> 選擇 YOLOv8 -> "Show Download Code" 即可看到
    # =========================================================================
    import os
    from dotenv import load_dotenv
    
    load_dotenv()
    API_KEY = os.environ.get("ROBOFLOW_API_KEY")
    
    if not API_KEY or API_KEY == "YOUR_API_KEY":
        print("[錯誤] 您尚未設定 ROBOFLOW_API_KEY！")
        print("請在專案根目錄建立一個 .env 檔案，內容為：")
        print("ROBOFLOW_API_KEY=您的真實金鑰")
        sys.exit(1)

    print("[INFO] 開始連接 Roboflow 並下載資料集...")
    
    try:
        rf = Roboflow(api_key=API_KEY)
        project = rf.workspace("yarok077-gmail-com").project("stanford_car-yaayi")
        
        # 預設下載最新版本，這裡以 version(1) 為例，若有錯誤請改為網站顯示的版本號
        version = project.version(1)
        
        # 下載 YOLOv8 格式
        dataset = version.download("yolov8")
        print(f"\n[INFO] 下載成功！資料集已儲存於: {dataset.location}")
    except Exception as e:
        print(f"\n[錯誤] 下載失敗，請檢查 API Key 或版本號是否正確。錯誤訊息: {e}")

if __name__ == "__main__":
    main()
