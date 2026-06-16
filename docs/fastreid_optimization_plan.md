# 跨攝影機多目標車輛追蹤 (BoT-SORT + FastReID) 優化實作計畫

本文件基於 CVGIP 2024 論文《Cross-Camera Multi-Target Vehicle Tracking》的研究成果，規劃如何將其最佳化配置與超參數整合至 AutoCamTracker 專案中。

## 背景與目標

論文指出在處理跨攝影機的車輛追蹤任務時，結合 **YOLOv7**、**FastReID** 以及 **BoT-SORT** 演算法，並針對 FastReID 進行超參數微調，能大幅提升追蹤的穩定性與準確率（特別是 IDF1 分數）。本計畫旨在為專案建立這套高精度的整合架構，並預先準備好對應的訓練環境。

因專案目前暫無專用的車輛資料集，故採取 **Ultralytics 原生整合架構（做法 A）**。此做法將維持現有追蹤邏輯的穩定性，並備妥未來可一鍵執行的訓練鷹架。

## 預定修改項目 (Proposed Changes)

### 1. 追蹤器設定檔更新 (Tracker Configuration)
維持 `Ultralytics` 原生的 BoT-SORT 整合方式，以確保系統穩定性。
- **目標檔案**：`autocam_tracker/tracking/custom_botsort_reid.yaml`
- **修改內容**：
  保持 `with_reid: True`。未來當我們擁有 FastReID 轉換後的權重時，只需將設定檔內的 `model: auto` 更新為指定權重的相對路徑（例如：`model: weights/fastreid_finetuned.pt`）。在獲得專屬權重前，暫時依賴系統預設的 OSNet 進行特徵萃取。

### 2. 建立未來專用的微調訓練腳本 (Fine-tuning Scripts)
將論文實證的最佳超參數直接封裝為 Python 訓練腳本。未來獲取資料集後，即可直接訓練出與論文同等精度的模型。
- **新增檔案**：`scripts/train_fastreid.py`
  - **應用最佳參數**：
    - `learning_rate`: `0.00035`
    - `batch_size`: `256` （若未來 GPU 記憶體不足，將內建 Gradient Accumulation 邏輯來模擬此大小）
    - `epochs`: `10`
  - **權重匯出規劃**：訓練結束後，需將 FastReID 權重匯出為 Ultralytics BoT-SORT 可讀取的 `.pt` 或 `.onnx` 格式，並填入 `reid_model_path`。

- **新增檔案**：`scripts/train_yolo.py`
  - **應用最佳參數**：
    - `learning_rate`: `0.01`
    - `batch_size`: `16`
    - `epochs`: `20`

### 3. App Controller 支援自訂模型載入
確保應用程式介面能正確讀取客製化的 ReID 模型。
- **目標檔案**：`autocam_tracker/app/app_controller.py`
- **修改內容**：更新 `AppConfig`，讓 Tracker 在初始化時，可動態吃入並套用客製化的 FastReID 權重路徑。

## 目前實作進度 (Current Implementation)

### 1. Roboflow 資料下載
- **腳本**：`scripts/download_roboflow_dataset.py`
- **用途**：從 Roboflow 下載 `stanford_car-yaayi` 的 YOLOv8 格式資料集，供 YOLO 偵測模型微調使用。
- **執行前準備**：在專案根目錄建立 `.env`：
```bash
ROBOFLOW_API_KEY=您的真實金鑰
```

- **執行方式**：
```bash
python scripts/download_roboflow_dataset.py
```

如需切換 Roboflow workspace、project、version 或格式，可使用：
```bash
python scripts/download_roboflow_dataset.py --workspace yarok077-gmail-com --project stanford_car-yaayi --version 1 --format yolov8
```

### 2. YOLO 偵測模型微調
- **腳本**：`scripts/train_yolo.py`
- **用途**：使用 Roboflow 下載後的 `data.yaml` 微調車輛偵測模型。
- **預設超參數**：
  - `learning_rate`: `0.01`
  - `batch_size`: `16`
  - `epochs`: `20`

先確認訓練設定：
```bash
python scripts/train_yolo.py --data path/to/data.yaml --dry-run
```

開始訓練：
```bash
python scripts/train_yolo.py --data path/to/data.yaml
```

訓練完成後，可將輸出的偵測權重填入 `autocam_tracker/config/default_config.json`：
```json
{
  "model_path": "runs/train/yolo_vehicle_detector/weights/best.pt"
}
```

### 3. FastReID 訓練入口
- **腳本**：`scripts/train_fastreid.py`
- **用途**：檢查 ReID 資料夾、統計身份與影像數、寫出 `training_plan.json`，並在提供 FastReID repo 與 config 後接上正式訓練。
- **預設超參數**：
  - `learning_rate`: `0.00035`
  - `batch_size`: `256`
  - `epochs`: `10`

目前預期的簡易 ReID 資料夾格式：
```text
datasets/vehicle_reid/
  car_001/
    image_001.jpg
    image_002.jpg
  car_002/
    image_001.jpg
```

先檢查資料與寫出訓練計畫：
```bash
python scripts/train_fastreid.py --dry-run
```

若已準備 FastReID repo 與正式 config：
```bash
python scripts/train_fastreid.py --run --fastreid-root path/to/fast-reid --config-file path/to/vehicle_reid_config.yaml
```

> 注意：Roboflow 的 YOLO detection dataset 主要提供車輛框選與類別標籤，不能直接等同於 ReID 訓練資料。FastReID 需要同一車輛身份的標籤，否則模型無法學到「同一台車」的外觀特徵。

### 3.1 Global identity 候選合併
- **腳本**：`scripts/cluster_vehicle_identity_tracks.py`
- **用途**：將多段影片切出的 tracker candidates 以 ReID embedding 聚合，並支援 `--manual-merge-csv` 只套用人工確認過的同車 track group。
- **建議流程**：
```bash
python scripts/cluster_vehicle_identity_tracks.py \
  --candidates-root datasets/vehicle_identity_candidates_videoplayback_batch \
  --output-dir datasets/vehicle_identity_videoplayback_global \
  --model weights/fastreid_videoplayback_auto/fastreid_vehicle_reid.torchscript \
  --similarity-threshold 0.90 \
  --manual-merge-csv datasets/vehicle_identity_videoplayback_manual_global_merge.csv \
  --clear-output
```
- **重要原則**：低門檻自動合併容易把外觀相近但不同的車放進同一個 identity；寧可少合併，也不要把錯誤正樣本放進 FastReID 訓練。
- **Warm-start 訓練**：若已有上一版影片 ReID checkpoint，可用 `--init-weights` 接續微調：
```bash
python scripts/train_fastreid.py --run \
  --fastreid-root external/fastreid \
  --config-file configs/fastreid/vehicle_bootstrap_bagtricks.yml \
  --fastreid-datasets-root datasets/fastreid_videoplayback_global \
  --data_dir datasets/vehicle_identity_videoplayback_global/train \
  --output_dir weights/fastreid_videoplayback_global_warmstart \
  --init-weights weights/fastreid_videoplayback_auto/model_best.pth \
  --batch_size 32 --epochs 10 --device cuda:0
```

### 4. App 端套用自訂 ReID 權重
- **設定檔**：`autocam_tracker/config/default_config.json`
- **Runtime 行為**：當 `tracker` 為 `botsort_reid` 且 `reid_model_path` 有值時，`AppController` 會動態產生 `.dynamic_botsort_reid.yaml`，把 BoT-SORT ReID 的 `model` 指向自訂權重。

範例：
```json
{
  "model_path": "yolo26n.pt",
  "reid_model_path": "weights/fastreid_finetuned/best.onnx",
  "tracker": "botsort_reid"
}
```

## 驗證計畫 (Verification Plan)

### 自動化測試 (Automated Tests)
- 單元測試：模擬權重轉檔後的載入過程，確保 `AppController` 能正確產生自訂 ReID tracker config。
- 腳本測試：確保 `train_fastreid.py` 與 `train_yolo.py` 的預設參數皆正確鎖定在論文推薦值。
- 資料檢查測試：確保 `train_fastreid.py` 能統計 ReID 資料夾內的身份數與影像數。

### 人工驗證 (Manual Verification)
- 當獲得資料集後，實際跑一次 `train_fastreid.py`，確認 Loss 值能在 10 個 Epoch 內收斂。
- 將生成的模型套用至 Live View 介面，觀察同一車輛在遮蔽後是否能正確維持相同 ID。

---

## 參考文獻 (Reference)
* Yu-Jen Chen, Tzu-Chia Tung, Jia-Hao Kang, Yu-Chin Chu, Chiou-Shann Fuh, Ping-Han Chen, Yu-Tang Liao, Chung Ming Yang, Li-Jin Huang, Shu-Ru Huang. "Cross-Camera Multi-Target Vehicle Tracking". *The 37th IPPR Conference on Computer Vision, Graphics, and Image Processing (CVGIP 2024)*. August 18–20, 2024, National Dong Hwa University, Hualien.
