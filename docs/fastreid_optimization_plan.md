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
  - **自動轉檔功能**：訓練結束後，腳本會自動將 FastReID 的權重匯出並轉換為 Ultralytics 支援的 `.pt` 或 `.onnx` 格式。

- **新增檔案**：`scripts/train_yolo.py`
  - **應用最佳參數**：
    - `learning_rate`: `0.01`
    - `batch_size`: `16`
    - `epochs`: `20`

### 3. App Controller 支援自訂模型載入
確保應用程式介面能正確讀取客製化的 ReID 模型。
- **目標檔案**：`autocam_tracker/app/app_controller.py`
- **修改內容**：更新 `AppConfig`，讓 Tracker 在初始化時，可動態吃入並套用客製化的 FastReID 權重路徑。

## 驗證計畫 (Verification Plan)

### 自動化測試 (Automated Tests)
- 單元測試：模擬權重轉檔後的載入過程，確保 `YOLO26Detector` 能正確解析自訂的 ReID 模型檔案而不崩潰。
- 腳本測試：確保 `train_fastreid.py` 的預設參數（LR、Batch Size）皆正確鎖定在論文推薦值。

### 人工驗證 (Manual Verification)
- 當獲得資料集後，實際跑一次 `train_fastreid.py`，確認 Loss 值能在 10 個 Epoch 內收斂。
- 將生成的模型套用至 Live View 介面，觀察同一車輛在遮蔽後是否能正確維持相同 ID。

---

## 參考文獻 (Reference)
* Yu-Jen Chen, Tzu-Chia Tung, Jia-Hao Kang, Yu-Chin Chu, Chiou-Shann Fuh, Ping-Han Chen, Yu-Tang Liao, Chung Ming Yang, Li-Jin Huang, Shu-Ru Huang. "Cross-Camera Multi-Target Vehicle Tracking". *The 37th IPPR Conference on Computer Vision, Graphics, and Image Processing (CVGIP 2024)*. August 18–20, 2024, National Dong Hwa University, Hualien.
