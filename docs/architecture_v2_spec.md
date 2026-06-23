# AutoCamTracker V1 - 架構優化與重構詳細實作計畫 (Implementation Plan)

為了確保接下來的實作階段毫無缺漏且不會產生誤解，本計畫將先前的概念收斂為嚴謹的檔案層級 (File-Level) 規格書。明確定義每個模組的增刪改細節。

## Proposed Changes

### 1. 介面精簡與移除冗餘元件 (UI & Data Pruning)
為了最大化預覽畫面並擺脫繁雜的舊追蹤邏輯，我們將徹底刪除以下不再需要的模組。

#### [DELETE] autocam_tracker/ui/vehicle_list_panel.py
#### [DELETE] autocam_tracker/ui/recognized_vehicle_list_panel.py
#### [DELETE] autocam_tracker/ui/global_identity_list_panel.py
#### [DELETE] autocam_tracker/data/recognized_vehicle_registry.py
#### [DELETE] autocam_tracker/data/candidate_ranker.py
#### [DELETE] autocam_tracker/data/detection_history.py

---

### 2. 核心大腦：階層式錨定資料庫與身分識別 (Hierarchical Identity)
建立新的身分識別大腦，並優化既有的特徵擷取器以支援高頻刷新。

#### [NEW] autocam_tracker/identity/hierarchical_anchor_database.py
- **實作內容**：
  - 建立 `Global_ID -> Shot_ID -> [Deep_ReID_Features]` 的樹狀字典結構。
  - `add_anchor(global_id, shot_id, feature)`：手動綁定時，將特徵寫入該分支。
  - `match_candidates(shot_id, candidate_features)`：傳入一個 Batch 的特徵，僅與該 `Shot_ID` 下的特徵進行 Cosine Similarity 比對，回傳得分最高的候選人與分數。

#### [MODIFY] autocam_tracker/identity/reid_feature_extractor.py
- **修改內容**：
  - 確認模型推論支援批次 (Batch) 處理。當 Pipeline 送來多張車輛縮圖時，能夠在一次 inference 中極速回傳多個 Embedding，以滿足「每 5 幀全圖重刷」的效能要求。

#### [MODIFY] autocam_tracker/identity/global_identity_manager.py
- **修改內容**：
  - 移除舊有對 Registry 的依賴。改為只負責記錄「當前正在追蹤的 `Global_ID` 是誰」以及其狀態 (Tracking / Searching Target 等)。

---

### 3. 機位場景辨識與大視野虛擬運鏡 (Scene & Framing)
賦予系統 1 毫秒內辨識舊機位的能力，並實作「虛擬跟焦裁切」。

#### [MODIFY] autocam_tracker/utils/scene_cut.py
- **修改內容**：
  - 加入輕量級場景辨識 (Low-Cost Scene Recognition)。
  - 當偵測到 Cut 時，將畫面縮小至 64x64 計算均值哈希 (aHash) 或色彩分佈。
  - 建立 `Scene_DB`。若新畫面的特徵與舊有 `Shot_ID` 的特徵相似度大於閾值，則**回傳該舊有的 `Shot_ID`**；否則建立新的 `Shot_ID`。

#### [MODIFY] autocam_tracker/framing/crop_controller.py
- **修改內容**：
  - 實作平滑運鏡 (Virtual Crop Panning)。
  - YOLO 偵測依然維持全圖尺寸，但 `CropController` 在產生輸出畫面時，根據目標的 `bbox` 套用指數移動平均 (EMA) 進行平滑插值，確保框出來的 1/8~1/4 畫面能流暢跟隨目標，不產生突兀抖動。

---

### 4. 主控管線與高頻自動糾錯 (Pipeline Worker)
整合原生 YOLO 追蹤與每 5 幀一次的深層特徵糾錯邏輯。

#### [MODIFY] autocam_tracker/app/pipeline_worker.py
- **修改內容**：
  - **移除舊 Registry 呼叫**，改為初始化並串接 `HierarchicalAnchorDatabase`。
  - **原生 YOLO 信任機制**：一般影格下，直接信任 YOLO `botsort` 輸出的 `local_track_id`。只要目標未消失，就持續更新其 `bbox` 供畫面裁切。
  - **每 5 幀高頻重刷 (Periodic Target Refresh)**：
    1. 當 `frame_counter % 5 == 0` 時，將畫面上「所有車輛」的 bbox 裁切送入 `ReidFeatureExtractor` 批次算出 Deep Features。
    2. 送入 `HierarchicalAnchorDatabase.match_candidates(current_shot_id, features)` 找出最高分車輛。
    3. 如果找到的最高分車輛與當前正在追蹤的 `local_track_id` 不同，且分數差距超過 `Hysteresis Margin`，則觸發**自動奪取 (Auto Re-Anchoring)**，更新追蹤目標，達成高速糾錯。

#### [MODIFY] autocam_tracker/utils/image_utils.py
- **修改內容**：
  - 找到畫 YOLO Bounding Box 的函式 (例如 `draw_boxes` 或 `draw_detections`)。
  - **強制放大** `cv2.putText` 的 `fontScale` (例如設定為 1.5 到 2.0) 與 `thickness` (設定為 2 到 3)，並使用高對比底色（如深藍底白字）確保 Demo 視覺清晰。

---

### 5. 使用者介面重組 (UI Layout)
將精簡後的模組組合到主畫面上，最大化視覺體驗。

#### [NEW] autocam_tracker/ui/anchor_db_panel.py
- **實作內容**：
  - 一個簡單的側邊欄或底部面板。
  - 列出當前活躍的 `Global_ID`。
  - 提供「Bind to GID」按鈕：點擊左側列表的某台車後，按下綁定，觸發向資料庫寫入當前 `Shot_ID` 特徵的事件。

#### [MODIFY] autocam_tracker/ui/main_window.py
- **修改內容**：
  - 拔除 Notebook 多頁籤架構。
  - 將原有的 `Raw View` 與 `Cropped View` 排列配置放大，使其佔滿主要空間。
  - 將新建立的 `Anchor DB Panel` 安置於右側或底部。
  - 確認 `TimelinePanel` 等必要組件依然能正常掛載與運作。
