# 階層式機位錨定資料庫 (Hierarchical Shot-Based Anchor Database) 實作計畫

這個思路非常清晰且專業！將跨視角的特徵全部混在同一個池子裡，確實會造成距離空間（Embedding Space）混亂，導致 ReID 失效。

您提出的**「全域 ID (GID) -> 個別機位鏡頭 (Shot) 分支」**架構，完美利用了「同機位下特徵高度穩定」的優勢，同時避免了跨機位的特徵污染。

## 核心設計理念：階層式特徵管理
資料庫結構將改為樹狀結構：
```json
{
  "Global_ID_1": {
    "label": "目標車輛 A",
    "shots": {
      "shot_1": [ feature_1, feature_2 ], // 第一個鏡頭的特徵
      "shot_2": [ feature_3 ],            // 第二個鏡頭的特徵 (可能是不同視角)
      "shot_5": [ feature_4, feature_5 ]  // 第五個鏡頭的特徵
    }
  }
}
```
**運作邏輯：**
- 當系統播放到 `shot_2` 時，追蹤器**只會**拿當下的車輛去跟 `Global_ID_1` 底下的 `shot_2` 特徵集做比對。
- 如果 `shot_2` 尚未建檔，系統自然無法辨識，您只需在該鏡頭手動點選車輛並指定為 `Global_ID_1`，系統就會建立 `shot_2` 分支並存入特徵。
- 這樣既保證了同鏡頭內的極高辨識率，又能透過最外層的 `Global_ID_1` 將整部影片的追蹤結果完美串連起來。

## 系統修改 (Proposed Changes)

### 1. 新增階層式資料庫類別 (Hierarchical Database)

#### [NEW] `autocam_tracker/identity/hierarchical_anchor_database.py`
- 負責管理 `Global ID -> Shot ID -> Features` 的多層結構。
- 實作 `add_feature(global_id: int, shot_id: int, feature: np.ndarray)` 供 UI 呼叫，將特徵精準存入對應的 Shot 分支。
- 實作 `match(shot_id: int, detection_feature: np.ndarray) -> tuple[int, float]` 讓 Pipeline 呼叫，**只在指定的 `shot_id` 分支內尋找匹配的 Global ID**。
- 提供存檔與讀檔功能 (`save_to_file`, `load_from_file`)，讓手動標註的成果能持久化保存。

### 2. 整合至 Pipeline 與 Registry

#### [MODIFY] `autocam_tracker/app/pipeline_worker.py`
- 初始化 `HierarchicalAnchorDatabase`。
- 在每個 frame，傳入當前的 `shot_id` 與提取的特徵給 Database 進行比對。
- 若在當前 `shot_id` 中找到符合的 Global ID，強制覆寫 Detection 的 ID，實現當前鏡頭內的錨定。

#### [MODIFY] `autocam_tracker/data/recognized_vehicle_registry.py`
- 調整邏輯以優先採納 Hierarchical Database 回傳的 Global ID，其次才使用系統預設的短期記憶 ReID 邏輯。

### 3. 使用者介面 (UI) 擴充

#### [MODIFY] `autocam_tracker/ui/main_window.py` 及相關 UI 面板
- 在畫面側邊或底部新增一個「全域 ID 管理器 (Global ID Manager)」。
- 列出目前已建立的 Global IDs。
- 當使用者選取某台車時，點擊「綁定至 GID X」，系統會自動擷取特徵，並將其歸入 **「GID X -> 目前的 Shot ID」** 分支中。

## Verification Plan

### Manual Verification
1. 播放影片，進入 `shot_1`。手動建立 `Global_ID_1`，並將目標車輛綁定至此 ID。系統應立即在 `shot_1` 中死死咬住該車。
2. 影片切換至 `shot_2`，目標車輛視角大變。因為 `shot_2` 分支尚未建立特徵，目標可能丟失或被視為新車。
3. 手動將該車綁定至 `Global_ID_1`。系統會在 `Global_ID_1` 下建立 `shot_2` 特徵分支。
4. 倒轉影片，從 `shot_1` 播放至 `shot_2`。觀察系統是否能在切換至 `shot_2` 的瞬間，透過 `shot_2` 分支的特徵，無縫找回並維持 `Global_ID_1` 的身份。

---

## 整體程式架構梳理 (Overall Program Architecture Summary)

為了確保我們對資料流與元件職責的理解完全一致，以下是加入「階層式錨定資料庫」後的整體系統運作架構：

### 1. 影像輸入與切片層 (Input & Segmentation Layer)
* **`VideoSource`**: 負責逐幀讀取影片。
* **`SceneCutDetector`**: 負責監控畫面差異，當偵測到鏡頭切換 (Camera Cut) 時，自動將目前的 `shot_id` 加 1。這個 `shot_id` 是整個系統區分不同機位視角的關鍵指標。

### 2. 偵測與特徵提取層 (Detection & Feature Layer)
* **`YOLO26Detector`**: 找出畫面中所有車輛的 Bounding Box，並賦予一個暫時的 `local_track_id`。
* **`RuntimeReIDFeatureExtractor`**: 為每一台被偵測到的車輛，提取其外觀特徵 (Feature Embedding)，這些特徵是一組高維度向量，代表車輛當下的長相。

### 3. 全域錨定比對層 (Global Anchoring Layer) **[本次核心新增]**
* **`HierarchicalAnchorDatabase`**: 這是系統最新的大腦，儲存著使用者「手動」教導系統的知識。
  * **儲存結構**: 記錄著 `Global_ID` 底下各個 `shot_id` 所對應的專屬特徵庫。
  * **比對邏輯**: 當 `PipelineWorker` 拿到車輛特徵後，會**優先**去這個資料庫查詢：*「在目前的 `shot_id` 庫中，這台車的特徵長得像誰？」*
  * **覆寫機制**: 如果比對分數極高（大於設定的閾值），系統就強制認定該車的 `global_vehicle_id` 為資料庫記載的 Global ID，這被稱為「絕對錨定 (Absolute Anchoring)」。

### 4. 短期記憶與關聯層 (Short-Term Registry Layer)
* **`RecognizedVehicleRegistry`**: 處理沒有被「全域錨定比對層」命中的車輛。
  * 負責維護一個視窗大小內的短期記憶（利用連續影格間的位置與短期外觀相似度），以彌補手動特徵庫未能涵蓋的盲區。
  * 結合來自上層的強制 GID 覆寫結果，來修正短期的 Tracking Tree。

### 5. 目標管理與追蹤狀態層 (Identity & Status Layer)
* **`GlobalIdentityManager`**: 負責管理當下 UI 「正在聚焦」的目標。
  * 它關注的是狀態機的切換：目前處於 "Tracking" (追蹤中)、"CameraCut" (剛切換鏡頭) 還是 "SearchingTarget" (丟失尋找中)。
  * 若目標在切換鏡頭後短暫遺失，它會呼叫 `ReacquireEngine` 嘗試重新找回。在有了新的 `HierarchicalAnchorDatabase` 後，丟失的機率將大幅降低。

### 6. 使用者互動層 (User Interface Layer)
* **UI 面板 (Main Window / Anchor DB Panel)**:
  * 顯示即時影像與目前的 `shot_id`。
  * 提供「建立新 GID」與「綁定目前目標」的按鈕。
  * 當使用者點擊「綁定」時，UI 將目標的特徵與當前的 `shot_id` 傳送給 `HierarchicalAnchorDatabase`，完成該機位下此目標的特徵建檔。
