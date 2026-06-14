# Git 提交流程指南 (標準 Pull Request 工作流程)

如果您是從 GitHub 下載專案的 ZIP 壓縮檔，資料夾內並不會包含 Git 的版本控制資訊（缺少 `.git` 資料夾）。若您修改了程式碼並希望推送到 GitHub，為了**保留原有的歷史紀錄並進行安全的程式碼審查 (Code Review)**，請依循以下標準業界流程。

## 第一階段：事前準備與環境同步

這階段的目的是讓本地端資料夾與 GitHub 上的遠端儲存庫建立連結，並同步歷史紀錄。

**1. 初始化 Git 儲存庫**
```bash
git init
```
*說明：這會在當前資料夾建立隱藏的 `.git` 資料夾，開始追蹤版本。*

**2. 連結遠端儲存庫 (Remote)**
```bash
git remote add origin https://github.com/您的帳號/您的專案名稱.git
```

**3. 下載遠端最新的歷史紀錄**
```bash
git fetch
```

**4. 對齊歷史紀錄（保留本地修改）**
```bash
git reset origin/main
```
*說明：這個指令是**最關鍵的一步**！它會把本地端的版本指向 GitHub 上最新的狀態，但**不會去更動您目前資料夾裡的任何檔案**。這樣 Git 就會把您的修改視為「尚未提交的新變更」。*


## 第二階段：身分設定與主分支對齊

**1. 設定提交者身分**
如果您沒有設定身分，Git 將不允許您 Commit。
```bash
git config user.name "您的名稱"
git config user.email "您的信箱@example.com"
```

**2. 對齊主分支名稱**
確保本地預設分支名稱與遠端一致：
```bash
git branch -m main
```


## 第三階段：同步 GitHub 上的最新版本

當 GitHub 上的專案已經有比較新的版本時，可以用以下流程把本地端同步到最新狀態。

**1. 先確認目前本地端狀態**
```bash
git status
```

如果畫面顯示 `working tree clean`，代表目前沒有尚未提交的本地修改，可以直接同步。

**2. 確認遠端儲存庫位置**
```bash
git remote -v
```

請確認 `origin` 指向正確的 GitHub 專案，例如：
```bash
https://github.com/SCP600/cartracking.git
```

如果還沒有設定遠端儲存庫，可以使用：
```bash
git remote add origin https://github.com/SCP600/cartracking.git
```

**3. 抓取遠端最新資訊**
```bash
git fetch origin --prune
```

*說明：`fetch` 只會更新本地端對遠端分支的認識，還不會修改目前工作區的檔案。`--prune` 會清掉已經在 GitHub 上刪除的遠端分支紀錄。*

**4. 檢查本地端是否落後遠端**
```bash
git status --short --branch
```

如果看到類似以下訊息，代表本地端 `main` 落後 GitHub 上的 `origin/main`：
```bash
## main...origin/main [behind 7]
```

也可以用以下指令查看本地與遠端各自多出幾個 commit：
```bash
git rev-list --left-right --count main...origin/main
```

若結果是：
```bash
0	7
```

代表本地端沒有比遠端多出的 commit，但遠端比本地端多 7 個 commit，通常可以安全地快轉同步。

**5. 將本地端更新到 GitHub 最新版本**
```bash
git pull --ff-only origin main
```

*說明：`--ff-only` 代表只允許「快轉更新」。如果本地端和遠端歷史已經分岔，Git 會停止並提醒您處理，避免自動產生不清楚的 merge commit。*

**6. 再次確認同步結果**
```bash
git status --short --branch
```

如果顯示：
```bash
## main...origin/main
```

且沒有 `[behind]` 或 `[ahead]`，代表本地端已經和 GitHub 上的 `main` 同步。

### 如果本地端有尚未提交的修改

如果 `git status` 顯示有修改過的檔案，建議先選擇其中一種方式處理，再進行同步。

**方式 A：把本地修改正式提交**
```bash
git add .
git commit -m "簡短描述您的本地修改"
git pull --ff-only origin main
```

**方式 B：暫時收起本地修改**
```bash
git stash push -m "暫存同步前的本地修改"
git pull --ff-only origin main
git stash pop
```

*說明：如果 `git stash pop` 後發生衝突，請依照 Git 顯示的檔案逐一修正，再重新 `git add` 與 `git commit`。*


## 第四階段：標準 Pull Request (PR) 提交流程

在業界標準中，我們**絕對不會直接把修改推送到 `main` 主分支**。所有的修改都應該在一個獨立的分支上完成，再透過發起 Pull Request 讓其他人或自己在合併前進行確認。

**1. 建立並切換到全新的開發分支 (Branch)**
每次要開發新功能或修復 Bug 時，請先開一個以該功能命名的分支：
```bash
git checkout -b feature/您的新功能名稱
```
*(範例：`git checkout -b feature/fastreid-integration`)*

**2. 將所有變更加入暫存區**
```bash
git add .
```

**3. 提交變更並寫下紀錄**
```bash
git commit -m "簡短描述您修改了什麼"
```

**4. 將新分支推送到 GitHub**
注意，推播時的分支名稱必須與您剛才建立的分支名稱相同：
```bash
git push origin feature/您的新功能名稱
```

**5. 發起 Pull Request (PR)**
推送完成後，前往您的 GitHub 專案首頁。您會看到畫面上方跳出一個綠色的按鈕寫著 **"Compare & pull request"**。
點擊它，填寫您的修改細節，然後建立 PR。確認沒有問題後，再點擊 **"Merge pull request"** 將這些程式碼正式合併進 `main` 之中！
