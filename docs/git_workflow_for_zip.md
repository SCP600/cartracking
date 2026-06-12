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


## 第三階段：標準 Pull Request (PR) 提交流程

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
