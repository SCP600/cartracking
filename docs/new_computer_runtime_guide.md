# New Computer Runtime Guide

> Purpose: give a new machine or a new coding agent enough context to run this project with the already-trained YOLO and FastReID models.
> Scope: runtime inference and Tkinter UI. This guide does not require retraining.

---

## 1. Important Rule

The GitHub repository contains source code and config, but it does not contain trained model weights.

These paths are intentionally ignored by git:

```text
weights/
runs/
datasets/
*.pt
*.onnx
*.torchscript
```

That means a fresh clone cannot use the custom-trained models until the model files are copied from the original training machine or downloaded from a separate artifact store.

---

## 2. Required Runtime Artifacts

For the current custom model setup, copy these files into the same relative paths in the new clone:

```text
runs/train/yolo_vehicle_detector_gpu_640/weights/best.pt
weights/fastreid_videoplayback_global_warmstart/fastreid_vehicle_reid.torchscript
```

The default runtime config expects exactly those paths:

```json
{
  "model_path": "runs/train/yolo_vehicle_detector_gpu_640/weights/best.pt",
  "reid_model_path": "weights/fastreid_videoplayback_global_warmstart/fastreid_vehicle_reid.torchscript"
}
```

Optional but useful ReID artifacts for future evaluation or export:

```text
weights/fastreid_videoplayback_global_warmstart/fastreid_vehicle_reid.onnx
weights/fastreid_videoplayback_global_warmstart/model_best.pth
weights/fastreid_videoplayback_global_warmstart/model_final.pth
weights/fastreid_videoplayback_global_warmstart/config.yaml
weights/fastreid_videoplayback_global_warmstart/metrics.json
weights/fastreid_videoplayback_global_warmstart/training_plan.json
```

The `datasets/` folder is not required for normal inference. Copy it only if the new machine needs to continue training, rebuild vehicle identity candidates, or audit the training crops.

---

## 3. Clone And Enter The Project

```powershell
git clone https://github.com/SCP600/cartracking.git
cd cartracking
```

If a specific PR or branch is needed, check it out before installing:

```powershell
git fetch origin
git switch main
git pull
```

---

## 4. Create The Conda Environment

Use the project environment name `cartracking`.

```powershell
conda create -n cartracking python=3.11 -y
conda activate cartracking
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
```

If PowerShell cannot find `conda`, use Anaconda Prompt, or call `conda.exe` directly:

```powershell
C:\Users\<you>\anaconda3\Scripts\conda.exe create -n cartracking python=3.11 -y
C:\Users\<you>\anaconda3\Scripts\conda.exe activate cartracking
```

---

## 5. Install PyTorch For The Target Machine

The project uses Ultralytics, YOLO, and TorchScript ReID. PyTorch must match the hardware.

### NVIDIA GPU

Use the official PyTorch selector:

```text
https://pytorch.org/get-started/locally/
```

After installing the selected command, verify CUDA:

```powershell
python -c "import torch; print('cuda:', torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu only')"
```

If this prints `cuda: False`, set the runtime config to CPU or reinstall the correct PyTorch CUDA build.

### CPU Only

CPU works, but inference will be slower.

```powershell
python -m pip install --upgrade torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

Then set the runtime config device to CPU as described below.

---

## 6. Copy The Trained Models

Create the folders if they do not exist:

```powershell
mkdir runs\train\yolo_vehicle_detector_gpu_640\weights
mkdir weights\fastreid_videoplayback_global_warmstart
```

Copy the trained files:

```text
best.pt
fastreid_vehicle_reid.torchscript
```

Expected final layout:

```text
cartracking/
  runs/
    train/
      yolo_vehicle_detector_gpu_640/
        weights/
          best.pt
  weights/
    fastreid_videoplayback_global_warmstart/
      fastreid_vehicle_reid.torchscript
```

Verify the files exist:

```powershell
Test-Path runs\train\yolo_vehicle_detector_gpu_640\weights\best.pt
Test-Path weights\fastreid_videoplayback_global_warmstart\fastreid_vehicle_reid.torchscript
```

Both commands must print `True`.

---

## 7. Check Runtime Config

Open:

```text
autocam_tracker/config/default_config.json
```

For an NVIDIA GPU machine, this is expected:

```json
"device": 0
```

For a CPU-only machine, change it to:

```json
"device": "cpu"
```

Keep these model paths unless the copied artifacts are placed somewhere else:

```json
"model_path": "runs/train/yolo_vehicle_detector_gpu_640/weights/best.pt",
"reid_model_path": "weights/fastreid_videoplayback_global_warmstart/fastreid_vehicle_reid.torchscript",
"tracker": "botsort_reid_custom"
```

The runtime GID ReID memory settings can stay at their defaults:

```json
"gid_reid_memory_size": 24,
"gid_reid_match_threshold": 0.82,
"gid_reid_cross_shot_threshold": 0.86,
"gid_reid_margin": 0.04,
"gid_reid_duplicate_similarity": 0.985
```

---

## 8. Smoke Checks

Run these from the project root inside the `cartracking` environment.

```powershell
python -c "import cv2, numpy, PIL, ultralytics, torch; print('packages ok')"
```

Check the YOLO model:

```powershell
python -c "from ultralytics import YOLO; YOLO('runs/train/yolo_vehicle_detector_gpu_640/weights/best.pt'); print('custom yolo ok')"
```

Check the TorchScript ReID model:

```powershell
python -c "import torch; m=torch.jit.load('weights/fastreid_videoplayback_global_warmstart/fastreid_vehicle_reid.torchscript', map_location='cpu').eval(); print('custom reid ok')"
```

Run the project smoke tests:

```powershell
python -m pytest tests\test_core_smoke.py -q
```

If `pytest` is missing:

```powershell
python -m pip install pytest
```

Do not run bare `pytest` unless you intentionally want to include `external/fastreid` upstream tests. Those tests can require optional packages that are not needed for normal runtime.

---

## 9. Start The Tkinter App

```powershell
python autocam_tracker\main.py
```

Expected behavior:

- The app opens a Tkinter UI.
- The tracker dropdown can use `botsort_reid_custom` for the custom ReID path.
- GID list entries show `mem` once runtime ReID feature memory starts collecting embeddings.
- If ReID loading fails, the app should still run, but GID ReID memory will be disabled.

---

## 10. Common Problems

### `FileNotFoundError` or model loading error

Check that these files exist:

```powershell
Test-Path runs\train\yolo_vehicle_detector_gpu_640\weights\best.pt
Test-Path weights\fastreid_videoplayback_global_warmstart\fastreid_vehicle_reid.torchscript
```

If either is `False`, copy the trained model from the original machine or update `default_config.json` to the actual path.

### CUDA is not detected

Run:

```powershell
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

If CUDA is unavailable and the machine has no compatible NVIDIA GPU, set:

```json
"device": "cpu"
```

### App is slow

Use a smaller input size or CPU-safe settings:

```json
"imgsz": 640,
"device": "cpu"
```

If GPU is available, prefer a matching CUDA PyTorch install and:

```json
"device": 0
```

### Custom ReID does not appear to help

Check the UI:

- `mem0` means that GID feature memory is not collecting embeddings.
- `mem` increasing means the runtime memory is active.
- Vehicle rows may show `R0.xx` when ReID similarity scores are available.

If `mem` stays at `0`, verify `reid_model_path` and check the console for ReID loading warnings.

---

## 11. What Not To Commit

Do not commit these generated or heavy artifacts unless the user explicitly requests a release-artifact workflow:

```text
datasets/
weights/
runs/
*.pt
*.onnx
*.torchscript
```

For sharing trained models, use one of these instead:

- A GitHub Release asset
- A private cloud drive folder
- An internal artifact server
- A zip file copied outside git

