# AutoCamTracker Environment Setup

> Target: macOS and Windows development machines  
> Project scope: Python + Tkinter + OpenCV + Ultralytics YOLO26n + BoT-SORT / ByteTrack + NumPy + Pillow  
> Principle: install only what V1 needs for the racing-video demo.

---

## 1. Read First

Before working on this project, read:

- `skills.md`
- `docs/spec.md`

The project is intentionally limited to a desktop Python demo. Do not add web frameworks, databases, cloud services, OCR systems, or custom training pipelines unless the user explicitly updates the spec.

---

## 2. Recommended Versions

Use Python 3.11 as the default development version.

Python 3.12 may also work, but Python 3.11 is a conservative choice for OpenCV, PyTorch, and Ultralytics compatibility across macOS and Windows.

Required runtime packages:

```text
ultralytics
opencv-python
numpy
pillow
lap
```

Optional package:

```text
mss
```

Use `mss` for the Tkinter "Select Region" source when testing real-time YOLO on a screen area.

---

## 3. Project Folder

Clone or copy the project to a local folder.

Example:

```text
car-tracking/
  docs/
  skills.md
  autocam_tracker/
```

All commands below assume you are inside the project root:

```bash
cd car-tracking
```

### 3.1 Anaconda / Conda Setup

Use this path if the machine already uses Anaconda or Miniconda.

Create the project environment:

```bash
conda create -n cartracking python=3.11 -y
conda activate cartracking
python -m pip install --upgrade pip setuptools wheel
```

If `conda` is not available in Windows PowerShell, either use Anaconda Prompt or call `conda.exe` directly:

```powershell
C:\Users\<you>\anaconda3\Scripts\conda.exe create -n cartracking python=3.11 -y
C:\Users\<you>\anaconda3\Scripts\conda.exe run -n cartracking python --version
```

For the first V1 demo, install CPU PyTorch first. This avoids accidentally downloading a large CUDA build on machines that do not need GPU acceleration:

```bash
python -m pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

Then install the project packages:

```bash
python -m pip install --no-cache-dir ultralytics opencv-python numpy pillow lap
```

Optional screen-region capture:

```bash
python -m pip install --no-cache-dir mss
```

Windows Traditional Chinese consoles may fail when tools print emoji or non-CP950 characters. If that happens, run:

```powershell
$env:PYTHONIOENCODING = "utf-8"
```

or call the environment Python directly:

```powershell
C:\Users\<you>\anaconda3\envs\cartracking\python.exe -c "from ultralytics import YOLO; YOLO('yolo26n.pt'); print('yolo26n ok')"
```

---

## 4. macOS Setup

### 4.1 Install Python

Recommended options:

1. Install Python 3.11 from python.org.
2. Or install with Homebrew:

```bash
brew install python@3.11
```

Check Python:

```bash
python3.11 --version
```

If `python3.11` is not available, try:

```bash
python3 --version
```

### 4.2 Check Tkinter

Tkinter is required for the V1 desktop UI.

```bash
python3.11 -c "import tkinter; print('tkinter ok')"
```

If this fails on Homebrew Python, install the matching Tk package:

```bash
brew install python-tk@3.11
```

Then retry the check.

### 4.3 Create Virtual Environment

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
```

Your shell prompt should show `(.venv)`.

### 4.4 Install Project Dependencies

CPU / Apple Silicon default:

```bash
python -m pip install -U ultralytics opencv-python numpy pillow lap
```

Optional screen-region capture:

```bash
python -m pip install mss
```

### 4.5 Apple Silicon Note

Apple Silicon Macs do not use NVIDIA CUDA. PyTorch can use Apple's MPS backend when available, but the first V1 demo should still work on CPU.

Check acceleration status:

```bash
python -c "import torch; print('mps:', hasattr(torch.backends, 'mps') and torch.backends.mps.is_available())"
```

---

## 5. Windows Setup

### 5.1 Install Python

Recommended options:

1. Install Python 3.11 from python.org and enable "Add python.exe to PATH".
2. Or install with winget:

```powershell
winget install Python.Python.3.11
```

Check Python:

```powershell
py -3.11 --version
```

If the Python launcher is not available, try:

```powershell
python --version
```

### 5.2 Check Tkinter

```powershell
py -3.11 -c "import tkinter; print('tkinter ok')"
```

If this fails, reinstall Python from python.org and make sure Tcl/Tk support is included.

### 5.3 Create Virtual Environment

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip setuptools wheel
```

If PowerShell blocks activation, run:

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

Then activate again:

```powershell
.\.venv\Scripts\Activate.ps1
```

### 5.4 Install Project Dependencies

CPU default:

```powershell
python -m pip install -U ultralytics opencv-python numpy pillow lap
```

Optional screen-region capture:

```powershell
python -m pip install mss
```

### 5.5 NVIDIA GPU Option

For NVIDIA GPU acceleration on Windows, install the PyTorch build that matches your CUDA setup before installing or testing Ultralytics.

Use the official PyTorch selector:

```text
https://pytorch.org/get-started/locally/
```

After installing the selected PyTorch command, verify CUDA:

```powershell
python -c "import torch; print('cuda:', torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu only')"
```

Then install the remaining project dependencies:

```powershell
python -m pip install -U ultralytics opencv-python numpy pillow lap
```

---

## 6. Verify the Environment

Run these checks after installing dependencies.

### 6.1 Basic Package Check

macOS:

```bash
python -c "import cv2, numpy, PIL, ultralytics, lap; print('packages ok')"
```

Windows:

```powershell
python -c "import cv2, numpy, PIL, ultralytics, lap; print('packages ok')"
```

### 6.2 YOLO26n Model Check

The first run may download `yolo26n.pt`.

```bash
python -c "from ultralytics import YOLO; model = YOLO('yolo26n.pt'); print('yolo26n ok')"
```

Use the same command in PowerShell on Windows.

### 6.3 OpenCV Video Check

This checks whether OpenCV can create a video reader.

```bash
python -c "import cv2; cap = cv2.VideoCapture(0); print('camera opened:', cap.isOpened()); cap.release()"
```

If no webcam is available, this can print `False`; that is acceptable for local-video-only development.

### 6.4 Screen Region Capture Check

This checks whether `mss` can capture the primary monitor.

```bash
python -c "from mss import MSS; sct=MSS(); img=sct.grab(sct.monitors[1]); print('screen capture:', img.size); sct.close()"
```

Use the same command in PowerShell on Windows.

---

## 7. Expected Runtime Flow

After implementation begins, the normal run command should be:

macOS:

```bash
source .venv/bin/activate
python autocam_tracker/main.py
```

Windows:

```powershell
.\.venv\Scripts\Activate.ps1
python autocam_tracker\main.py
```

If the entry point changes later, update this section and keep `docs/spec.md` aligned.

For real-time screen-region testing, click `Select Region`, drag the capture area, then press `Start`. Press `Esc` or right-click while selecting to cancel. After a valid region is selected, the app should immediately show a captured preview in the Tkinter views before YOLO starts.

---

## 8. Local Video Assets

Do not commit large racing videos into the repository unless the user explicitly asks.

Recommended local folder:

```text
assets/
  videos/
    reference_race.mp4
```

The app should allow the user to select a local video file from the Tkinter UI.

---

## 9. Tracker Config Files

The V1 demo should start with BoT-SORT and keep ByteTrack as fallback.

Expected files after implementation:

```text
autocam_tracker/
  tracking/
    custom_botsort.yaml
    custom_bytetrack.yaml
```

Default tracker direction:

```text
BoT-SORT + ReID first
BoT-SORT baseline A/B test
ByteTrack fallback
ReID on for current demo
```

If ReID is too slow on a CPU-only machine, switch the Tkinter tracker dropdown back to `botsort`.

---

## 10. Troubleshooting

### `ModuleNotFoundError`

Make sure the virtual environment is activated:

macOS:

```bash
source .venv/bin/activate
```

Windows:

```powershell
.\.venv\Scripts\Activate.ps1
```

Then reinstall dependencies:

```bash
python -m pip install -U ultralytics opencv-python numpy pillow lap
```

### Tkinter import fails

macOS Homebrew Python:

```bash
brew install python-tk@3.11
```

Windows:

Reinstall Python from python.org and include Tcl/Tk support.

### YOLO model download fails

The first `YOLO("yolo26n.pt")` call needs network access. If the computer is offline or behind a restricted network, download the model on another machine and place the `.pt` file in the project root or a configured model directory.

### GPU is not used on Windows

Check CUDA through PyTorch:

```powershell
python -c "import torch; print(torch.cuda.is_available())"
```

If it prints `False`, reinstall PyTorch using the command generated by the official PyTorch selector for your CUDA version.

### OpenCV cannot open video

Try:

- Use `.mp4` encoded with H.264.
- Move the video path to a simple ASCII path.
- Avoid cloud-synced folders for first testing.
- Test with a short local clip before using a full race broadcast.

---

## 11. Official References

- Python venv: https://docs.python.org/3/library/venv.html
- Ultralytics install: https://docs.ultralytics.com/quickstart
- Ultralytics PyPI: https://pypi.org/project/ultralytics/
- PyTorch install selector: https://pytorch.org/get-started/locally/
