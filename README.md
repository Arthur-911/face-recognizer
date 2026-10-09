# InsightFace Real-Time Face Recognition & Dataset Collector (Secured)

A high-performance, modular Python pipeline for real-time facial dataset collection, 512-dimensional feature embedding extraction using InsightFace (`buffalo_l`), and live webcam face detection and recognition with **built-in presentation attack detection (liveness verification)** and **hardened security protections**.

---

## Overview

This repository provides an end-to-end framework for face recognition:
- **Dataset Collector**: GUI and CLI tools to capture and label reference facial images with path traversal protection and paced capture.
- **Embedding Pipeline**: Deep feature extraction utilizing InsightFace arcface models (`buffalo_l`) with hardware acceleration (CUDA / DirectML / CPU) and safe, unpickled storage (`.npz`).
- **Real-Time Recognition & Liveness**: Real-time webcam inference engine with dynamic HUD overlays, similarity confidence metrics, eye-blink/motion liveness detection against 2D photo spoofing, and FPS telemetry.

---

## Key Security & Architectural Enhancements

- **Safe NumPy Storage (`.npz`)**: Replaced Python `pickle` with compressed NumPy arrays (`allow_pickle=False`) to eliminate Remote Code Execution (RCE) deserialization vulnerabilities.
- **Input Sanitization & Path Confinement**: Strictly validates names with regex whitelisting and verifies directory confinement (`is_relative_to(DATASET_DIR)`) to block path traversal attacks (`../`).
- **Presentation Attack Detection (Liveness)**: Computes Eye Aspect Ratio (EAR) from 68 3D landmarks to verify natural eye blinks and micro-movements, detecting static 2D printouts or phone screens.
- **Numerical Stability**: Safe embedding normalization with $\epsilon$ offsets to prevent zero-norm division and `NaN` propagation.
- **Deterministic Resource Cleanup**: Camera capture feeds and OpenCV windows are wrapped in `try ... finally` blocks to prevent DirectShow hardware device lockups.

---

## Directory Architecture

```text
face-recognizer/
├── src/
│   ├── __init__.py          # Package initialization
│   ├── config.py            # Centralized system configurations & security constants
│   ├── utils.py             # Execution providers, sanitizers, safe serializers & liveness tracker
│   ├── capture.py           # GUI & CLI face dataset collector module (secured)
│   ├── trainer.py           # Feature embedding extractor & safe database builder
│   └── recognizer.py        # Real-time webcam recognition engine with liveness HUD
├── scripts/
│   ├── capture_dataset.py   # Entry script for dataset collection
│   ├── train_model.py       # Entry script for model training
│   └── run_recognition.py   # Entry script for live recognition
├── dataset/                 # Raw image dataset directory grouped by identity
├── models/                  # Trained embedding databases (face_db.npz)
│   └── .gitkeep
├── app.py                   # Root alias for dataset collection
├── train.py                 # Root alias for model training
├── use.py                   # Root alias for live recognition feed
├── .gitignore               # Optimized Git exclusion rules (ignores weights, datasets, .npz, .pkl)
├── LICENSE                  # MIT Open Source License
├── README.md                # Project documentation
└── requirements.txt         # Project dependencies
```

---

## Installation

### Prerequisites
- Python 3.8 to 3.12
- C++ Build Tools (required for InsightFace C extensions)
- Optional: CUDA Toolkit or DirectML for GPU hardware acceleration

### Setup Environment

1. Clone the repository:
   ```bash
   git clone https://github.com/Arthur-911/face-recognizer.git
   cd face-recognizer
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   # On Windows:
   venv\Scripts\activate
   # On Linux/macOS:
   source venv/bin/activate
   ```

3. Install required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

---

## Quick Start Guide

### 🚀 One-Click Launchers (Windows)

You don't need to type terminal commands! Just double-click these files in the folder:

1. **`1_Capture_and_Train.bat`**: Double-click to launch the camera, enter your name, and auto-capture photos. When you close the window, it **automatically trains** the face recognition model!
2. **`2_Start_Recognizer.bat`**: Double-click anytime to launch the GPU-accelerated real-time recognition feed!

---

### Command-Line Usage

### Step 1: Collect Facial Dataset

Launch the interactive capture GUI to collect face samples:

```bash
python app.py
```
*Alternatively, run `python scripts/capture_dataset.py`.*

- Enter the target individual's identifier (e.g., `John`).
- Click **Start Auto Capture** to acquire 10 dataset images into `dataset/John/`. Auto-capture paces shots to allow varied facial angles.

---

### Step 2: Train Model & Extract Embeddings

Extract 512-dimensional normalized embeddings for all persons in `dataset/`:

```bash
python train.py
```
*Alternatively, run `python scripts/train_model.py`.*

The model automatically detects hardware acceleration (CUDA / DirectML / CPU) and generates the safe feature database in `models/face_db.npz`.

---

### Step 3: Run Real-Time Face Recognition & Liveness

Launch live recognition via webcam feed:

```bash
python use.py
```
*Alternatively, run `python scripts/run_recognition.py`.*

- **Green Bounding Box + `LIVE: PASS`**: Verified match and active liveness (blink detected).
- **Yellow Bounding Box + `LIVE: VERIFYING`**: Recognized identity, awaiting blink/micro-motion.
- **Red Alert Box + `LIVE: SUSPECT`**: Static photo spoof suspected (no motion / no blinks).
- **Red Bounding Box + `Unknown`**: Unindexed or below-threshold detections.
- Press **`q`** or **`ESC`** to exit the camera stream.

---

## Technical Specifications

| Parameter | Value / Description |
| :--- | :--- |
| Core Model | InsightFace `buffalo_l` |
| Feature Dimension | 512-d normalized floating point vector |
| Database Format | Compressed NumPy Archive (`.npz`, unpickled) |
| Distance Metric | Cosine Similarity (`np.dot` on normalized vectors) |
| Similarity Threshold | Default: `0.40` (configurable in `src/config.py`) |
| Anti-Spoof Liveness | Eye Aspect Ratio (EAR) blink detection + Micro-motion tracking |
| Supported Execution Backends | `CUDAExecutionProvider`, `DmlExecutionProvider`, `CPUExecutionProvider` |

---

## License

This project is open-source under the [MIT License](LICENSE).
