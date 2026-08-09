# InsightFace Real-Time Face Recognition & Dataset Collector

A high-performance, modular Python pipeline for real-time facial dataset collection, 512-dimensional feature embedding extraction using InsightFace (`buffalo_l`), and live webcam face detection and recognition.

---

## Overview

This repository provides an end-to-end framework for face recognition:
- **Dataset Collector**: GUI and CLI tools to capture and label reference facial images.
- **Embedding Pipeline**: Deep feature extraction utilizing InsightFace arcface models (`buffalo_l`) with GPU acceleration (CUDA / DirectML).
- **Real-Time Recognition**: Real-time webcam inference engine with dynamic HUD overlays, similarity confidence metrics, and FPS telemetry.

---

## Directory Architecture

```text
facedetection/
├── src/
│   ├── __init__.py          # Package initialization
│   ├── config.py            # Centralized system configurations & paths
│   ├── utils.py             # Execution provider detector & helper utilities
│   ├── capture.py           # GUI & CLI face dataset collector module
│   ├── trainer.py           # Feature embedding extractor & database builder
│   └── recognizer.py        # Real-time webcam recognition engine
├── scripts/
│   ├── capture_dataset.py   # Entry script for dataset collection
│   ├── train_model.py       # Entry script for model training
│   └── run_recognition.py   # Entry script for live recognition
├── dataset/                 # Raw image dataset directory grouped by identity
├── models/                  # Trained embedding databases (face_db.pkl)
│   └── .gitkeep
├── app.py                   # Root alias for dataset collection
├── train.py                 # Root alias for model training
├── use.py                   # Root alias for live recognition feed
├── .gitignore               # Optimized Git exclusion rules
├── LICENSE                  # MIT Open Source License
├── README.md                # Project documentation
└── requirements.txt         # Project dependencies
```

---

## Installation

### Prerequisites
- Python 3.8 or higher
- C++ Build Tools (required for certain InsightFace C extensions)
- Optional: CUDA Toolkit or DirectML for GPU hardware acceleration

### Setup Environment

1. Clone the repository:
   ```bash
   git clone https://github.com/Arthur-911/facedetection.git
   cd facedetection
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

### Step 1: Collect Facial Dataset

Launch the interactive capture GUI to collect face samples:

```bash
python app.py
```
*Alternatively, run `python scripts/capture_dataset.py`.*

- Enter the target individual's identifier (e.g., `John`).
- Click **Start Auto Capture** to acquire 10 dataset images into `dataset/John/`.

---

### Step 2: Train Model & Extract Embeddings

Extract 512-dimensional normalized embeddings for all persons in `dataset/`:

```bash
python train.py
```
*Alternatively, run `python scripts/train_model.py`.*

The model automatically detects hardware acceleration (CUDA / DirectML / CPU) and generates the feature database in `models/face_db.pkl`.

---

### Step 3: Run Real-Time Face Recognition

Launch live recognition via webcam feed:

```bash
python use.py
```
*Alternatively, run `python scripts/run_recognition.py`.*

- **Green Bounding Box**: Matches identified person with similarity confidence score.
- **Red Bounding Box**: Displayed for unindexed or below-threshold detections (`Unknown`).
- Press **`q`** or **`ESC`** to exit the camera stream.

---

## Technical Specifications

| Parameter | Value / Description |
| :--- | :--- |
| Core Model | InsightFace `buffalo_l` |
| Feature Dimension | 512-d normalized floating point vector |
| Distance Metric | Cosine Similarity (`np.dot` on normalized vectors) |
| Similarity Threshold | Default: `0.40` (configurable in `src/config.py`) |
| Supported Execution Backends | `CUDAExecutionProvider`, `DmlExecutionProvider`, `CPUExecutionProvider` |

---

## License

This project is open-source under the [MIT License](file:///C:/itznan/facedetection/LICENSE).
