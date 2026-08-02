import os
from pathlib import Path

# Base Directory Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_DIR = BASE_DIR / "dataset"
MODELS_DIR = BASE_DIR / "models"
DEFAULT_DB_PATH = MODELS_DIR / "face_db.pkl"
LEGACY_DB_PATH = BASE_DIR / "face_db.pkl"

# Model Configuration
DEFAULT_MODEL_NAME = "buffalo_l"
DETECTION_SIZE = (640, 640)
SIMILARITY_THRESHOLD = 0.40

# Camera & UI Settings
DEFAULT_CAMERA_INDEX = 0
CAPTURE_TARGET_COUNT = 10
SUPPORTED_EXTENSIONS = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp")

# Ensure required directories exist
DATASET_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)
