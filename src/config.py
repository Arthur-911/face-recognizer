import os
from pathlib import Path

# Base Directory Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_DIR = BASE_DIR / "dataset"
MODELS_DIR = BASE_DIR / "models"

# Database Storage: Migrated to safe compressed NumPy format (.npz)
DEFAULT_DB_PATH = MODELS_DIR / "face_db.npz"
LEGACY_DB_NPZ = BASE_DIR / "face_db.npz"
LEGACY_MODELS_PKL = MODELS_DIR / "face_db.pkl"
LEGACY_ROOT_PKL = BASE_DIR / "face_db.pkl"

# Model Configuration
DEFAULT_MODEL_NAME = "buffalo_l"
DETECTION_SIZE = (640, 640)
SIMILARITY_THRESHOLD = 0.40
EPSILON = 1e-10

# Presentation Attack & Liveness Configuration
ENABLE_LIVENESS_CHECK = True
LIVENESS_EAR_THRESHOLD = 0.20  # Eye Aspect Ratio threshold for blink detection
LIVENESS_WINDOW_FRAMES = 20    # Sliding window frame count for motion/blink analysis

# Camera & Performance Optimization Settings
DEFAULT_CAMERA_INDEX = 0
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
TARGET_CAMERA_FPS = 30
PROCESS_EVERY_N_FRAMES = 2  # Interleaved AI inference frequency for maximum display FPS
CAPTURE_TARGET_COUNT = 10
AUTO_CAPTURE_INTERVAL = 0.35  # Minimum seconds between auto-captures for pose variation
SUPPORTED_EXTENSIONS = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp")

# Ensure required directories exist
DATASET_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

