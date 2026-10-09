"""
Utility helper functions:
- Hardware execution provider detection
- Input sanitization & path traversal mitigation
- Safe NumPy database serialization & deserialization (anti-RCE)
- Safe embedding normalization (numerical stability)
- Liveness & Presentation Attack Detection (Anti-Spoofing)
"""

import os
import re
import warnings
from pathlib import Path
from collections import deque
import numpy as np

from src.config import (
    DATASET_DIR,
    EPSILON,
    LIVENESS_EAR_THRESHOLD,
    LIVENESS_WINDOW_FRAMES,
)

WINDOWS_RESERVED_NAMES = {
    "CON", "PRN", "AUX", "NUL",
    "COM1", "COM2", "COM3", "COM4", "COM5", "COM6", "COM7", "COM8", "COM9",
    "LPT1", "LPT2", "LPT3", "LPT4", "LPT5", "LPT6", "LPT7", "LPT8", "LPT9"
}


def get_available_providers():
    """
    Detects available ONNX Runtime execution providers.
    Prioritizes CUDA and DirectML before falling back to CPU execution.
    """
    try:
        import onnxruntime as ort
        available = ort.get_available_providers()
        providers = []
        if 'CUDAExecutionProvider' in available:
            providers.append('CUDAExecutionProvider')
        if 'DmlExecutionProvider' in available:
            providers.append('DmlExecutionProvider')
        providers.append('CPUExecutionProvider')
        return providers
    except ImportError:
        return ['CPUExecutionProvider']


def sanitize_person_name(name: str) -> str:
    """
    Validates and sanitizes person name input to prevent path traversal and filesystem issues.
    Enforces whitelist [a-zA-Z0-9_-], rejects path separators, and confines target path inside DATASET_DIR.
    """
    if not name or not isinstance(name, str):
        raise ValueError("Person name must be a non-empty string.")

    cleaned = name.strip()
    if not cleaned:
        raise ValueError("Person name cannot be blank.")

    if len(cleaned) > 50:
        raise ValueError("Person name exceeds maximum allowed length of 50 characters.")

    # Reject path traversal tokens and forbidden characters
    if ".." in cleaned or "/" in cleaned or "\\" in cleaned or "\0" in cleaned:
        raise ValueError("Person name contains illegal path traversal characters.")

    # Match against safe character whitelist (letters, numbers, spaces, hyphens, underscores)
    if not re.match(r"^[a-zA-Z0-9_\- ]+$", cleaned):
        raise ValueError("Person name contains invalid characters. Use only letters, numbers, spaces, '-' or '_'.")

    # Reject Windows reserved names
    base_check = cleaned.upper().split()[0]
    if base_check in WINDOWS_RESERVED_NAMES:
        raise ValueError(f"'{cleaned}' is a reserved system name and cannot be used.")

    # Ensure path resolution strictly stays inside DATASET_DIR
    target_dir = (DATASET_DIR / cleaned).resolve()
    dataset_base = DATASET_DIR.resolve()
    if not target_dir.is_relative_to(dataset_base):
        raise ValueError("Security violation: Target directory resolves outside DATASET_DIR.")

    return cleaned


def normalize_embedding(embedding: np.ndarray, eps: float = EPSILON) -> np.ndarray:
    """
    Safely computes L2-normalized embedding with epsilon to prevent division by zero / NaNs.
    """
    if embedding is None:
        raise ValueError("Embedding vector cannot be None.")

    emb = np.asarray(embedding, dtype=np.float32)
    norm = np.linalg.norm(emb)
    if norm < 1e-7:
        return np.zeros_like(emb, dtype=np.float32)

    return (emb / (norm + eps)).astype(np.float32)


def safe_save_database(
    filepath: Path,
    face_database: dict[str, list[np.ndarray]],
    model_name: str,
    legacy_path: Path | None = None
) -> None:
    """
    Safely saves face database using compressed NumPy archive (.npz).
    Completely eliminates Python pickle arbitrary code execution vulnerabilities.
    """
    save_dict = {
        "__meta_model__": np.array([model_name]),
        "__meta_names__": np.array(list(face_database.keys())),
    }

    for person_name, embeddings_list in face_database.items():
        arr = np.array(embeddings_list, dtype=np.float32)
        save_dict[f"emb_{person_name}"] = arr

    filepath.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(filepath, **save_dict)

    if legacy_path is not None and legacy_path != filepath:
        legacy_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(legacy_path, **save_dict)


def safe_load_database(
    primary_path: Path,
    fallback_paths: list[Path] | None = None
) -> tuple[dict[str, list[np.ndarray]], str]:
    """
    Safely loads face embeddings database without allowing arbitrary code execution.
    Prefers safe .npz format (allow_pickle=False).
    Falls back to legacy .pkl only if no .npz exists, printing an explicit security warning.
    """
    candidates = [primary_path]
    if fallback_paths:
        candidates.extend(fallback_paths)

    # 1. First priority: look for safe .npz databases
    for cand in candidates:
        npz_candidate = cand.with_suffix(".npz") if cand.suffix != ".npz" else cand
        if npz_candidate.exists():
            with np.load(npz_candidate, allow_pickle=False) as data:
                model_name = str(data["__meta_model__"][0]) if "__meta_model__" in data else "buffalo_l"
                names = [str(n) for n in data["__meta_names__"]]
                face_database = {}
                for name in names:
                    key = f"emb_{name}"
                    if key in data:
                        face_database[name] = [emb for emb in data[key]]
                return face_database, model_name

    # 2. Backward compatibility: legacy .pkl with security warning
    for cand in candidates:
        pkl_candidate = cand.with_suffix(".pkl") if cand.suffix != ".pkl" else cand
        if pkl_candidate.exists():
            import pickle
            warnings.warn(
                f"[SECURITY WARNING] Loading legacy pickle database at '{pkl_candidate}'. "
                "Pickle deserialization is vulnerable to arbitrary code execution if untrusted. "
                "Please run 'python train.py' to upgrade to a secure .npz database.",
                UserWarning,
                stacklevel=2
            )
            with open(pkl_candidate, "rb") as f:
                db_data = pickle.load(f)
            return db_data.get("database", {}), db_data.get("model_name", "buffalo_l")

    raise FileNotFoundError(f"No face database found at '{primary_path}' or any fallback path.")


class LivenessTracker:
    """
    Presentation Attack Detection (PAD) tracker to defend against 2D static spoofing.
    Tracks eye blinks via Eye Aspect Ratio (EAR) and head micro-movements across frames.
    """

    def __init__(self, window_size: int = LIVENESS_WINDOW_FRAMES, ear_thresh: float = LIVENESS_EAR_THRESHOLD):
        self.window_size = window_size
        self.ear_thresh = ear_thresh
        self.ear_history = deque(maxlen=window_size)
        self.center_history = deque(maxlen=window_size)
        self.blink_count = 0
        self.consec_below = 0
        self.verified = False

    @staticmethod
    def _calculate_eye_ear(pts: np.ndarray) -> float:
        """
        Calculates Eye Aspect Ratio (EAR) from 6 2D/3D eye landmarks.
        """
        v1 = np.linalg.norm(pts[1] - pts[5])
        v2 = np.linalg.norm(pts[2] - pts[4])
        h = np.linalg.norm(pts[0] - pts[3])
        return float((v1 + v2) / (2.0 * h + 1e-6))

    def update(self, face) -> tuple[str, str, tuple[int, int, int]]:
        """
        Updates liveness status for the detected face.
        Returns: (state_badge, detail_message, bgr_color)
        """
        # Calculate face bounding box center
        bbox = face.bbox
        center_x = (bbox[0] + bbox[2]) / 2.0
        center_y = (bbox[1] + bbox[3]) / 2.0
        self.center_history.append((center_x, center_y))

        # Check landmark 3d / 68-points for eye aspect ratio
        current_ear = None
        if hasattr(face, "landmark_3d_68") and face.landmark_3d_68 is not None:
            lm68 = face.landmark_3d_68
            left_eye_ear = self._calculate_eye_ear(lm68[36:42])
            right_eye_ear = self._calculate_eye_ear(lm68[42:48])
            current_ear = (left_eye_ear + right_eye_ear) / 2.0
        elif hasattr(face, "kps") and face.kps is not None and len(face.kps) == 5:
            # Fallback for 5-point landmarks: relative eye-to-mouth ratio
            kps = face.kps
            eye_dist = np.linalg.norm(kps[0] - kps[1])
            mouth_dist = np.linalg.norm(kps[3] - kps[4]) + 1e-6
            current_ear = float(eye_dist / mouth_dist * 0.15)

        if current_ear is not None:
            self.ear_history.append(current_ear)
            # Detect blink transitions
            if current_ear < self.ear_thresh:
                self.consec_below += 1
            else:
                if 1 <= self.consec_below <= 6:
                    self.blink_count += 1
                    self.verified = True
                self.consec_below = 0

        # Motion variance check: checks if face is an unnatural completely frozen static print
        motion_variance = 0.0
        if len(self.center_history) >= 10:
            centers = np.array(self.center_history)
            motion_variance = float(np.var(centers[:, 0]) + np.var(centers[:, 1]))

        if self.verified or self.blink_count >= 1:
            return "LIVE: PASS", f"Blink verified ({self.blink_count})", (0, 255, 127)

        if len(self.center_history) >= 15 and motion_variance < 0.15:
            return "LIVE: SUSPECT", "Static image suspected", (0, 0, 255)

        return "LIVE: VERIFYING", "Blink eyes to verify", (0, 215, 255)


class ThreadedCamera:
    """
    High-performance non-blocking threaded camera capture.
    Runs frame capture in a background daemon thread with MJPG compression,
    decoupling physical sensor I/O latency from the main processing thread.
    """
    def __init__(self, src=0, width=640, height=480, fps=30):
        import cv2
        self.cap = cv2.VideoCapture(src, cv2.CAP_DSHOW)
        if not self.cap.isOpened():
            self.cap = cv2.VideoCapture(src)

        if self.cap.isOpened():
            self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
            if width and height:
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            if fps:
                self.cap.set(cv2.CAP_PROP_FPS, fps)

        self.ret, self.frame = self.cap.read() if self.cap.isOpened() else (False, None)
        self.stopped = False
        import threading
        self.lock = threading.Lock()
        self.thread = threading.Thread(target=self._capture_loop, daemon=True)

    def start(self):
        self.thread.start()
        return self

    def _capture_loop(self):
        while not self.stopped:
            if not self.cap.isOpened():
                break
            ret, frame = self.cap.read()
            if not ret:
                continue
            with self.lock:
                self.ret = ret
                self.frame = frame

    def read(self):
        with self.lock:
            if self.frame is None:
                return self.ret, None
            return self.ret, self.frame.copy()

    def isOpened(self):
        return self.cap.isOpened()

    def release(self):
        self.stopped = True
        if self.thread.is_alive():
            self.thread.join(timeout=1.0)
        if self.cap.isOpened():
            self.cap.release()

