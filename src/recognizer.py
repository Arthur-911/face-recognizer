"""
Real-time webcam face recognition pipeline with InsightFace embeddings matching,
Presentation Attack Detection (Liveness verification), non-blocking threaded camera capture,
and interleaved GPU inference for maximum display frame rates.
"""

import os
import time
from pathlib import Path
import cv2
import numpy as np
import insightface
from insightface.app import FaceAnalysis

from src.config import (
    DEFAULT_DB_PATH,
    LEGACY_DB_NPZ,
    LEGACY_MODELS_PKL,
    LEGACY_ROOT_PKL,
    DEFAULT_MODEL_NAME,
    DETECTION_SIZE,
    SIMILARITY_THRESHOLD,
    DEFAULT_CAMERA_INDEX,
    ENABLE_LIVENESS_CHECK,
    FRAME_WIDTH,
    FRAME_HEIGHT,
    TARGET_CAMERA_FPS,
    PROCESS_EVERY_N_FRAMES,
)
from src.utils import (
    get_available_providers,
    safe_load_database,
    normalize_embedding,
    LivenessTracker,
    ThreadedCamera,
)


def run_face_recognition(
    db_file: Path = DEFAULT_DB_PATH,
    model_name: str = DEFAULT_MODEL_NAME,
    sim_threshold: float = SIMILARITY_THRESHOLD,
    enable_liveness: bool = ENABLE_LIVENESS_CHECK,
    process_interval: int = PROCESS_EVERY_N_FRAMES,
):
    """
    Loads trained face embeddings database safely and streams webcam to identify faces in real-time.
    Uses non-blocking threaded camera acquisition and interleaved GPU inference to deliver silky-smooth 30-60 FPS.
    """
    print("[*] Loading face embeddings database...")
    try:
        face_database, trained_model = safe_load_database(
            db_file,
            fallback_paths=[LEGACY_DB_NPZ, LEGACY_MODELS_PKL, LEGACY_ROOT_PKL]
        )
    except FileNotFoundError as e:
        print(f"[!] Error: {e}")
        print("-> Please run 'python train.py' first to build embeddings from your dataset.")
        return

    if not face_database:
        print("[!] Error: Loaded face database is empty.")
        return

    print(f"[+] Loaded embeddings for {len(face_database)} person(s): {list(face_database.keys())}")

    providers = get_available_providers()
    print(f"\n[*] Initializing InsightFace model '{model_name}' on Providers: {providers}...")
    app = FaceAnalysis(
        name=model_name,
        providers=providers,
        allowed_modules=['detection', 'recognition', 'landmark_3d_68']
    )
    app.prepare(ctx_id=0, det_size=DETECTION_SIZE)

    # Pre-build vectorized embeddings matrix for O(1) batch similarity search
    db_names = []
    db_embeddings = []
    for person_name, embeddings_list in face_database.items():
        for ref_embed in embeddings_list:
            db_names.append(person_name)
            db_embeddings.append(ref_embed)
    db_matrix = np.array(db_embeddings, dtype=np.float32) if db_embeddings else np.empty((0, 512), dtype=np.float32)

    print("\n[*] Starting non-blocking threaded camera capture...")
    camera = ThreadedCamera(
        src=DEFAULT_CAMERA_INDEX,
        width=FRAME_WIDTH,
        height=FRAME_HEIGHT,
        fps=TARGET_CAMERA_FPS,
    ).start()

    time.sleep(0.3)  # Allow sensor pipeline to warm up

    if not camera.isOpened():
        print("[!] Error: Could not open camera feed.")
        camera.release()
        return

    print("\n" + "=" * 60)
    print(" Camera Feed Active | Press 'q' or 'ESC' to exit")
    print(f" Anti-Spoof Liveness: {'ENABLED' if enable_liveness else 'DISABLED'}")
    print(f" Inference Mode: Interleaved (every {process_interval} frames) with Threaded I/O")
    print("=" * 60)

    liveness_trackers = {}
    cached_render_items = []
    frame_idx = 0
    prev_time = time.time()
    fps_smooth = 30.0

    try:
        while True:
            ret, frame = camera.read()
            if not ret or frame is None:
                time.sleep(0.005)
                continue

            frame = cv2.flip(frame, 1)
            display_frame = frame.copy()
            frame_idx += 1

            # Interleaved inference: Run heavy neural nets every N frames
            if frame_idx % process_interval == 0:
                faces = app.get(frame)
                new_render_items = []

                for i, face in enumerate(faces):
                    bbox = face.bbox.astype(int)
                    x1, y1 = max(0, bbox[0]), max(0, bbox[1])
                    x2, y2 = min(frame.shape[1], bbox[2]), min(frame.shape[0], bbox[3])

                    raw_embed = getattr(face, "embedding", None)
                    if raw_embed is None:
                        continue

                    query_embed = normalize_embedding(raw_embed)

                    if len(db_matrix) > 0:
                        sim_scores = np.dot(db_matrix, query_embed)
                        best_idx = int(np.argmax(sim_scores))
                        best_score = float(sim_scores[best_idx])
                        best_match = db_names[best_idx]
                    else:
                        best_match = "Unknown"
                        best_score = -1.0

                    is_recognized = best_score >= sim_threshold
                    if is_recognized:
                        identity_label = f"{best_match} ({best_score * 100:.1f}%)"
                        box_color = (0, 255, 127)  # Vibrant green
                    else:
                        identity_label = f"Unknown ({best_score * 100:.1f}%)" if best_score > 0 else "Unknown"
                        box_color = (0, 0, 255)    # Red

                    liveness_badge = ""
                    if enable_liveness:
                        tracker_key = best_match if is_recognized else f"face_{i}"
                        if tracker_key not in liveness_trackers:
                            liveness_trackers[tracker_key] = LivenessTracker()

                        badge, detail, _ = liveness_trackers[tracker_key].update(face)
                        liveness_badge = f" | {badge}"

                        if badge == "LIVE: SUSPECT":
                            box_color = (0, 0, 255)  # Alert Red

                    full_label = f"{identity_label}{liveness_badge}"
                    new_render_items.append((x1, y1, x2, y2, box_color, full_label))

                cached_render_items = new_render_items

            # Draw cached face boxes and HUD overlays on the fresh frame
            for (x1, y1, x2, y2, box_color, full_label) in cached_render_items:
                cv2.rectangle(display_frame, (x1, y1), (x2, y2), box_color, 2)

                # Draw corner reticles
                w, h = x2 - x1, y2 - y1
                l = int(min(w, h) * 0.15)
                if l > 0:
                    cv2.line(display_frame, (x1, y1), (x1 + l, y1), box_color, 3)
                    cv2.line(display_frame, (x1, y1), (x1, y1 + l), box_color, 3)
                    cv2.line(display_frame, (x2, y1), (x2 - l, y1), box_color, 3)
                    cv2.line(display_frame, (x2, y1), (x2, y1 + l), box_color, 3)
                    cv2.line(display_frame, (x1, y2), (x1 + l, y2), box_color, 3)
                    cv2.line(display_frame, (x1, y2), (x1, y2 - l), box_color, 3)
                    cv2.line(display_frame, (x2, y2), (x2 - l, y2), box_color, 3)
                    cv2.line(display_frame, (x2, y2), (x2, y2 - l), box_color, 3)

                # Label banner
                text_size, _ = cv2.getTextSize(full_label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
                banner_y1 = max(0, y1 - text_size[1] - 12)
                cv2.rectangle(
                    display_frame,
                    (x1, banner_y1),
                    (x1 + text_size[0] + 10, banner_y1 + text_size[1] + 10),
                    box_color,
                    -1
                )
                cv2.putText(
                    display_frame,
                    full_label,
                    (x1 + 5, banner_y1 + text_size[1] + 4),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (0, 0, 0),
                    2
                )

            # Performance telemetry calculation
            curr_time = time.time()
            frame_dt = curr_time - prev_time
            prev_time = curr_time
            instant_fps = 1.0 / frame_dt if frame_dt > 0 else 30.0
            fps_smooth = 0.85 * fps_smooth + 0.15 * instant_fps

            provider_name = providers[0].replace("ExecutionProvider", "")
            cv2.putText(
                display_frame,
                f"FPS: {fps_smooth:.1f} | Backend: {provider_name}",
                (15, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 255),
                2
            )
            cv2.putText(
                display_frame,
                "Press 'q' or 'ESC' to exit",
                (15, 55),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (200, 200, 200),
                1
            )

            cv2.imshow("InsightFace Real-Time Recognition (Secured)", display_frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:
                print("\n[*] Exiting recognition stream...")
                break

    finally:
        # Guarantee camera thread and window cleanup
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    run_face_recognition()
