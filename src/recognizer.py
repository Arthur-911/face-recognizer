"""
Real-time webcam face recognition pipeline with InsightFace embeddings matching.
"""

import os
import time
import pickle
import numpy as np
import cv2
from pathlib import Path
import insightface
from insightface.app import FaceAnalysis

from src.config import (
    DEFAULT_DB_PATH,
    LEGACY_DB_PATH,
    DEFAULT_MODEL_NAME,
    DETECTION_SIZE,
    SIMILARITY_THRESHOLD,
    DEFAULT_CAMERA_INDEX
)
from src.utils import get_available_providers

def run_face_recognition(
    db_file: Path = DEFAULT_DB_PATH,
    model_name: str = DEFAULT_MODEL_NAME,
    sim_threshold: float = SIMILARITY_THRESHOLD
):
    """
    Loads trained face embeddings database and streams webcam to identify faces in real-time.
    """
    target_db = db_file if db_file.exists() else LEGACY_DB_PATH

    if not target_db.exists():
        print(f"[!] Error: Trained database file not found at '{db_file}' or '{LEGACY_DB_PATH}'")
        print("-> Please run 'python train.py' first to build embeddings from your dataset.")
        return

    print(f"[*] Loading face embeddings database from '{target_db.name}'...")
    with open(target_db, "rb") as f:
        db_data = pickle.load(f)

    face_database = db_data.get("database", {})
    if not face_database:
        print("[!] Error: Loaded face database is empty.")
        return

    print(f"[+] Loaded embeddings for {len(face_database)} person(s): {list(face_database.keys())}")

    providers = get_available_providers()
    print(f"\n[*] Initializing InsightFace model '{model_name}' on Providers: {providers}...")
    app = FaceAnalysis(name=model_name, providers=providers)
    app.prepare(ctx_id=0, det_size=DETECTION_SIZE)

    print("\n[*] Opening webcam stream...")
    cap = cv2.VideoCapture(DEFAULT_CAMERA_INDEX, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(DEFAULT_CAMERA_INDEX)

    if not cap.isOpened():
        print("[!] Error: Could not open camera feed.")
        return

    print("\n" + "=" * 55)
    print(" Camera Feed Active | Press 'q' or 'ESC' to exit")
    print("=" * 55)

    prev_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[!] Failed to capture webcam frame.")
            break

        frame = cv2.flip(frame, 1)
        display_frame = frame.copy()

        faces = app.get(frame)

        for face in faces:
            bbox = face.bbox.astype(int)
            x1, y1, x2, y2 = bbox[0], bbox[1], bbox[2], bbox[3]

            query_embed = face.embedding
            query_embed = query_embed / np.linalg.norm(query_embed)

            best_match = "Unknown"
            best_score = -1.0

            for person_name, embeddings_list in face_database.items():
                for ref_embed in embeddings_list:
                    similarity = float(np.dot(query_embed, ref_embed))
                    if similarity > best_score:
                        best_score = similarity
                        best_match = person_name

            if best_score >= sim_threshold:
                label = f"{best_match} ({best_score * 100:.1f}%)"
                color = (0, 255, 127)  # Vibrant Green
            else:
                label = f"Unknown ({best_score * 100:.1f}%)" if best_score > 0 else "Unknown"
                color = (0, 0, 255)    # Red

            cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)

            # Draw HUD corner reticles
            w = x2 - x1
            h = y2 - y1
            l = int(min(w, h) * 0.15)
            cv2.line(display_frame, (x1, y1), (x1 + l, y1), color, 3)
            cv2.line(display_frame, (x1, y1), (x1, y1 + l), color, 3)
            cv2.line(display_frame, (x2, y1), (x2 - l, y1), color, 3)
            cv2.line(display_frame, (x2, y1), (x2, y1 + l), color, 3)
            cv2.line(display_frame, (x1, y2), (x1 + l, y2), color, 3)
            cv2.line(display_frame, (x1, y2), (x1, y2 - l), color, 3)
            cv2.line(display_frame, (x2, y2), (x2 - l, y2), color, 3)
            cv2.line(display_frame, (x2, y2), (x2, y2 - l), color, 3)

            # Label banner overlay
            text_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            banner_y1 = max(0, y1 - text_size[1] - 12)
            cv2.rectangle(
                display_frame, 
                (x1, banner_y1), 
                (x1 + text_size[0] + 10, banner_y1 + text_size[1] + 10), 
                color, 
                -1
            )
            cv2.putText(
                display_frame, 
                label, 
                (x1 + 5, banner_y1 + text_size[1] + 4), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.6, 
                (0, 0, 0), 
                2
            )

        curr_time = time.time()
        fps = 1.0 / (curr_time - prev_time) if (curr_time - prev_time) > 0 else 30.0
        prev_time = curr_time

        provider_name = providers[0].replace("ExecutionProvider", "")
        cv2.putText(
            display_frame, 
            f"FPS: {fps:.1f} | Backend: {provider_name}", 
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

        cv2.imshow("InsightFace Real-Time Recognition", display_frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            print("\n[*] Exiting recognition stream...")
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_face_recognition()
