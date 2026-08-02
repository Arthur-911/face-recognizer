"""
Face model training module: extracts 512-dimensional face embeddings via InsightFace
and constructs the reference feature database (face_db.pkl).
"""

import os
import glob
import pickle
import numpy as np
import cv2
from pathlib import Path
import insightface
from insightface.app import FaceAnalysis

from src.config import (
    DATASET_DIR,
    DEFAULT_DB_PATH,
    LEGACY_DB_PATH,
    DEFAULT_MODEL_NAME,
    DETECTION_SIZE,
    SUPPORTED_EXTENSIONS
)
from src.utils import get_available_providers

def train_face_model(
    dataset_dir: Path = DATASET_DIR,
    output_db: Path = DEFAULT_DB_PATH,
    model_name: str = DEFAULT_MODEL_NAME
):
    """
    Scans dataset_dir for subfolders containing face images for each person,
    extracts face embeddings using InsightFace's model with hardware acceleration,
    and saves the trained embeddings database to output_db.
    """
    providers = get_available_providers()
    print(f"[*] Initializing InsightFace model '{model_name}' on Execution Providers: {providers}...")

    app = FaceAnalysis(name=model_name, providers=providers)
    app.prepare(ctx_id=0, det_size=DETECTION_SIZE)

    if not dataset_dir.exists():
        print(f"[!] Error: Dataset directory '{dataset_dir}' not found!")
        print("Please capture images first or populate the dataset directory.")
        return False

    person_dirs = [d for d in dataset_dir.iterdir() if d.is_dir()]

    if not person_dirs:
        print(f"[!] No person folders found inside '{dataset_dir}'.")
        return False

    print(f"[*] Found {len(person_dirs)} person folder(s): {[d.name for d in person_dirs]}")

    face_database = {}
    total_images_processed = 0

    for person_dir in person_dirs:
        person_name = person_dir.name
        image_files = []

        for ext in SUPPORTED_EXTENSIONS:
            image_files.extend(glob.glob(str(person_dir / ext)))
            image_files.extend(glob.glob(str(person_dir / ext.upper())))

        image_files = sorted(list(set(image_files)))

        if not image_files:
            print(f"[!] Skipping '{person_name}': No images found.")
            continue

        print(f"\n[*] Processing '{person_name}' ({len(image_files)} images)...")
        embeddings_list = []

        for img_path in image_files:
            img = cv2.imread(img_path)
            if img is None:
                print(f"  [!] Could not read image: {Path(img_path).name}")
                continue

            faces = app.get(img)

            if len(faces) == 0:
                print(f"  [!] No face detected in: {Path(img_path).name}")
                continue

            # Select primary face by bounding box area
            primary_face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))

            embedding = primary_face.embedding
            norm_embedding = embedding / np.linalg.norm(embedding)

            embeddings_list.append(norm_embedding)
            total_images_processed += 1
            print(f"  [+] Extracted embedding from {Path(img_path).name}")

        if embeddings_list:
            face_database[person_name] = embeddings_list
            print(f"[*] Trained {len(embeddings_list)} face samples for '{person_name}'")
        else:
            print(f"[!] Failed to extract any valid faces for '{person_name}'")

    if not face_database:
        print("\n[!] Training failed: No valid face embeddings were extracted.")
        return False

    db_data = {
        "model_name": model_name,
        "database": face_database
    }

    output_db.parent.mkdir(parents=True, exist_ok=True)
    with open(output_db, "wb") as f:
        pickle.dump(db_data, f)

    # Save a copy to legacy root path for seamless backwards compatibility
    with open(LEGACY_DB_PATH, "wb") as f:
        pickle.dump(db_data, f)

    print("\n" + "=" * 55)
    print(" Training Complete")
    print(f" Total Persons: {len(face_database)}")
    print(f" Total Face Embeddings Processed: {total_images_processed}")
    print(f" Trained Database Saved to: {output_db.resolve()}")
    print("=" * 55)
    print("-> Next step: run 'python use.py' or 'python scripts/run_recognition.py'")
    return True

if __name__ == "__main__":
    train_face_model()
