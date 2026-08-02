#!/usr/bin/env python3
"""
Script entrypoint to train face embeddings model.
"""
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from src.trainer import train_face_model

if __name__ == "__main__":
    train_face_model()
