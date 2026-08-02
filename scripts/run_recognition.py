#!/usr/bin/env python3
"""
Script entrypoint for real-time webcam face recognition.
"""
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from src.recognizer import run_face_recognition

if __name__ == "__main__":
    run_face_recognition()
