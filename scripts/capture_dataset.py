#!/usr/bin/env python3
"""
Script entrypoint to launch dataset capture tool.
"""
import sys
from pathlib import Path

# Ensure project root is in python path
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from src.capture import launch_capture_gui

if __name__ == "__main__":
    launch_capture_gui()
