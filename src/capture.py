"""
Dataset collection module for capturing reference face images via camera feed.
Includes input sanitization, path boundary enforcement, safe camera resource cleanup,
and paced auto-capture for diverse facial angles.
"""

import os
import sys
import time
import cv2
from pathlib import Path

from src.config import (
    DATASET_DIR,
    CAPTURE_TARGET_COUNT,
    DEFAULT_CAMERA_INDEX,
    AUTO_CAPTURE_INTERVAL,
)
from src.utils import sanitize_person_name


def capture_dataset_cli(person_name: str, count: int = CAPTURE_TARGET_COUNT):
    """
    CLI / OpenCV interactive interface for capturing dataset images with sanitization and safe cleanup.
    """
    try:
        clean_name = sanitize_person_name(person_name)
    except ValueError as e:
        print(f"[!] Validation Error: {e}")
        return

    person_dir = (DATASET_DIR / clean_name).resolve()
    if not person_dir.is_relative_to(DATASET_DIR.resolve()):
        print("[!] Security Violation: Output directory resolves outside dataset folder.")
        return

    person_dir.mkdir(parents=True, exist_ok=True)

    print(f"[*] Starting camera for dataset capture: '{clean_name}'...")
    cap = cv2.VideoCapture(DEFAULT_CAMERA_INDEX, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(DEFAULT_CAMERA_INDEX)

    if not cap.isOpened():
        print("[!] Error: Could not access camera.")
        return

    captured = 0
    print(f"[*] Target count: {count} images.")
    print("[*] Press SPACE to capture frame, or 'q' to quit.")

    try:
        while captured < count:
            ret, frame = cap.read()
            if not ret:
                print("[!] Failed to grab frame.")
                break

            frame_display = cv2.flip(frame, 1)
            h, w, _ = frame_display.shape

            cv2.putText(
                frame_display,
                f"Person: {clean_name} | Captured: {captured}/{count}",
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )
            cv2.putText(
                frame_display,
                "Press SPACE to capture, 'q' to finish",
                (20, h - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (200, 200, 200),
                1
            )

            cv2.imshow("Dataset Collection", frame_display)
            key = cv2.waitKey(1) & 0xFF

            if key == 32:  # Spacebar
                captured += 1
                file_path = person_dir / f"{clean_name}_{captured:02d}.jpg"
                cv2.imwrite(str(file_path), frame)
                print(f"[+] Saved image {captured}/{count}: {file_path.name}")
                time.sleep(0.2)
            elif key == ord('q') or key == 27:
                print("[*] Collection stopped by user.")
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()

    print(f"[+] Dataset collection complete for '{clean_name}'. Total saved: {captured}")


def launch_capture_gui():
    """
    Launches CustomTkinter GUI interface for image collection.
    Falls back to CLI capture if CustomTkinter is unavailable.
    """
    try:
        import customtkinter as ctk
        from PIL import Image, ImageTk

        ctk.set_appearance_mode("Dark")
        ctk.set_default_color_theme("blue")

        class CaptureApp(ctk.CTk):
            def __init__(self):
                super().__init__()
                self.title("Face Dataset Collector (Secured)")
                self.geometry("720x600")
                self.resizable(False, False)

                self.cap = None
                self.is_capturing = False
                self.capture_count = 0
                self.target_count = CAPTURE_TARGET_COUNT
                self.person_name = ""
                self.last_capture_time = 0.0

                self.setup_ui()

            def setup_ui(self):
                # Header
                header = ctk.CTkLabel(
                    self, 
                    text="Face Dataset Collection Tool", 
                    font=ctk.CTkFont(size=22, weight="bold")
                )
                header.pack(pady=(15, 5))

                sub = ctk.CTkLabel(
                    self, 
                    text="Capture high-quality face samples for embedding training (Secured)",
                    font=ctk.CTkFont(size=12)
                )
                sub.pack(pady=(0, 15))

                # Controls
                control_frame = ctk.CTkFrame(self)
                control_frame.pack(fill="x", padx=20, pady=5)

                name_lbl = ctk.CTkLabel(control_frame, text="Person Name:", font=ctk.CTkFont(size=14))
                name_lbl.pack(side="left", padx=(15, 5), pady=10)

                self.name_entry = ctk.CTkEntry(control_frame, width=200, placeholder_text="e.g. John")
                self.name_entry.pack(side="left", padx=5, pady=10)

                self.btn_start = ctk.CTkButton(
                    control_frame, 
                    text="Start Auto Capture", 
                    command=self.start_auto_capture
                )
                self.btn_start.pack(side="left", padx=10, pady=10)

                # Camera Display Canvas
                self.video_label = ctk.CTkLabel(self, text="Camera Feed Offline", width=640, height=360, fg_color="gray20")
                self.video_label.pack(pady=15)

                # Status Bar
                self.status_label = ctk.CTkLabel(
                    self, 
                    text="Ready. Enter name and start capture.", 
                    font=ctk.CTkFont(size=13)
                )
                self.status_label.pack(pady=5)

                self.start_camera()

            def start_camera(self):
                self.cap = cv2.VideoCapture(DEFAULT_CAMERA_INDEX, cv2.CAP_DSHOW)
                if not self.cap.isOpened():
                    self.cap = cv2.VideoCapture(DEFAULT_CAMERA_INDEX)
                self.update_feed()

            def update_feed(self):
                if self.cap and self.cap.isOpened():
                    ret, frame = self.cap.read()
                    if ret:
                        flipped_frame = cv2.flip(frame, 1)
                        now = time.time()

                        # Paced capture: require AUTO_CAPTURE_INTERVAL seconds between frames
                        if self.is_capturing and self.capture_count < self.target_count:
                            if now - self.last_capture_time >= AUTO_CAPTURE_INTERVAL:
                                self.last_capture_time = now
                                self.capture_count += 1
                                person_dir = (DATASET_DIR / self.person_name).resolve()
                                if person_dir.is_relative_to(DATASET_DIR.resolve()):
                                    person_dir.mkdir(parents=True, exist_ok=True)
                                    file_path = person_dir / f"{self.person_name}_{self.capture_count:02d}.jpg"
                                    cv2.imwrite(str(file_path), frame)
                                    self.status_label.configure(
                                        text=f"Captured {self.capture_count}/{self.target_count} for '{self.person_name}' (Turn face slightly)"
                                    )

                                if self.capture_count >= self.target_count:
                                    self.is_capturing = False
                                    self.btn_start.configure(state="normal")
                                    self.status_label.configure(
                                        text=f"Completed! {self.target_count} samples saved to dataset/{self.person_name}"
                                    )

                        img_rgb = cv2.cvtColor(flipped_frame, cv2.COLOR_BGR2RGB)
                        img_pil = Image.fromarray(img_rgb).resize((640, 360))
                        img_tk = ImageTk.PhotoImage(image=img_pil)
                        self.video_label.configure(image=img_tk, text="")
                        self.video_label.image = img_tk

                self.after(30, self.update_feed)

            def start_auto_capture(self):
                raw_name = self.name_entry.get()
                try:
                    clean_name = sanitize_person_name(raw_name)
                except ValueError as e:
                    self.status_label.configure(text=f"Error: {e}")
                    return

                self.person_name = clean_name
                self.capture_count = 0
                self.last_capture_time = 0.0
                self.is_capturing = True
                self.btn_start.configure(state="disabled")
                self.status_label.configure(text=f"Capturing samples for '{clean_name}'... Turn head slowly.")

            def on_closing(self):
                if self.cap and self.cap.isOpened():
                    self.cap.release()
                self.destroy()

        app = CaptureApp()
        app.protocol("WM_DELETE_WINDOW", app.on_closing)
        app.mainloop()

    except Exception as e:
        print(f"[*] CustomTkinter GUI launch note: {e}")
        print("[*] Falling back to CLI / OpenCV interactive mode...")
        while True:
            raw_name = input("Enter person name to capture: ").strip()
            try:
                clean_name = sanitize_person_name(raw_name)
                capture_dataset_cli(clean_name)
                break
            except ValueError as val_err:
                print(f"[!] Invalid name: {val_err}. Please try again.")


if __name__ == "__main__":
    launch_capture_gui()
