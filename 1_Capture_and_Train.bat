@echo off
title Face Recognizer - Capture and Train
cd /d "%~dp0"

echo ============================================================
echo          FACE RECOGNIZER - DATASET CAPTURE & TRAINING
echo ============================================================
echo.

:: Detect Python environment
set "PY_CMD=python"
if exist "venv\Scripts\python.exe" (
    set "PY_CMD=venv\Scripts\python.exe"
) else if exist ".venv\Scripts\python.exe" (
    set "PY_CMD=.venv\Scripts\python.exe"
)

:: Step 1: Dataset Capture Tool
echo [*] STEP 1/2: Launching Face Capture Tool...
echo     1. Type your name in the text box.
echo     2. Click "Start Auto Capture" and turn your head slowly.
echo     3. Close the window when finished.
echo.
%PY_CMD% app.py

echo.
echo ============================================================
echo [*] STEP 2/2: Training Neural Network Embeddings...
echo ============================================================
echo.
%PY_CMD% train.py

echo.
if %ERRORLEVEL% EQU 0 (
    echo ============================================================
    echo  [SUCCESS] Face model trained and updated successfully!
    echo  Now double-click "2_Start_Recognizer.bat" to start recognition.
    echo ============================================================
) else (
    echo  [!] An error occurred during training.
)

echo.
pause
