@echo off
title Face Recognizer - Real-Time Recognition
cd /d "%~dp0"

echo ============================================================
echo          FACE RECOGNIZER - REAL-TIME RECOGNITION
echo ============================================================
echo.

:: Detect Python environment
set "PY_CMD=python"
if exist "venv\Scripts\python.exe" (
    set "PY_CMD=venv\Scripts\python.exe"
) else if exist ".venv\Scripts\python.exe" (
    set "PY_CMD=.venv\Scripts\python.exe"
)

echo [*] Starting GPU-Accelerated Face Recognition Feed...
echo [*] Controls: Press 'q' or 'ESC' on the video window to quit.
echo.

%PY_CMD% use.py

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo [!] Process exited with code %ERRORLEVEL%.
    pause
)
