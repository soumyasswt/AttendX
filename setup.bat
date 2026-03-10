@echo off
title AttendX Setup
echo ============================================
echo   AttendX - Setup Wizard (Windows)
echo ============================================
echo.

:: Check Python
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python is not installed or not in PATH.
    echo Download it from https://www.python.org/downloads/
    pause
    exit /b 1
)
echo [OK] Python found.

:: Create virtual environment
if not exist "venv\" (
    echo [*] Creating virtual environment...
    python -m venv venv
    if errorlevel 1 (
        echo [ERROR] Failed to create virtual environment.
        pause
        exit /b 1
    )
    echo [OK] Virtual environment created.
) else (
    echo [OK] Virtual environment already exists.
)

:: Activate venv
call venv\Scripts\activate.bat

:: Upgrade pip
echo [*] Upgrading pip...
python -m pip install --upgrade pip --quiet

:: Install core dependencies
echo [*] Installing core dependencies (Flask, OpenCV, etc.)...
pip install flask loguru opencv-python numpy gunicorn openpyxl pandas --quiet
if errorlevel 1 (
    echo [ERROR] Core dependency installation failed.
    pause
    exit /b 1
)
echo [OK] Core dependencies installed.

:: Optional: Face Recognition
echo.
set /p INSTALL_ML="Install face recognition packages? (insightface, onnxruntime ~400MB) [y/N]: "
if /i "%INSTALL_ML%"=="y" (
    echo [*] Installing face recognition packages (this may take a few minutes)...
    pip install insightface onnxruntime --quiet
    echo [*] Downloading InsightFace models...
    python scripts\setup.py
    echo [OK] Face recognition ready.
)

:: Create required directories
if not exist "data" mkdir data
if not exist "uploads" mkdir uploads
if not exist "logs" mkdir logs

:: Copy .env if not present
if not exist ".env" (
    if exist ".env.example" (
        copy ".env.example" ".env" >nul
        echo [OK] Created .env from .env.example
    )
)

echo.
echo ============================================
echo   Setup complete!
echo   Run the app with:  run.bat
echo   Or manually:       python main_web.py
echo ============================================
pause
