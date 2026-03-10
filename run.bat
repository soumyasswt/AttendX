@echo off
title AttendX
echo Starting AttendX...

:: Use venv if it exists, otherwise system Python
if exist "venv\Scripts\activate.bat" (
    call venv\Scripts\activate.bat
)

:: Create directories if missing
if not exist "data" mkdir data
if not exist "uploads" mkdir uploads
if not exist "logs" mkdir logs

python main_web.py
pause
