@echo off
title AttendX Builder

echo [*] Installing build tools...
pip install pyinstaller PyQt6 PyQt6-WebEngine --quiet

echo [*] Building AttendX Executable...
:: Build as a standalone folder (easier for heavy ML models)
pyinstaller --name "AttendX" ^
  --windowed ^
  --noconfirm ^
  --icon=NONE ^
  --add-data "web_app/templates;web_app/templates" ^
  --add-data "web_app/static;web_app/static" ^
  --add-data "data;data" ^
  --hidden-import PyQt6 ^
  --hidden-import PyQt6.QtWebEngineWidgets ^
  --hidden-import PyQt6.QtWebEngineCore ^
  main_web.py

echo.
echo ==============================================
echo   Build complete! 
echo   Your Desktop App is in the 'dist\AttendX' folder.
echo   You can zip this folder and share it!
echo ==============================================
pause
