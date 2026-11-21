@echo off
title LexBrief AI - Court Case Summarization & Legal Research Assistant
color 0b

echo ==============================================================================
echo       LexBrief AI - AI-Based Court Case Summarization & Legal Research
echo                         Project ID: CSEAIML2026-P13
echo ==============================================================================
echo.

:: Check Python installation
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python is not installed or not added to PATH.
    echo Please install Python 3.10+ and try again.
    pause
    exit /b 1
)

echo [*] Checking dependencies...
pip install -r requirements.txt >nul 2>&1

echo [*] Pre-indexing landmark case repository & running quick benchmark...
python evaluate_system.py

echo.
echo ==============================================================================
echo [*] Starting LexBrief AI Server at http://127.0.0.1:5000
echo [*] Open your web browser and navigate to: http://127.0.0.1:5000
echo [*] Press CTRL+C to stop the application server.
echo ==============================================================================
echo.

python app.py
pause
