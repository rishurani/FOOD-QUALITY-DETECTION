#!/usr/bin/env bash
# LexBrief AI - Launcher Script for Linux / macOS
# Project ID: CSEAIML2026-P13

echo "=============================================================================="
echo "      LexBrief AI - AI-Based Court Case Summarization & Legal Research"
echo "                        Project ID: CSEAIML2026-P13"
echo "=============================================================================="
echo ""

if ! command -v python3 &> /dev/null; then
    echo "[ERROR] Python 3 is not installed or not in PATH."
    exit 1
fi

echo "[*] Checking Python dependencies..."
pip install -r requirements.txt > /dev/null 2>&1

echo "[*] Initializing evaluation benchmark..."
python3 evaluate_system.py

echo ""
echo "=============================================================================="
echo "[*] Launching server on http://127.0.0.1:5000"
echo "[*] Press CTRL+C to stop the server."
echo "=============================================================================="
echo ""

python3 app.py
