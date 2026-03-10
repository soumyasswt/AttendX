#!/usr/bin/env bash
set -e

echo "============================================"
echo "  AttendX - Setup Wizard (Linux/macOS)"
echo "============================================"
echo ""

# Check Python
if ! command -v python3 &>/dev/null; then
    echo "[ERROR] python3 not found. Install Python 3.10+ first."
    exit 1
fi
echo "[OK] Python found: $(python3 --version)"

# Create virtual environment
if [ ! -d "venv" ]; then
    echo "[*] Creating virtual environment..."
    python3 -m venv venv
    echo "[OK] Virtual environment created."
else
    echo "[OK] Virtual environment already exists."
fi

# Activate
source venv/bin/activate

# Upgrade pip
echo "[*] Upgrading pip..."
pip install --upgrade pip --quiet

# Core deps
echo "[*] Installing core dependencies..."
pip install flask loguru opencv-python numpy gunicorn openpyxl pandas --quiet
echo "[OK] Core dependencies installed."

# Optional ML
echo ""
read -rp "Install face recognition packages? (insightface, onnxruntime ~400MB) [y/N]: " INSTALL_ML
if [[ "$INSTALL_ML" =~ ^[Yy]$ ]]; then
    echo "[*] Installing face recognition packages..."
    pip install insightface onnxruntime --quiet
    echo "[*] Downloading InsightFace models..."
    python scripts/setup.py
    echo "[OK] Face recognition ready."
fi

# Create directories
mkdir -p data uploads logs

# .env
if [ ! -f ".env" ] && [ -f ".env.example" ]; then
    cp .env.example .env
    echo "[OK] Created .env from .env.example"
fi

echo ""
echo "============================================"
echo "  Setup complete!"
echo "  Run the app:  ./run.sh"
echo "  Or manually:  python main_web.py"
echo "============================================"
