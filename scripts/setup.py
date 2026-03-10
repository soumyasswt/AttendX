"""
scripts/setup.py
================
One-time setup script: verifies dependencies and pre-downloads ML models.
Run this ONCE before starting the attendance system.

  python scripts/setup.py
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


REQUIRED_PACKAGES = [
    "opencv-python",
    "insightface",
    "onnxruntime",
    "mediapipe",
    "customtkinter",
    "Pillow",
    "pandas",
    "openpyxl",
    "loguru",
    "numpy",
    "scipy",
]

OPTIONAL_PACKAGES = [
    ("faiss-cpu", "FAISS vector search (recommended for large deployments)"),
]


def _check_import(module: str) -> bool:
    try:
        __import__(module)
        return True
    except ImportError:
        return False


def verify_dependencies() -> None:
    print("\n[1/3] Checking Python dependencies…\n")
    missing = []
    for pkg in REQUIRED_PACKAGES:
        mod = pkg.replace("-", "_").split(">=")[0]
        # Map package name to importable name
        import_map = {
            "opencv_python": "cv2",
            "Pillow": "PIL",
            "scikit_learn": "sklearn",
        }
        mod_name = import_map.get(mod, mod)
        ok = _check_import(mod_name)
        status = "✅" if ok else "❌"
        print(f"  {status}  {pkg}")
        if not ok:
            missing.append(pkg)

    for pkg, desc in OPTIONAL_PACKAGES:
        mod = pkg.replace("-", "_")
        ok = _check_import(mod)
        status = "✅" if ok else "⚠️ "
        print(f"  {status}  {pkg}  [{desc}]")

    if missing:
        print(f"\n[WARN] Missing packages: {missing}")
        ans = input("\nInstall missing packages now? [y/N] ").strip().lower()
        if ans == "y":
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", *missing]
            )
        else:
            print("[INFO] Please run: pip install -r requirements.txt")
    else:
        print("\n  All required packages present ✅")


def preload_insightface_models() -> None:
    print("\n[2/3] Pre-downloading InsightFace models…\n")
    try:
        from insightface.app import FaceAnalysis
        app = FaceAnalysis(
            name="buffalo_l",
            providers=["CPUExecutionProvider"],
        )
        app.prepare(ctx_id=-1, det_size=(320, 320))
        print("  ✅  InsightFace buffalo_l downloaded and ready")
    except Exception as exc:
        print(f"  ❌  InsightFace model download failed: {exc}")
        print("     The system will fall back to OpenCV's Haar cascade.")


def init_database() -> None:
    print("\n[3/3] Initializing database…\n")
    from config.settings import config
    from database.db import DB
    db = DB(config.database.db_path)
    print(f"  ✅  Database ready at: {config.database.db_path}")


def main() -> None:
    print("=" * 55)
    print("  Smart Attendance System – Setup")
    print("=" * 55)

    verify_dependencies()
    preload_insightface_models()
    init_database()

    print("\n" + "=" * 55)
    print("  Setup complete!")
    print("\n  Next steps:")
    print("  1. Enroll faces:")
    print('     python scripts/enroll.py --name "Your Name" --dept "Engineering"')
    print("  2. Start the system:")
    print("     python main.py")
    print("=" * 55 + "\n")


if __name__ == "__main__":
    main()
