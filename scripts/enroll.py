"""
scripts/enroll.py
=================
Interactive command-line tool to enroll a new face into the system.

Usage
-----
  python scripts/enroll.py --name "Jane Doe" --dept "Engineering"
  python scripts/enroll.py --name "Jane Doe" --dept "Engineering" --samples 5

The script:
  1. Opens the webcam
  2. Captures N face samples
  3. Computes embeddings for each sample
  4. Averages the embeddings (more robust than a single shot)
  5. Saves to the database
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np
from loguru import logger

from config.settings import config
from database.db import DB
from vision.detector import FaceDetector
from vision.aligner import FaceAligner
from recognition.embedder import ArcFaceEmbedder


def capture_embeddings(
    detector: FaceDetector,
    aligner: FaceAligner,
    embedder: ArcFaceEmbedder,
    n_samples: int = 5,
    camera_id: int = 0,
) -> list[np.ndarray]:
    """
    Open the webcam, display a live preview, and capture N good face samples.
    Returns a list of (512,) embedding arrays.
    """
    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera {camera_id}")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    embeddings = []
    print(f"\n[INFO] Look at the camera. Capturing {n_samples} samples…")
    print("       Press 'SPACE' to capture, 'Q' to quit.\n")

    while len(embeddings) < n_samples:
        ret, frame = cap.read()
        if not ret:
            continue

        detections = detector.detect(frame)
        display = frame.copy()

        for det in detections:
            x1, y1, x2, y2 = det.bbox
            cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 0), 2)

        remaining = n_samples - len(embeddings)
        cv2.putText(
            display,
            f"Press SPACE to capture ({remaining} remaining) | Q to quit",
            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2,
        )
        cv2.putText(
            display,
            f"Captured: {len(embeddings)}/{n_samples}",
            (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2,
        )

        cv2.imshow("Enrollment - Smart Attendance", display)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            print("[INFO] Enrollment cancelled.")
            break

        if key == ord(" "):
            if not detections:
                print("  [WARN] No face detected – try again")
                continue

            # Use the largest/best face
            best = max(detections, key=lambda d: d.area)
            aligned = aligner.align(frame, best.bbox, best.landmarks)
            if aligned is None:
                print("  [WARN] Alignment failed – try again")
                continue

            embedding = embedder.get_embedding(aligned)
            if embedding is None:
                print("  [WARN] Embedding failed – try again")
                continue

            embeddings.append(embedding)
            print(f"  ✓ Sample {len(embeddings)}/{n_samples} captured")

            # Green flash
            flash = frame.copy()
            cv2.rectangle(flash, (0, 0), (frame.shape[1], frame.shape[0]),
                          (0, 255, 0), 30)
            cv2.addWeighted(flash, 0.3, display, 0.7, 0, display)
            cv2.imshow("Enrollment - Smart Attendance", display)
            cv2.waitKey(300)

    cap.release()
    cv2.destroyAllWindows()
    return embeddings


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Enroll a new face into the Smart Attendance System"
    )
    parser.add_argument("--name",    required=True, help="Full name of the person")
    parser.add_argument("--dept",    default="General", help="Department")
    parser.add_argument("--email",   default=None, help="Email address (optional)")
    parser.add_argument("--role",    default="Employee", help="Job role")
    parser.add_argument("--samples", type=int, default=5,
                        help="Number of face samples to capture (default: 5)")
    parser.add_argument("--camera",  type=int, default=config.camera.device_id,
                        help="Camera device ID")
    args = parser.parse_args()

    print(f"\n{'='*55}")
    print(f"  Smart Attendance System – Face Enrollment")
    print(f"{'='*55}")
    print(f"  Name      : {args.name}")
    print(f"  Department: {args.dept}")
    print(f"  Samples   : {args.samples}")
    print(f"{'='*55}\n")

    # Initialize components
    print("[1/4] Initializing database…")
    db = Database(config)
    db.initialize()

    print("[2/4] Loading face detector…")
    detector = FaceDetector(config)
    detector.initialize()

    aligner = FaceAligner(config)

    print("[3/4] Loading embedding model…")
    embedder = ArcFaceEmbedder(config)
    embedder.initialize(face_analysis=detector.face_analysis)

    print("[4/4] Capturing face samples…\n")

    embeddings = capture_embeddings(
        detector, aligner, embedder,
        n_samples=args.samples,
        camera_id=args.camera,
    )

    if not embeddings:
        print("[ERROR] No samples captured. Exiting.")
        sys.exit(1)

    # Average embeddings (more robust representation)
    avg_embedding = np.mean(np.stack(embeddings, axis=0), axis=0)
    norm = np.linalg.norm(avg_embedding)
    if norm > 0:
        avg_embedding /= norm

    print(f"\n[INFO] Averaging {len(embeddings)} embeddings…")

    # Save to database
    usr_q = UserQueries(db)
    user_id = usr_q.create_user(
        name=args.name,
        embedding=avg_embedding,
        department=args.dept,
        email=args.email,
        role=args.role,
    )
    db.connection.commit()

    print(f"\n{'='*55}")
    print(f"  ✅  Enrollment successful!")
    print(f"  Name    : {args.name}")
    print(f"  User ID : {user_id}")
    print(f"  Dept    : {args.dept}")
    print(f"{'='*55}\n")


if __name__ == "__main__":
    main()
