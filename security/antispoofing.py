"""
security/antispoofing.py
========================
Liveness and anti-spoofing detection layer.

Defends against:
  1. Printed photo attacks  → LBP texture variance (screens/prints are smooth)
  2. Screen replay attacks  → Fourier frequency analysis
  3. Static image attacks   → Eye blink detection via Eye Aspect Ratio (EAR)

Strategy
--------
Each check produces a score ∈ [0, 1].  The final liveness_score is
a weighted combination.  The system requires the score to exceed
`security.liveness_threshold` *and* at least one blink to be
detected over `min_liveness_frames` consecutive frames.

This approach is heuristic-based (no external ML model required),
making it easy to deploy without model downloads.  A plug-in
interface (`AntiSpoofingPlugin`) is provided so a dedicated
MiniFASNet ONNX model can be added later.
"""

from __future__ import annotations

import math
import threading
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, List, Optional, Tuple

import cv2
import numpy as np
from loguru import logger

from config.settings import AppConfig


# ─────────────────────────────────────────────────────────────────────────────
# Result type
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class LivenessResult:
    """Per-face liveness assessment."""
    score: float          # combined [0, 1]
    passed: bool          # True if score ≥ threshold
    texture_score: float  # LBP variance sub-score
    freq_score: float     # Frequency domain sub-score
    ear_score: float      # Eye Aspect Ratio sub-score
    blink_detected: bool
    reason: str = ""      # Human-readable verdict

    def __str__(self) -> str:
        status = "LIVE" if self.passed else "SPOOF"
        return f"[{status}] score={self.score:.3f} ({self.reason})"


# ─────────────────────────────────────────────────────────────────────────────
# Eye Aspect Ratio (EAR) blink detector
# ─────────────────────────────────────────────────────────────────────────────

def _eye_aspect_ratio(eye_pts: np.ndarray) -> float:
    """
    EAR = (|p2-p6| + |p3-p5|) / (2 * |p1-p4|)
    Eye points ordered: left-corner, top-left, top-right, right-corner,
                        bottom-right, bottom-left
    """
    if eye_pts.shape[0] < 6:
        return 0.3  # default open-eye value
    A = np.linalg.norm(eye_pts[1] - eye_pts[5])
    B = np.linalg.norm(eye_pts[2] - eye_pts[4])
    C = np.linalg.norm(eye_pts[0] - eye_pts[3])
    return (A + B) / (2.0 * C + 1e-6)


class BlinkDetector:
    """
    Tracks EAR history per track ID and detects blinks.
    A blink = EAR drops below threshold then rises back above.
    """

    def __init__(self, ear_threshold: float = 0.22, history_len: int = 30) -> None:
        self._thresh = ear_threshold
        self._history: dict[int, Deque[float]] = {}
        self._blink_counts: dict[int, int] = {}
        self._below: dict[int, bool] = {}

    def update(
        self,
        track_id: int,
        left_eye: Optional[np.ndarray],
        right_eye: Optional[np.ndarray],
    ) -> Tuple[float, bool]:
        """
        Update EAR for a track and return (current_ear, blink_detected_this_frame).
        """
        if track_id not in self._history:
            self._history[track_id] = deque(maxlen=30)
            self._blink_counts[track_id] = 0
            self._below[track_id] = False

        if left_eye is not None and right_eye is not None:
            ear = (_eye_aspect_ratio(left_eye) + _eye_aspect_ratio(right_eye)) / 2.0
        elif left_eye is not None:
            ear = _eye_aspect_ratio(left_eye)
        elif right_eye is not None:
            ear = _eye_aspect_ratio(right_eye)
        else:
            ear = 0.3  # assume open

        self._history[track_id].append(ear)

        # Detect blink transition
        blink_now = False
        if ear < self._thresh:
            self._below[track_id] = True
        elif self._below[track_id]:
            # Just re-opened: a blink completed
            self._below[track_id] = False
            self._blink_counts[track_id] += 1
            blink_now = True

        return ear, blink_now

    def get_blink_count(self, track_id: int) -> int:
        return self._blink_counts.get(track_id, 0)

    def reset(self, track_id: int) -> None:
        self._history.pop(track_id, None)
        self._blink_counts.pop(track_id, None)
        self._below.pop(track_id, None)


# ─────────────────────────────────────────────────────────────────────────────
# Texture analysis (LBP variance)
# ─────────────────────────────────────────────────────────────────────────────

def _lbp_variance(gray: np.ndarray) -> float:
    """
    Compute Local Binary Pattern variance of a grayscale image.

    Printed photos and screen images have lower high-frequency texture
    energy than real faces captured by a webcam.

    Returns a scalar ∈ [0, ∞); typical real face ≈ 60–150,
    printed photo ≈ 10–40, screen ≈ 5–25.
    """
    if gray.ndim == 3:
        gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (64, 64))

    # Compute gradient magnitude as texture proxy (faster than full LBP)
    grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    mag = np.sqrt(grad_x ** 2 + grad_y ** 2)
    return float(np.var(mag))


def _frequency_score(gray: np.ndarray) -> float:
    """
    Fourier-domain high-frequency energy ratio.

    Real faces have more high-frequency energy (pores, hair)
    than flat printed/screen images.
    Returns a score ∈ [0, 1]; higher = more likely real.
    """
    if gray.ndim == 3:
        gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (64, 64)).astype(np.float32)

    f = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)
    magnitude = np.abs(fshift)

    h, w = gray.shape
    cy, cx = h // 2, w // 2
    radius = min(h, w) // 4  # inner low-freq circle

    Y, X = np.ogrid[:h, :w]
    mask_low = (X - cx) ** 2 + (Y - cy) ** 2 <= radius ** 2
    low_energy = magnitude[mask_low].sum()
    total_energy = magnitude.sum() + 1e-6
    high_ratio = 1.0 - (low_energy / total_energy)
    # Normalize to [0, 1] – empirically calibrated
    return min(1.0, high_ratio * 5.0)


# ─────────────────────────────────────────────────────────────────────────────
# Mediapipe face mesh helper (extracts eye landmarks)
# ─────────────────────────────────────────────────────────────────────────────

class MeshLandmarkExtractor:
    """
    Uses MediaPipe Face Mesh to extract 468 face landmarks,
    specifically the eye sub-regions needed for EAR.
    """

    # MediaPipe face mesh landmark indices for eyes
    LEFT_EYE_INDICES  = [362, 385, 387, 263, 373, 380]
    RIGHT_EYE_INDICES = [33,  160, 158,  133, 153, 144]

    def __init__(self) -> None:
        self._mesh = None
        self._available = False
        self._init()

    def _init(self) -> None:
        try:
            import mediapipe as mp
            self._mp_face_mesh = mp.solutions.face_mesh
            self._mesh = mp.solutions.face_mesh.FaceMesh(
                static_image_mode=False,
                max_num_faces=5,
                refine_landmarks=False,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            )
            self._available = True
            logger.info("MediaPipe Face Mesh initialized for blink detection")
        except Exception as exc:
            logger.warning("MediaPipe not available ({}); blink detection disabled", exc)

    def extract_eyes(
        self, face_crop: np.ndarray
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Returns (left_eye_pts, right_eye_pts) in pixel coords,
        each shape (6, 2), or (None, None) on failure.
        """
        if not self._available or self._mesh is None:
            return None, None

        try:
            rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
            results = self._mesh.process(rgb)
            if not results.multi_face_landmarks:
                return None, None

            lmks = results.multi_face_landmarks[0]
            h, w = face_crop.shape[:2]

            def _pts(indices):
                return np.array(
                    [[lmks.landmark[i].x * w, lmks.landmark[i].y * h] for i in indices],
                    dtype=np.float32,
                )

            return _pts(self.LEFT_EYE_INDICES), _pts(self.RIGHT_EYE_INDICES)
        except Exception:
            return None, None

    def close(self) -> None:
        if self._mesh:
            self._mesh.close()


# ─────────────────────────────────────────────────────────────────────────────
# Main LivenessDetector
# ─────────────────────────────────────────────────────────────────────────────

class LivenessDetector:
    """
    Combines texture, frequency, and blink cues into a liveness score.

    Usage
    -----
    detector = LivenessDetector(cfg)
    result   = detector.check(track_id=1, face_crop=img)
    if result.passed:
        # allow recognition
    """

    # Score weights
    W_TEXTURE = 0.35
    W_FREQ    = 0.25
    W_BLINK   = 0.40

    def __init__(self, cfg: AppConfig) -> None:
        self._cfg = cfg
        self._scfg = cfg.security
        self._mesh = MeshLandmarkExtractor()
        self._blink_detector = BlinkDetector(ear_threshold=self._scfg.ear_blink_threshold)
        self._frame_counts: dict[int, int] = {}
        self._lock = threading.Lock()

    def check(
        self,
        track_id: int,
        face_crop: np.ndarray,
    ) -> LivenessResult:
        """
        Run liveness analysis on a face crop.

        Parameters
        ----------
        track_id  : unique integer track identifier
        face_crop : BGR face crop (any reasonable size)
        """
        if not self._scfg.enabled:
            return LivenessResult(
                score=1.0, passed=True,
                texture_score=1.0, freq_score=1.0, ear_score=0.3,
                blink_detected=True, reason="disabled",
            )

        with self._lock:
            self._frame_counts[track_id] = self._frame_counts.get(track_id, 0) + 1

        gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)

        # ── Texture score ─────────────────────────────────────────────────
        lbp_var = _lbp_variance(gray)
        texture_score = min(1.0, lbp_var / self._scfg.texture_lbp_threshold)

        # ── Frequency score ───────────────────────────────────────────────
        freq_score = _frequency_score(gray)

        # ── Blink / EAR score ─────────────────────────────────────────────
        left_eye, right_eye = self._mesh.extract_eyes(face_crop)
        ear, blink_now = self._blink_detector.update(track_id, left_eye, right_eye)
        blink_count = self._blink_detector.get_blink_count(track_id)

        # EAR score: reward being an eye that varies
        ear_score = min(1.0, ear / 0.35)  # normalized; open eye ≈ 0.3
        blink_detected = blink_count >= 1

        # ── Weighted combination ─────────────────────────────────────────
        score = (
            self.W_TEXTURE * texture_score
            + self.W_FREQ * freq_score
            + self.W_BLINK * (1.0 if blink_detected else ear_score * 0.5)
        )
        score = float(np.clip(score, 0.0, 1.0))

        # Must have enough frames before making a final decision
        frame_count = self._frame_counts.get(track_id, 0)
        if frame_count < self._scfg.min_liveness_frames:
            # Too early; give benefit of the doubt
            passed = False
            reason = f"accumulating ({frame_count}/{self._scfg.min_liveness_frames} frames)"
        else:
            if self._scfg.blink_required and not blink_detected:
                passed = False
                reason = "no blink detected"
            elif score >= self._scfg.liveness_threshold:
                passed = True
                reason = "live"
            else:
                passed = False
                reason = f"low score ({score:.2f} < {self._scfg.liveness_threshold})"

        return LivenessResult(
            score=score,
            passed=passed,
            texture_score=texture_score,
            freq_score=freq_score,
            ear_score=ear_score,
            blink_detected=blink_detected,
            reason=reason,
        )

    def reset_track(self, track_id: int) -> None:
        """Clean up state for a lost track."""
        with self._lock:
            self._frame_counts.pop(track_id, None)
        self._blink_detector.reset(track_id)

    def close(self) -> None:
        self._mesh.close()
