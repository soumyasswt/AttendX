"""
vision/aligner.py
=================
Face alignment stage.

Transforms a raw face crop into a standardized 112×112 image
aligned to the ArcFace canonical landmark positions.

When InsightFace landmarks are available, we use an affine warp
(similarity transform) that aligns eyes, nose, and mouth corners
to fixed template positions.

When no landmarks are available (e.g., OpenCV DNN backend), we
fall back to a simple resize + mean-normalization crop.
"""

from __future__ import annotations

from typing import Optional, Tuple

import cv2
import numpy as np
from loguru import logger

from config.settings import AppConfig

# ─────────────────────────────────────────────────────────────────────────────
# ArcFace canonical landmark template (112×112 coordinate space)
# Source: InsightFace / ArcFace paper
# ─────────────────────────────────────────────────────────────────────────────

ARCFACE_TEMPLATE = np.array(
    [
        [38.2946, 51.6963],   # left eye
        [73.5318, 51.5014],   # right eye
        [56.0252, 71.7366],   # nose tip
        [41.5493, 92.3655],   # left mouth corner
        [70.7299, 92.2041],   # right mouth corner
    ],
    dtype=np.float32,
)


def _estimate_norm(lmk: np.ndarray, image_size: int = 112) -> np.ndarray:
    """
    Estimate a 2×3 affine transformation matrix that maps *lmk*
    (5 × 2 source landmarks) to the ArcFace template.

    Uses `cv2.estimateAffinePartial2D` (similarity transform:
    rotation + uniform scale + translation) to preserve face geometry.
    """
    assert lmk.shape == (5, 2), f"Expected (5,2) landmarks, got {lmk.shape}"
    template = ARCFACE_TEMPLATE.copy()
    if image_size != 112:
        scale = image_size / 112.0
        template *= scale

    M, _ = cv2.estimateAffinePartial2D(lmk, template, method=cv2.RANSAC)
    return M  # 2×3


class FaceAligner:
    """
    Aligns and normalises face crops to the canonical 112×112 space.

    Pipeline contract
    -----------------
    Input : BGR frame (full resolution) + Track (has bbox + optional landmarks)
    Output: aligned face image (np.ndarray, uint8, BGR, 112×112)
    """

    def __init__(self, cfg: AppConfig) -> None:
        self._cfg = cfg
        self._out_size = cfg.vision.alignment_size  # (112, 112)

    def align(
        self,
        frame: np.ndarray,
        bbox: Tuple[int, int, int, int],
        landmarks: Optional[np.ndarray] = None,
    ) -> Optional[np.ndarray]:
        """
        Return the aligned face image or None if the crop is invalid.

        Parameters
        ----------
        frame     : full BGR frame
        bbox      : (x1,y1,x2,y2) face bounding box
        landmarks : (5,2) array of facial landmarks, or None
        """
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = bbox
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)

        if x2 <= x1 or y2 <= y1:
            return None

        if landmarks is not None and landmarks.shape == (5, 2):
            return self._align_with_landmarks(frame, landmarks)
        else:
            return self._align_simple(frame, (x1, y1, x2, y2))

    def _align_with_landmarks(
        self, frame: np.ndarray, landmarks: np.ndarray
    ) -> Optional[np.ndarray]:
        """Affine-warp using 5-point similarity transform."""
        try:
            M = _estimate_norm(landmarks.astype(np.float32))
            if M is None:
                return None
            aligned = cv2.warpAffine(
                frame, M, self._out_size,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=(0, 0, 0),
            )
            return aligned
        except Exception as exc:
            logger.debug("Landmark alignment failed: {}", exc)
            return None

    def _align_simple(
        self, frame: np.ndarray, bbox: Tuple[int, int, int, int]
    ) -> np.ndarray:
        """
        Fallback: crop + resize + pad to square.

        Used when landmarks are unavailable.
        """
        x1, y1, x2, y2 = bbox
        # Make the crop square by extending shorter axis
        fw, fh = x2 - x1, y2 - y1
        if fw > fh:
            delta = fw - fh
            y1 = max(0, y1 - delta // 2)
            y2 = min(frame.shape[0], y2 + delta // 2)
        else:
            delta = fh - fw
            x1 = max(0, x1 - delta // 2)
            x2 = min(frame.shape[1], x2 + delta // 2)

        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return np.zeros((*self._out_size, 3), dtype=np.uint8)
        return cv2.resize(crop, self._out_size, interpolation=cv2.INTER_LINEAR)

    @staticmethod
    def preprocess_for_model(aligned: np.ndarray) -> np.ndarray:
        """
        Normalize aligned face for embedding model input.

        Returns float32 array in shape (1, 3, 112, 112) following
        the NCHW format expected by ONNX ArcFace models.
        Pixel values normalized to [-1, 1].
        """
        img = aligned.astype(np.float32)
        img = (img - 127.5) / 127.5                # → [-1, 1]
        img = img[:, :, ::-1]                       # BGR → RGB
        img = img.transpose(2, 0, 1)               # HWC → CHW
        img = np.expand_dims(img, axis=0)           # → (1, 3, 112, 112)
        return img
