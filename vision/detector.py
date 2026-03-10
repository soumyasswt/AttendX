"""
vision/detector.py
==================
Face detection stage of the CV pipeline.

Primary backend : InsightFace (RetinaFace via ONNX Runtime – CPU)
Fallback backend: OpenCV DNN (res10_300x300_ssd) when InsightFace
                  is not installed or models are unavailable.

Each backend returns a unified list of FaceDetection objects so the
rest of the pipeline is backend-agnostic.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import cv2
import numpy as np
from loguru import logger

from config.settings import AppConfig


@dataclass
class FaceDetection:
    """
    Canonical result from the detection stage.

    Attributes
    ----------
    bbox      : (x1, y1, x2, y2) pixel coordinates
    confidence: detector confidence score [0, 1]
    landmarks : 5-point facial landmarks as (x,y) pairs, or None
    """

    bbox: Tuple[int, int, int, int]
    confidence: float
    landmarks: Optional[np.ndarray] = field(default=None, repr=False)

    # Convenience properties
    @property
    def x1(self) -> int: return self.bbox[0]
    @property
    def y1(self) -> int: return self.bbox[1]
    @property
    def x2(self) -> int: return self.bbox[2]
    @property
    def y2(self) -> int: return self.bbox[3]
    @property
    def width(self) -> int: return self.bbox[2] - self.bbox[0]
    @property
    def height(self) -> int: return self.bbox[3] - self.bbox[1]
    @property
    def center(self) -> Tuple[int, int]:
        return ((self.bbox[0] + self.bbox[2]) // 2,
                (self.bbox[1] + self.bbox[3]) // 2)
    @property
    def area(self) -> int: return self.width * self.height

    def to_xywh(self) -> Tuple[int, int, int, int]:
        """Convert to (x, y, w, h) format."""
        return (self.bbox[0], self.bbox[1], self.width, self.height)

    def crop(self, frame: np.ndarray, padding: float = 0.1) -> np.ndarray:
        """Return the face crop with optional padding fraction."""
        h, w = frame.shape[:2]
        pad_x = int(self.width * padding)
        pad_y = int(self.height * padding)
        x1 = max(0, self.bbox[0] - pad_x)
        y1 = max(0, self.bbox[1] - pad_y)
        x2 = min(w, self.bbox[2] + pad_x)
        y2 = min(h, self.bbox[3] + pad_y)
        return frame[y1:y2, x1:x2]


# ─────────────────────────────────────────────────────────────────────────────
# InsightFace backend
# ─────────────────────────────────────────────────────────────────────────────

class InsightFaceDetector:
    """
    Wraps insightface.app.FaceAnalysis (RetinaFace + ArcFace).

    The same FaceAnalysis object is reused for *detection only* here;
    the recognition module will call get_feat() separately so we can
    skip recognition on every detection cycle.
    """

    def __init__(self, cfg: AppConfig) -> None:
        self._cfg = cfg
        self._model = None
        self._initialized = False

    def initialize(self) -> bool:
        try:
            import insightface
            from insightface.app import FaceAnalysis

            logger.info("Loading InsightFace model pack '{}'…",
                        self._cfg.recognition.model_name)
            app = FaceAnalysis(
                name=self._cfg.recognition.model_name,
                providers=(
                    ["CUDAExecutionProvider", "CPUExecutionProvider"]
                    if self._cfg.recognition.use_gpu
                    else ["CPUExecutionProvider"]
                ),
            )
            # det_size must be multiples of 32; 320x320 is fast on CPU
            app.prepare(ctx_id=self._cfg.recognition.ctx_id, det_size=(320, 320))
            self._model = app
            self._initialized = True
            logger.success("InsightFace initialized (det_size 320×320, CPU)")
            return True
        except Exception as exc:
            logger.warning("InsightFace not available: {}; using OpenCV fallback", exc)
            return False

    def detect(
        self,
        frame: np.ndarray,
        min_confidence: float,
        min_face_px: int,
    ) -> List[FaceDetection]:
        if not self._initialized or self._model is None:
            return []

        faces = self._model.get(frame)
        results: List[FaceDetection] = []

        for f in faces:
            score = float(f.det_score) if hasattr(f, "det_score") else 1.0
            if score < min_confidence:
                continue
            x1, y1, x2, y2 = [int(v) for v in f.bbox]
            w, h = x2 - x1, y2 - y1
            if w < min_face_px or h < min_face_px:
                continue
            lmk = f.kps.astype(np.float32) if hasattr(f, "kps") and f.kps is not None else None
            results.append(
                FaceDetection(
                    bbox=(max(0, x1), max(0, y1), x2, y2),
                    confidence=score,
                    landmarks=lmk,
                )
            )
        return results

    @property
    def face_analysis(self):
        """Expose underlying FaceAnalysis for embedding extraction."""
        return self._model


# ─────────────────────────────────────────────────────────────────────────────
# OpenCV DNN fallback
# ─────────────────────────────────────────────────────────────────────────────

class OpenCVDNNDetector:
    """
    Lightweight fallback using OpenCV's built-in SSD face detector.

    Does NOT provide landmarks; the alignment stage will use a
    generic 5-point estimator when landmarks are None.
    """

    # Caffe model shipped with opencv-contrib
    PROTOTXT = "deploy.prototxt"
    CAFFEMODEL = "res10_300x300_ssd_iter_140000.caffemodel"

    def __init__(self, cfg: AppConfig) -> None:
        self._cfg = cfg
        self._net = None
        self._initialized = False

    def initialize(self) -> bool:
        try:
            # Try to load from models dir first
            proto = str(self._cfg.models_dir / self.PROTOTXT)
            model = str(self._cfg.models_dir / self.CAFFEMODEL)

            import os
            if not (os.path.exists(proto) and os.path.exists(model)):
                # Fall back to Haar cascade (always available in OpenCV)
                logger.info("SSD model files not found; using Haar cascade")
                self._net = cv2.CascadeClassifier(
                    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
                )
                self._use_haar = True
            else:
                self._net = cv2.dnn.readNetFromCaffe(proto, model)
                self._use_haar = False

            self._initialized = True
            logger.success("OpenCV detector initialized (haar={})", self._use_haar)
            return True
        except Exception as exc:
            logger.error("OpenCV detector failed: {}", exc)
            return False

    def detect(
        self,
        frame: np.ndarray,
        min_confidence: float,
        min_face_px: int,
    ) -> List[FaceDetection]:
        if not self._initialized or self._net is None:
            return []

        h, w = frame.shape[:2]

        if getattr(self, "_use_haar", False):
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            rects = self._net.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=5, minSize=(min_face_px, min_face_px)
            )
            results = []
            for (x, y, fw, fh) in rects:
                results.append(
                    FaceDetection(
                        bbox=(x, y, x + fw, y + fh),
                        confidence=0.9,  # Haar doesn't give confidence
                        landmarks=None,
                    )
                )
            return results

        # SSD DNN path
        blob = cv2.dnn.blobFromImage(
            cv2.resize(frame, (300, 300)), 1.0, (300, 300),
            (104.0, 177.0, 123.0)
        )
        self._net.setInput(blob)
        detections = self._net.forward()
        results = []

        for i in range(detections.shape[2]):
            confidence = float(detections[0, 0, i, 2])
            if confidence < min_confidence:
                continue
            x1 = int(detections[0, 0, i, 3] * w)
            y1 = int(detections[0, 0, i, 4] * h)
            x2 = int(detections[0, 0, i, 5] * w)
            y2 = int(detections[0, 0, i, 6] * h)
            fw, fh = x2 - x1, y2 - y1
            if fw < min_face_px or fh < min_face_px:
                continue
            results.append(
                FaceDetection(
                    bbox=(max(0, x1), max(0, y1), min(w, x2), min(h, y2)),
                    confidence=confidence,
                    landmarks=None,
                )
            )
        return results


# ─────────────────────────────────────────────────────────────────────────────
# Unified FaceDetector (selects best available backend)
# ─────────────────────────────────────────────────────────────────────────────

class FaceDetector:
    """
    Top-level detector that auto-selects the best available backend.

    Pipeline contract
    -----------------
    Input : BGR frame (np.ndarray, uint8)
    Output: List[FaceDetection]
    """

    def __init__(self, cfg: AppConfig) -> None:
        self._cfg = cfg
        self._backend: Optional[InsightFaceDetector | OpenCVDNNDetector] = None
        self._backend_name = "none"

        # Perf stats
        self._inference_times: List[float] = []

    def initialize(self) -> None:
        """Try InsightFace first, fall back to OpenCV."""
        insight = InsightFaceDetector(self._cfg)
        if insight.initialize():
            self._backend = insight
            self._backend_name = "insightface"
            return

        ocv = OpenCVDNNDetector(self._cfg)
        if ocv.initialize():
            self._backend = ocv
            self._backend_name = "opencv_dnn"
            return

        raise RuntimeError("No face detector backend could be initialized.")

    def detect(self, frame: np.ndarray) -> List[FaceDetection]:
        """
        Run face detection on a single frame.

        Returns an empty list if no faces found.
        """
        if self._backend is None:
            return []

        t0 = time.perf_counter()
        results = self._backend.detect(
            frame,
            min_confidence=self._cfg.vision.detection_confidence,
            min_face_px=self._cfg.vision.min_face_size,
        )
        # Cap to max_faces (highest confidence first)
        results.sort(key=lambda d: d.confidence, reverse=True)
        results = results[: self._cfg.vision.max_faces]

        elapsed = time.perf_counter() - t0
        self._inference_times.append(elapsed)
        if len(self._inference_times) > 100:
            self._inference_times.pop(0)

        return results

    @property
    def backend_name(self) -> str:
        return self._backend_name

    @property
    def face_analysis(self):
        """Return the InsightFace FaceAnalysis object, or None."""
        if isinstance(self._backend, InsightFaceDetector):
            return self._backend.face_analysis
        return None

    @property
    def avg_inference_ms(self) -> float:
        if not self._inference_times:
            return 0.0
        return 1000.0 * sum(self._inference_times) / len(self._inference_times)
