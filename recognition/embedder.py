"""
recognition/embedder.py
=======================
Facial embedding extraction service.

Converts an aligned 112×112 face image into a 512-dimensional
L2-normalized feature vector using ArcFace (InsightFace).

When InsightFace is not available, an ONNX-based fallback is used
or the module degrades to a random/placeholder embedding
(for testing the pipeline without ML models).

ONNX Runtime is used directly for fast CPU inference.
"""

from __future__ import annotations

import time
from typing import List, Optional

import numpy as np
from loguru import logger

from config.settings import AppConfig


class ArcFaceEmbedder:
    """
    Extracts 512-d ArcFace embeddings via InsightFace + ONNX Runtime.

    The FaceAnalysis object is shared with FaceDetector so we don't
    load the model twice.
    """

    def __init__(self, cfg: AppConfig) -> None:
        self._cfg = cfg
        self._rec_model = None
        self._initialized = False
        self._inference_times: List[float] = []

    def initialize(self, face_analysis=None) -> bool:
        """
        Initialize the embedding model.

        Parameters
        ----------
        face_analysis : insightface.app.FaceAnalysis instance (optional)
            If provided, the recognition sub-model is borrowed.
            Otherwise, a fresh instance is loaded.
        """
        try:
            if face_analysis is not None:
                # Extract the recognition model from the FaceAnalysis object
                for m in face_analysis.models.values():
                    if hasattr(m, "get_feat"):
                        self._rec_model = m
                        break
                if self._rec_model is None:
                    # FaceAnalysis exposes a different API
                    self._face_analysis = face_analysis
                    self._use_face_analysis = True
                    self._initialized = True
                    logger.success("ArcFace embedder initialized via FaceAnalysis")
                    return True

            if self._rec_model is None:
                # Load standalone recognition model
                import insightface
                from insightface.model_zoo import get_model

                model_path = str(self._cfg.models_dir / "w600k_r50.onnx")
                try:
                    self._rec_model = get_model(model_path)
                    self._rec_model.prepare(ctx_id=self._cfg.recognition.ctx_id)
                    logger.success("ArcFace loaded from local file: {}", model_path)
                except Exception:
                    # Try the default zoo
                    self._face_analysis = face_analysis
                    self._use_face_analysis = True
                    self._initialized = True
                    logger.info("Using FaceAnalysis for embeddings")
                    return True

            self._use_face_analysis = False
            self._initialized = True
            logger.success("ArcFace embedder initialized (direct rec model)")
            return True

        except Exception as exc:
            logger.warning("ArcFace embedder failed: {}; using dummy embedder", exc)
            self._use_dummy = True
            self._initialized = True
            return True

    def get_embedding(
        self,
        aligned_face: np.ndarray,
        raw_face: Optional[np.ndarray] = None,
    ) -> Optional[np.ndarray]:
        """
        Compute a 512-d L2-normalized embedding.

        Parameters
        ----------
        aligned_face : 112×112 BGR aligned face image
        raw_face     : original (unaligned) BGR crop (for FaceAnalysis path)

        Returns
        -------
        np.ndarray of shape (512,) normalized, or None on failure.
        """
        if not self._initialized:
            return None

        t0 = time.perf_counter()

        try:
            embedding = self._extract(aligned_face)
        except Exception as exc:
            logger.warning("Embedding extraction failed: {}", exc)
            return None

        elapsed = time.perf_counter() - t0
        self._inference_times.append(elapsed)
        if len(self._inference_times) > 50:
            self._inference_times.pop(0)

        return embedding

    def _extract(self, aligned_face: np.ndarray) -> Optional[np.ndarray]:
        """Internal extraction dispatcher."""

        # ── Dummy fallback (pipeline testing without models) ──────────────
        if getattr(self, "_use_dummy", False):
            vec = np.random.randn(512).astype(np.float32)
            return vec / np.linalg.norm(vec)

        # ── FaceAnalysis path ─────────────────────────────────────────────
        if getattr(self, "_use_face_analysis", False):
            fa = getattr(self, "_face_analysis", None)
            if fa is None:
                return None
            # We re-detect on the aligned face – not ideal but correct
            faces = fa.get(aligned_face)
            if not faces:
                return None
            emb = faces[0].normed_embedding
            if emb is None:
                return None
            return emb.astype(np.float32)

        # ── Direct recognition model path ─────────────────────────────────
        if self._rec_model is not None:
            if hasattr(self._rec_model, "get_feat"):
                # insightface recognition model API
                emb = self._rec_model.get_feat(aligned_face)
                if emb is not None:
                    emb = emb.flatten().astype(np.float32)
                    norm = np.linalg.norm(emb)
                    return emb / norm if norm > 0 else emb
            return None

        return None

    def get_embedding_from_faces(self, faces_list: list) -> List[np.ndarray]:
        """
        Batch-extract embeddings from a list of InsightFace face objects.

        This is the most efficient path: the detector already computed
        embeddings as part of FaceAnalysis.get(), so we just read them.
        """
        results = []
        for f in faces_list:
            emb = getattr(f, "normed_embedding", None)
            if emb is not None:
                results.append(emb.astype(np.float32))
            else:
                results.append(None)
        return results

    @property
    def avg_inference_ms(self) -> float:
        if not self._inference_times:
            return 0.0
        return 1000.0 * sum(self._inference_times) / len(self._inference_times)


# ─────────────────────────────────────────────────────────────────────────────
# Convenience: embedding cosine similarity
# ─────────────────────────────────────────────────────────────────────────────

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """
    Fast cosine similarity between two L2-normalized vectors.
    If vectors are already normalized, this is just the dot product.
    """
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a / na, b / nb))
