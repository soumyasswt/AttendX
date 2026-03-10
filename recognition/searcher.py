"""
recognition/searcher.py
=======================
Vector similarity search for face recognition.

Primary engine : FAISS (Facebook AI Similarity Search)
Fallback engine: NumPy brute-force dot-product search

Both engines expose the same `search()` interface, so the rest of
the application is transparent to which is used.

The index is rebuilt whenever a new face is enrolled.  For a
system with up to a few thousand enrolled users, FAISS FlatIP
(exact inner-product search on L2-normalized vectors = cosine sim)
gives sub-millisecond search latency even on CPU.

Thread safety
-------------
A threading.RWLock pattern is emulated with a regular lock so
concurrent reads are serialised.  This is fine since search is
fast; a proper RW lock can be added if needed.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
from loguru import logger

from config.settings import AppConfig
from recognition.embedder import cosine_similarity


# ─────────────────────────────────────────────────────────────────────────────
# Search result
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class SearchResult:
    """Outcome of a nearest-neighbour search."""
    user_id: int
    name: str
    department: str
    similarity: float          # cosine similarity [0, 1]

    @property
    def is_match(self) -> bool:
        return True             # caller decides threshold

    def __str__(self) -> str:
        return f"{self.name} ({self.similarity:.3f})"


# ─────────────────────────────────────────────────────────────────────────────
# FAISS engine
# ─────────────────────────────────────────────────────────────────────────────

class FAISSEngine:
    """Exact inner-product (cosine) search using FAISS FlatIP."""

    def __init__(self, dim: int) -> None:
        self._dim = dim
        self._index = None
        self._available = False
        self._try_import()

    def _try_import(self) -> None:
        try:
            import faiss
            self._faiss = faiss
            self._index = faiss.IndexFlatIP(self._dim)
            self._available = True
            logger.info("FAISS backend available (IndexFlatIP, dim={})", self._dim)
        except ImportError:
            logger.info("faiss-cpu not installed; using NumPy fallback")

    def build(self, embeddings: np.ndarray) -> None:
        """(Re)build the index from an (N, dim) float32 array."""
        assert embeddings.shape[1] == self._dim
        self._index.reset()
        # L2-normalize (cosine sim = IP on unit vectors)
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)
        normalized = (embeddings / norms).astype(np.float32)
        self._index.add(normalized)

    def search(self, query: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns (similarities, indices) arrays of length k.
        similarities ∈ [-1, 1]; clipped to [0, 1] for cosine.
        """
        norm = np.linalg.norm(query)
        q = (query / norm).reshape(1, -1).astype(np.float32) if norm > 0 else query.reshape(1, -1)
        sims, idxs = self._index.search(q, k)
        return sims[0], idxs[0]

    @property
    def size(self) -> int:
        return self._index.ntotal if self._index else 0

    @property
    def available(self) -> bool:
        return self._available


class NumpyEngine:
    """Brute-force cosine similarity using NumPy."""

    def __init__(self, dim: int) -> None:
        self._dim = dim
        self._embeddings: Optional[np.ndarray] = None  # (N, dim)

    def build(self, embeddings: np.ndarray) -> None:
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)
        self._embeddings = (embeddings / norms).astype(np.float32)

    def search(self, query: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        if self._embeddings is None or len(self._embeddings) == 0:
            return np.array([]), np.array([], dtype=np.int64)
        norm = np.linalg.norm(query)
        q = (query / norm).astype(np.float32) if norm > 0 else query.astype(np.float32)
        sims = self._embeddings @ q                     # (N,)
        k = min(k, len(sims))
        idxs = np.argsort(sims)[::-1][:k]
        return sims[idxs], idxs

    @property
    def size(self) -> int:
        return len(self._embeddings) if self._embeddings is not None else 0


# ─────────────────────────────────────────────────────────────────────────────
# High-level EmbeddingSearcher
# ─────────────────────────────────────────────────────────────────────────────

class EmbeddingSearcher:
    """
    Manages the face embedding index and provides named search.

    The index maps integer slot positions → (user_id, name, dept).
    When faces are added/removed the index is fully rebuilt.
    """

    def __init__(self, cfg: AppConfig) -> None:
        self._cfg = cfg
        self._dim = cfg.recognition.embedding_dim
        self._threshold = cfg.recognition.similarity_threshold
        self._lock = threading.Lock()

        # Slot arrays (parallel indexed)
        self._user_ids: List[int] = []
        self._names: List[str] = []
        self._departments: List[str] = []
        self._raw_embeddings: List[np.ndarray] = []

        # Search engine (FAISS preferred)
        faiss_eng = FAISSEngine(self._dim)
        self._engine = faiss_eng if faiss_eng.available else NumpyEngine(self._dim)
        self._index_built = False

    # ------------------------------------------------------------------ #
    # Index management                                                     #
    # ------------------------------------------------------------------ #

    def add_face(
        self,
        user_id: int,
        name: str,
        department: str,
        embedding: np.ndarray,
    ) -> None:
        """Add a new face embedding to the index."""
        with self._lock:
            self._user_ids.append(user_id)
            self._names.append(name)
            self._departments.append(department)
            self._raw_embeddings.append(embedding.astype(np.float32))
            self._rebuild_index()
            logger.debug("Enrolled: {} (user_id={}) – index size={}", name, user_id, self.size)

    def add_faces_bulk(
        self, records: List[Dict]
    ) -> None:
        """
        Bulk-load from the database.

        Each record: {"user_id": int, "name": str, "department": str,
                      "embedding": np.ndarray}
        """
        with self._lock:
            self._user_ids.clear()
            self._names.clear()
            self._departments.clear()
            self._raw_embeddings.clear()

            for r in records:
                self._user_ids.append(r["user_id"])
                self._names.append(r["name"])
                self._departments.append(r.get("department", ""))
                self._raw_embeddings.append(r["embedding"].astype(np.float32))

            if self._raw_embeddings:
                self._rebuild_index()
            logger.info("Bulk loaded {} face embeddings into search index", len(records))

    def remove_face(self, user_id: int) -> bool:
        """Remove all embeddings for a user and rebuild the index."""
        with self._lock:
            indices = [i for i, uid in enumerate(self._user_ids) if uid == user_id]
            if not indices:
                return False
            for i in sorted(indices, reverse=True):
                del self._user_ids[i]
                del self._names[i]
                del self._departments[i]
                del self._raw_embeddings[i]
            self._rebuild_index()
            return True

    def _rebuild_index(self) -> None:
        """Internal: rebuild the search engine from current lists."""
        if not self._raw_embeddings:
            self._index_built = False
            return
        mat = np.stack(self._raw_embeddings, axis=0)    # (N, 512)
        self._engine.build(mat)
        self._index_built = True

    # ------------------------------------------------------------------ #
    # Search                                                               #
    # ------------------------------------------------------------------ #

    def search(
        self, query_embedding: np.ndarray, top_k: int = 1
    ) -> Optional[SearchResult]:
        """
        Find the closest face in the index.

        Returns SearchResult if similarity ≥ threshold, else None.
        """
        with self._lock:
            if not self._index_built or self.size == 0:
                return None

            sims, idxs = self._engine.search(query_embedding, top_k)
            if len(sims) == 0:
                return None

            best_sim = float(sims[0])
            best_idx = int(idxs[0])

            if best_sim < self._threshold:
                return None

            return SearchResult(
                user_id=self._user_ids[best_idx],
                name=self._names[best_idx],
                department=self._departments[best_idx],
                similarity=best_sim,
            )

    def search_top_k(
        self, query_embedding: np.ndarray, k: int = 5
    ) -> List[SearchResult]:
        """Return top-k candidates regardless of threshold."""
        with self._lock:
            if not self._index_built or self.size == 0:
                return []
            sims, idxs = self._engine.search(query_embedding, k)
            results = []
            for sim, idx in zip(sims, idxs):
                if idx < 0 or idx >= len(self._user_ids):
                    continue
                results.append(
                    SearchResult(
                        user_id=self._user_ids[idx],
                        name=self._names[idx],
                        department=self._departments[idx],
                        similarity=float(sim),
                    )
                )
            return results

    @property
    def size(self) -> int:
        return len(self._user_ids)

    @property
    def is_ready(self) -> bool:
        return self._index_built and self.size > 0
