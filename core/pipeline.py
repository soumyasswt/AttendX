"""
core/pipeline.py
================
The inference pipeline – the heart of the application.

Runs in a dedicated background thread and orchestrates:

  Frame → Detect (every N frames) → Track → Align → Embed →
  Search → Liveness → Attendance decision → DB log → UI notify

Threading model
---------------
  Camera thread  ─[frame queue]→  Pipeline thread  ─[result queue]→  UI thread

The pipeline thread reads frames from `camera.read()`, runs
inference, then puts `PipelineResult` objects into a result queue
for the UI to consume.

Frame skipping
--------------
Detection runs every `vision.detection_interval` frames; between
runs the tracker predicts positions from velocity.  Embedding is
extracted only for tracks that have NOT yet been recognized.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from queue import Full, Queue
from typing import Dict, List, Optional

import cv2
import numpy as np
from loguru import logger

from config.settings import AppConfig
from core.camera import CameraCapture
from database.db import DB
from recognition.embedder import ArcFaceEmbedder
from recognition.searcher import EmbeddingSearcher
from security.antispoofing import LivenessDetector
from vision.aligner import FaceAligner
from vision.detector import FaceDetection, FaceDetector
from vision.tracker import FaceTracker, Track


# ─────────────────────────────────────────────────────────────────────────────
# Result type sent to the UI
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class FaceResult:
    """Per-face annotation data for the UI overlay."""
    track_id: int
    bbox: tuple
    name: str                    # "Unknown" or enrolled name
    identity_conf: float
    liveness_score: float
    liveness_passed: bool
    event_type: Optional[str]    # "ENTRY" | "EXIT" | None
    color: tuple = (0, 255, 0)   # BGR annotation color


@dataclass
class PipelineResult:
    """Complete result for a single processed frame."""
    frame: np.ndarray = field(repr=False)
    faces: List[FaceResult] = field(default_factory=list)
    fps: float = 0.0
    frame_idx: int = 0
    timestamp: float = field(default_factory=time.time)


# ─────────────────────────────────────────────────────────────────────────────
# In-memory recognition cache
# ─────────────────────────────────────────────────────────────────────────────

class RecognitionCache:
    """
    Short-lived in-memory cache mapping track_id → recognition result.
    Avoids redundant DB queries when the same face is in view.
    """

    def __init__(self, ttl_seconds: int = 300) -> None:
        self._cache: Dict[int, dict] = {}
        self._timestamps: Dict[int, float] = {}
        self._ttl = ttl_seconds
        self._lock = threading.Lock()

    def get(self, track_id: int) -> Optional[dict]:
        with self._lock:
            if track_id not in self._cache:
                return None
            if time.time() - self._timestamps[track_id] > self._ttl:
                del self._cache[track_id]
                del self._timestamps[track_id]
                return None
            return self._cache[track_id]

    def set(self, track_id: int, data: dict) -> None:
        with self._lock:
            self._cache[track_id] = data
            self._timestamps[track_id] = time.time()

    def invalidate(self, track_id: int) -> None:
        with self._lock:
            self._cache.pop(track_id, None)
            self._timestamps.pop(track_id, None)


# ─────────────────────────────────────────────────────────────────────────────
# Inference Pipeline
# ─────────────────────────────────────────────────────────────────────────────

class InferencePipeline:
    """
    Orchestrates the full CV inference pipeline in a background thread.

    Constructor parameters are injected (dependency injection pattern)
    so each component can be mocked or swapped in tests.
    """

    def __init__(
        self,
        cfg: AppConfig,
        camera: CameraCapture,
        detector: FaceDetector,
        tracker: FaceTracker,
        aligner: FaceAligner,
        embedder: ArcFaceEmbedder,
        searcher: EmbeddingSearcher,
        liveness: LivenessDetector,
        db: DB,
        result_queue: Queue,
    ) -> None:
        self._cfg = cfg
        self._camera = camera
        self._detector = detector
        self._tracker = tracker
        self._aligner = aligner
        self._embedder = embedder
        self._searcher = searcher
        self._liveness = liveness
        self._db = db
        self._result_q = result_queue

        self._cache = RecognitionCache(
            ttl_seconds=cfg.performance.embedding_cache_ttl_seconds
        )

        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._frame_idx = 0

        # FPS measurement
        self._fps_counter = 0
        self._fps_time = time.time()
        self._current_fps = 0.0

    # ------------------------------------------------------------------ #
    # Lifecycle                                                            #
    # ------------------------------------------------------------------ #

    def start(self) -> None:
        self._running = True
        self._thread = threading.Thread(
            target=self._pipeline_loop,
            name="InferencePipeline",
            daemon=True,
        )
        self._thread.start()
        logger.info("Inference pipeline started")

    def stop(self) -> None:
        self._running = False
        if self._thread:
            self._thread.join(timeout=5.0)
        self._liveness.close()
        logger.info("Inference pipeline stopped")

    # ------------------------------------------------------------------ #
    # Main loop                                                            #
    # ------------------------------------------------------------------ #

    def _pipeline_loop(self) -> None:
        interval = self._cfg.vision.detection_interval

        while self._running:
            frame = self._camera.read(timeout=0.1)
            if frame is None:
                continue

            self._frame_idx += 1
            self._fps_counter += 1

            # ── Stage 1: Detection (every N frames) ──────────────────────
            if self._frame_idx % interval == 0:
                detections = self._detector.detect(frame)
                active_tracks = self._tracker.update(detections)
            else:
                active_tracks = self._tracker.predict()

            # ── Stage 2: Per-track recognition ────────────────────────────
            face_results: List[FaceResult] = []

            for track in active_tracks:
                result = self._process_track(frame, track)
                if result:
                    face_results.append(result)

            # ── Stage 3: Annotate frame ───────────────────────────────────
            annotated = self._draw_annotations(frame.copy(), face_results)

            # ── Stage 4: Push to UI queue ─────────────────────────────────
            self._update_fps()
            pipeline_result = PipelineResult(
                frame=annotated,
                faces=face_results,
                fps=self._current_fps,
                frame_idx=self._frame_idx,
            )

            try:
                self._result_q.put_nowait(pipeline_result)
            except Full:
                try:
                    self._result_q.get_nowait()
                except Exception:
                    pass
                try:
                    self._result_q.put_nowait(pipeline_result)
                except Full:
                    pass

    # ------------------------------------------------------------------ #
    # Per-track processing                                                 #
    # ------------------------------------------------------------------ #

    def _process_track(
        self, frame: np.ndarray, track: Track
    ) -> Optional[FaceResult]:
        """Run recognition + liveness + attendance for one track."""

        # ── Already fully recognized and attendance logged ─────────────
        if track.recognition_done and track.attendance_logged:
            return FaceResult(
                track_id=track.track_id,
                bbox=track.bbox,
                name=track.identity_name or "Unknown",
                identity_conf=track.identity_conf,
                liveness_score=track.liveness_score,
                liveness_passed=track.liveness_passed,
                event_type=None,
                color=self._annotation_color(track),
            )

        # ── Crop and align face ────────────────────────────────────────
        aligned = self._aligner.align(frame, track.bbox, track.landmarks)
        if aligned is None:
            return None

        # ── Liveness check ─────────────────────────────────────────────
        face_crop = track.bbox
        raw_crop = frame[
            max(0, face_crop[1]):face_crop[3],
            max(0, face_crop[0]):face_crop[2],
        ]
        if raw_crop.size == 0:
            return None

        liveness = self._liveness.check(track.track_id, raw_crop)
        track.liveness_score = liveness.score
        track.liveness_passed = liveness.passed

        # ── Recognition (only when liveness passes) ────────────────────
        if not track.recognition_done:
            if not liveness.passed:
                # Still accumulating liveness evidence; show unknown
                return FaceResult(
                    track_id=track.track_id,
                    bbox=track.bbox,
                    name="Verifying…",
                    identity_conf=0.0,
                    liveness_score=liveness.score,
                    liveness_passed=False,
                    event_type=None,
                    color=(0, 165, 255),  # orange
                )

            embedding = self._embedder.get_embedding(aligned)
            if embedding is None:
                return None

            search_result = self._searcher.search(embedding)

            if search_result is not None:
                track.identity_id = search_result.user_id
                track.identity_name = search_result.name
                track.identity_conf = search_result.similarity
            else:
                track.identity_id = None
                track.identity_name = "Unknown"
                track.identity_conf = 0.0

            track.recognition_done = True

        # ── Attendance decision ────────────────────────────────────────
        event_type = None
        if (
            track.recognition_done
            and not track.attendance_logged
            and track.identity_id is not None
            and track.liveness_passed
            and track.identity_conf >= self._cfg.attendance.min_recognition_confidence
        ):
            in_cooldown = self._att_q.is_in_cooldown(
                track.identity_id,
                self._cfg.attendance.cooldown_minutes,
            )
            if not in_cooldown:
                event_type = self._att_q.determine_event_type(track.identity_id)
                self._log_attendance(track, event_type)
                track.attendance_logged = True

        return FaceResult(
            track_id=track.track_id,
            bbox=track.bbox,
            name=track.identity_name or "Unknown",
            identity_conf=track.identity_conf,
            liveness_score=track.liveness_score,
            liveness_passed=track.liveness_passed,
            event_type=event_type,
            color=self._annotation_color(track),
        )

    def _log_attendance(self, track: Track, event_type: str) -> None:
        try:
            if event_type == "ENTRY":
                self._att_q.log_entry(
                    user_id=track.identity_id,
                    recognition_score=track.identity_conf,
                    liveness_score=track.liveness_score,
                    late_threshold_hour=self._cfg.attendance.late_arrival_hour,
                )
            else:
                self._att_q.log_exit(
                    user_id=track.identity_id,
                    recognition_score=track.identity_conf,
                    liveness_score=track.liveness_score,
                )
            self._att_q.log_recognition_event(
                user_id=track.identity_id,
                similarity=track.identity_conf,
                liveness_score=track.liveness_score,
                action=event_type.lower(),
            )
            logger.info(
                "{} logged: {} (conf={:.3f}, liveness={:.3f})",
                event_type, track.identity_name, track.identity_conf, track.liveness_score
            )
        except Exception as exc:
            logger.error("Failed to log attendance: {}", exc)

    # ------------------------------------------------------------------ #
    # Frame annotation                                                     #
    # ------------------------------------------------------------------ #

    def _draw_annotations(
        self, frame: np.ndarray, faces: List[FaceResult]
    ) -> np.ndarray:
        for face in faces:
            x1, y1, x2, y2 = face.bbox
            color = face.color

            # Bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            # Corner markers (professional style)
            length = max(15, (x2 - x1) // 5)
            lw = 3
            for (px, py, dx, dy) in [
                (x1, y1,  1,  1), (x2, y1, -1,  1),
                (x1, y2,  1, -1), (x2, y2, -1, -1),
            ]:
                cv2.line(frame, (px, py), (px + dx * length, py), color, lw)
                cv2.line(frame, (px, py), (px, py + dy * length), color, lw)

            # Name label
            label = face.name
            if face.identity_conf > 0:
                label += f"  {face.identity_conf:.0%}"
            if face.event_type:
                label += f"  [{face.event_type}]"

            font = cv2.FONT_HERSHEY_SIMPLEX
            scale, thickness = 0.6, 2
            (tw, th), baseline = cv2.getTextSize(label, font, scale, thickness)
            label_y = max(y1 - 10, th + 10)

            # Label background
            cv2.rectangle(
                frame,
                (x1, label_y - th - baseline - 5),
                (x1 + tw + 10, label_y + 5),
                color, -1,
            )
            # Label text (dark)
            cv2.putText(
                frame, label,
                (x1 + 5, label_y),
                font, scale, (0, 0, 0), thickness, cv2.LINE_AA,
            )

            # Liveness indicator dot
            dot_color = (0, 255, 0) if face.liveness_passed else (0, 0, 255)
            cv2.circle(frame, (x2 - 10, y1 + 10), 6, dot_color, -1)

        # FPS counter
        cv2.putText(
            frame, f"FPS: {self._current_fps:.1f}",
            (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2,
        )
        return frame

    @staticmethod
    def _annotation_color(track: Track) -> tuple:
        if track.identity_id is None:
            return (0, 0, 255)          # red = unknown
        if not track.liveness_passed:
            return (0, 165, 255)        # orange = liveness fail
        return (0, 255, 0)              # green = recognized + live

    def _update_fps(self) -> None:
        self._fps_counter += 1
        now = time.time()
        elapsed = now - self._fps_time
        if elapsed >= 1.0:
            self._current_fps = self._fps_counter / elapsed
            self._fps_counter = 0
            self._fps_time = now
