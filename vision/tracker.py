"""
vision/tracker.py
=================
Centroid-based face tracker with IoU assignment.

Purpose
-------
Running the heavy ML detector *every* frame would make the system
too slow on CPU.  This tracker bridges detection cycles:

  Frame 0   → detector fires  → create/update tracks
  Frame 1-5 → tracker only    → propagate bboxes, maintain IDs
  Frame 6   → detector fires  → re-associate detections to tracks

Algorithm
---------
1. On each frame, update existing tracks with the latest bbox
   (from detector output or predicted by simple velocity model).
2. Use the Hungarian / greedy IoU assignment to match new
   detections to existing tracks.
3. Tracks not updated for `max_disappeared` frames are removed.
4. New detections with no match spawn new tracks.

Track lifecycle
---------------
  NEW  → ACTIVE (once confirmed after `min_hits` frames) → LOST → removed
"""

from __future__ import annotations

import threading
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
from loguru import logger

from config.settings import AppConfig
from vision.detector import FaceDetection


# ─────────────────────────────────────────────────────────────────────────────
# Track state
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Track:
    """
    Represents a single continuously tracked face.

    Fields
    ------
    track_id       : unique integer ID (monotonically increasing)
    bbox           : current (x1,y1,x2,y2)
    confidence     : latest detection confidence
    landmarks      : latest 5-point landmarks (may be None)
    age            : total frames this track has existed
    hits           : frames with a detection match
    disappeared    : consecutive frames with no match
    identity_id    : resolved DB user ID (None until recognized)
    identity_name  : human-readable name
    identity_conf  : recognition cosine similarity
    liveness_score : anti-spoofing score [0,1]
    liveness_passed: did most recent liveness check pass?
    recognition_done: True once identity has been resolved
    """

    track_id: int
    bbox: Tuple[int, int, int, int]
    confidence: float
    landmarks: Optional[np.ndarray] = field(default=None, repr=False)
    age: int = 0
    hits: int = 1
    disappeared: int = 0

    # Recognition results (filled in by recognition module)
    identity_id: Optional[int] = None
    identity_name: Optional[str] = None
    identity_conf: float = 0.0
    liveness_score: float = 0.0
    liveness_passed: bool = False
    recognition_done: bool = False
    attendance_logged: bool = False

    # Velocity (smoothed) for bbox prediction
    _vx: float = field(default=0.0, repr=False)
    _vy: float = field(default=0.0, repr=False)

    @property
    def center(self) -> Tuple[int, int]:
        return (
            (self.bbox[0] + self.bbox[2]) // 2,
            (self.bbox[1] + self.bbox[3]) // 2,
        )

    @property
    def width(self) -> int:
        return self.bbox[2] - self.bbox[0]

    @property
    def height(self) -> int:
        return self.bbox[3] - self.bbox[1]

    @property
    def is_confirmed(self) -> bool:
        return self.hits >= 2 and self.disappeared == 0

    def predict(self) -> "Track":
        """Apply simple linear velocity prediction."""
        x1, y1, x2, y2 = self.bbox
        self.bbox = (
            int(x1 + self._vx),
            int(y1 + self._vy),
            int(x2 + self._vx),
            int(y2 + self._vy),
        )
        self.disappeared += 1
        return self

    def update(self, detection: FaceDetection) -> "Track":
        """Merge a new detection into this track."""
        alpha = 0.7  # smoothing factor
        cx_old, cy_old = self.center
        x1, y1, x2, y2 = detection.bbox
        cx_new = (x1 + x2) // 2
        cy_new = (y1 + y2) // 2

        self._vx = alpha * self._vx + (1 - alpha) * (cx_new - cx_old)
        self._vy = alpha * self._vy + (1 - alpha) * (cy_new - cy_old)

        self.bbox = detection.bbox
        self.confidence = detection.confidence
        if detection.landmarks is not None:
            self.landmarks = detection.landmarks
        self.hits += 1
        self.disappeared = 0
        self.age += 1
        return self


# ─────────────────────────────────────────────────────────────────────────────
# Assignment helpers
# ─────────────────────────────────────────────────────────────────────────────

def _iou(box_a: Tuple, box_b: Tuple) -> float:
    """Intersection-over-Union for two (x1,y1,x2,y2) boxes."""
    xa1, ya1, xa2, ya2 = box_a
    xb1, yb1, xb2, yb2 = box_b

    inter_x1 = max(xa1, xb1)
    inter_y1 = max(ya1, yb1)
    inter_x2 = min(xa2, xb2)
    inter_y2 = min(ya2, yb2)

    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter = inter_w * inter_h

    area_a = max(0, xa2 - xa1) * max(0, ya2 - ya1)
    area_b = max(0, xb2 - xb1) * max(0, yb2 - yb1)
    union = area_a + area_b - inter

    return inter / union if union > 0 else 0.0


def _centroid_distance(bbox_a: Tuple, bbox_b: Tuple) -> float:
    cx_a = (bbox_a[0] + bbox_a[2]) / 2
    cy_a = (bbox_a[1] + bbox_a[3]) / 2
    cx_b = (bbox_b[0] + bbox_b[2]) / 2
    cy_b = (bbox_b[1] + bbox_b[3]) / 2
    return float(np.hypot(cx_a - cx_b, cy_a - cy_b))


def _greedy_assign(
    tracks: List[Track],
    detections: List[FaceDetection],
    max_distance: int,
    iou_min: float,
) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
    """
    Greedy IoU + centroid distance assignment.

    Returns
    -------
    matches        : list of (track_idx, det_idx) pairs
    unmatched_trks : track indices with no detection
    unmatched_dets : detection indices with no track
    """
    if not tracks or not detections:
        return [], list(range(len(tracks))), list(range(len(detections)))

    n, m = len(tracks), len(detections)
    cost = np.full((n, m), 1e6)

    for ti, trk in enumerate(tracks):
        for di, det in enumerate(detections):
            iou = _iou(trk.bbox, det.bbox)
            dist = _centroid_distance(trk.bbox, det.bbox)
            if iou >= iou_min or dist <= max_distance:
                cost[ti, di] = dist * (1 - iou)

    matches, used_t, used_d = [], set(), set()
    # Sort by cost ascending
    indices = np.dstack(np.unravel_index(np.argsort(cost.ravel()), cost.shape))[0]
    for ti, di in indices:
        if cost[ti, di] >= 1e6:
            break
        if ti not in used_t and di not in used_d:
            matches.append((int(ti), int(di)))
            used_t.add(ti)
            used_d.add(di)

    unmatched_trks = [i for i in range(n) if i not in used_t]
    unmatched_dets = [i for i in range(m) if i not in used_d]
    return matches, unmatched_trks, unmatched_dets


# ─────────────────────────────────────────────────────────────────────────────
# FaceTracker
# ─────────────────────────────────────────────────────────────────────────────

class FaceTracker:
    """
    Maintains identity continuity between detector runs.

    Thread-safe for concurrent read from the UI thread and
    write from the inference thread.
    """

    def __init__(self, cfg: AppConfig) -> None:
        self._cfg = cfg
        self._tcfg = cfg.tracker
        self._tracks: Dict[int, Track] = OrderedDict()
        self._next_id = 1
        self._lock = threading.Lock()

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def update(self, detections: List[FaceDetection]) -> List[Track]:
        """
        Integrate new detections and return the current active tracks.

        Call this every time the detector fires.
        Between detector calls, call `predict()` to advance tracks.
        """
        with self._lock:
            track_list = list(self._tracks.values())
            matches, unmatched_trks, unmatched_dets = _greedy_assign(
                track_list,
                detections,
                self._tcfg.max_distance,
                self._tcfg.iou_min,
            )

            # Update matched tracks
            for ti, di in matches:
                track_list[ti].update(detections[di])

            # Age out unmatched tracks
            for ti in unmatched_trks:
                track_list[ti].predict()

            # Remove lost tracks
            to_remove = [
                t.track_id for t in track_list
                if t.disappeared > self._tcfg.max_disappeared
            ]
            for tid in to_remove:
                logger.debug("Track {} lost after {} frames", tid, self._tcfg.max_disappeared)
                del self._tracks[tid]

            # Create new tracks for unmatched detections
            for di in unmatched_dets:
                det = detections[di]
                new_track = Track(
                    track_id=self._next_id,
                    bbox=det.bbox,
                    confidence=det.confidence,
                    landmarks=det.landmarks,
                )
                self._tracks[self._next_id] = new_track
                logger.debug("New track created: id={}", self._next_id)
                self._next_id += 1

            return [t for t in self._tracks.values() if t.is_confirmed]

    def predict(self) -> List[Track]:
        """
        Advance all tracks by one frame without a detection.
        Call this when the detector is skipped (frame interval).
        """
        with self._lock:
            for track in list(self._tracks.values()):
                track.age += 1
                if track.disappeared > 0:  # already unmatched; predict further
                    track.predict()
                else:
                    track.disappeared += 1  # mark as temporarily missing

            # Remove stale
            for tid in [
                t.track_id for t in self._tracks.values()
                if t.disappeared > self._tcfg.max_disappeared
            ]:
                del self._tracks[tid]

            return list(self._tracks.values())

    def get_active_tracks(self) -> List[Track]:
        """Thread-safe snapshot of confirmed, active tracks."""
        with self._lock:
            return [t for t in self._tracks.values() if t.is_confirmed]

    def get_track(self, track_id: int) -> Optional[Track]:
        with self._lock:
            return self._tracks.get(track_id)

    def update_recognition(
        self,
        track_id: int,
        identity_id: Optional[int],
        identity_name: Optional[str],
        identity_conf: float,
        liveness_score: float,
        liveness_passed: bool,
    ) -> None:
        """
        Write recognition results back into a track (called from
        the inference worker thread).
        """
        with self._lock:
            track = self._tracks.get(track_id)
            if track is None:
                return
            track.identity_id = identity_id
            track.identity_name = identity_name
            track.identity_conf = identity_conf
            track.liveness_score = liveness_score
            track.liveness_passed = liveness_passed
            track.recognition_done = True

    def mark_attendance_logged(self, track_id: int) -> None:
        with self._lock:
            track = self._tracks.get(track_id)
            if track:
                track.attendance_logged = True

    def reset_recognition(self, track_id: int) -> None:
        """Allow re-recognition for a track (e.g., after cooldown)."""
        with self._lock:
            track = self._tracks.get(track_id)
            if track:
                track.recognition_done = False
                track.attendance_logged = False

    @property
    def track_count(self) -> int:
        with self._lock:
            return len(self._tracks)
