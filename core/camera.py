"""
core/camera.py
==============
Threaded camera capture service with frame buffering.

Runs capture in a dedicated daemon thread so the main thread
(and UI) never blocks on I/O.  Exposes a thread-safe queue
that downstream consumers can read at their own pace.

Architecture note
-----------------
This sits at the very first stage of the CV pipeline.
The bounded queue (maxsize=config.performance.frame_queue_maxsize)
acts as backpressure: when downstream is slow, oldest frames are
dropped (newest-frame policy) rather than queuing forever.
"""

from __future__ import annotations

import threading
import time
from queue import Empty, Full, Queue
from typing import Optional, Tuple

import cv2
import numpy as np
from loguru import logger

from config.settings import AppConfig


class FrameBuffer:
    """
    Thread-safe newest-frame buffer.

    Unlike a normal FIFO queue, this always provides the *latest*
    captured frame.  Stale frames are discarded automatically.
    """

    def __init__(self) -> None:
        self._frame: Optional[np.ndarray] = None
        self._lock = threading.Lock()
        self._event = threading.Event()

    def put(self, frame: np.ndarray) -> None:
        with self._lock:
            self._frame = frame
        self._event.set()

    def get(self, timeout: float = 1.0) -> Optional[np.ndarray]:
        if not self._event.wait(timeout):
            return None
        with self._lock:
            frame = self._frame
            self._event.clear()
        return frame

    @property
    def has_frame(self) -> bool:
        return self._frame is not None


class CameraCapture:
    """
    Manages a single webcam using a background capture thread.

    Usage
    -----
    >>> cam = CameraCapture(cfg)
    >>> cam.start()
    >>> frame = cam.read()   # non-blocking; None if no frame yet
    >>> cam.stop()
    """

    def __init__(self, cfg: AppConfig) -> None:
        self._cfg = cfg
        self._cam_cfg = cfg.camera
        self._cap: Optional[cv2.VideoCapture] = None

        # Output queue – bounded so we never pile up stale frames
        self._queue: Queue[np.ndarray] = Queue(
            maxsize=cfg.performance.frame_queue_maxsize
        )

        self._running = False
        self._thread: Optional[threading.Thread] = None

        # Diagnostics
        self._frame_count: int = 0
        self._dropped_frames: int = 0
        self._fps_measured: float = 0.0
        self._last_fps_time: float = 0.0
        self._fps_frame_counter: int = 0

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def start(self) -> "CameraCapture":
        """Open the camera and start the capture thread."""
        if self._running:
            logger.warning("CameraCapture.start() called while already running")
            return self

        self._open_camera()
        self._running = True
        self._thread = threading.Thread(
            target=self._capture_loop,
            name="CameraCapture",
            daemon=True,
        )
        self._thread.start()
        logger.info(
            "Camera capture started: device={} res={}x{} fps={}",
            self._cam_cfg.device_id,
            self._cam_cfg.width,
            self._cam_cfg.height,
            self._cam_cfg.fps,
        )
        return self

    def stop(self) -> None:
        """Signal the capture thread to exit and release hardware."""
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=3.0)
        if self._cap and self._cap.isOpened():
            self._cap.release()
            logger.info("Camera released")

    def read(self, timeout: float = 0.05) -> Optional[np.ndarray]:
        """
        Read the next frame from the queue.

        Returns None if no frame is available within *timeout* seconds.
        This method is safe to call from any thread.
        """
        try:
            return self._queue.get(timeout=timeout)
        except Empty:
            return None

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def frame_count(self) -> int:
        return self._frame_count

    @property
    def dropped_frames(self) -> int:
        return self._dropped_frames

    @property
    def measured_fps(self) -> float:
        return self._fps_measured

    # ------------------------------------------------------------------ #
    # Internal helpers                                                     #
    # ------------------------------------------------------------------ #

    def _open_camera(self) -> None:
        """Open cv2.VideoCapture with the configured parameters."""
        cap = cv2.VideoCapture(self._cam_cfg.device_id)
        if not cap.isOpened():
            raise RuntimeError(
                f"Cannot open camera device {self._cam_cfg.device_id}"
            )

        # Apply preferred codec (MJPG gives max FPS on USB cams)
        try:
            fourcc = cv2.VideoWriter_fourcc(*self._cam_cfg.fourcc)
            cap.set(cv2.CAP_PROP_FOURCC, fourcc)
        except Exception:
            pass  # Ignore if codec not supported

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._cam_cfg.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._cam_cfg.height)
        cap.set(cv2.CAP_PROP_FPS, self._cam_cfg.fps)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, self._cam_cfg.buffer_size)

        # Verify actual resolution (camera may not support requested)
        actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = cap.get(cv2.CAP_PROP_FPS)
        logger.debug(
            "Actual camera params: {}x{} @{}fps",
            actual_w, actual_h, actual_fps
        )

        self._cap = cap
        self._actual_width = actual_w
        self._actual_height = actual_h

    def _capture_loop(self) -> None:
        """Background thread: read frames and push to queue."""
        self._last_fps_time = time.time()
        logger.debug("Capture loop started")
        consecutive_failures = 0
        max_consecutive_failures = 100

        while self._running:
            if self._cap is None or not self._cap.isOpened():
                logger.error("Camera lost; attempting to reopen...")
                try:
                    if self._cap:
                        self._cap.release()
                    self._open_camera()
                    logger.info("Camera reopened successfully")
                    consecutive_failures = 0
                except Exception as e:
                    consecutive_failures += 1
                    logger.warning(
                        "Failed to reopen camera (attempt {}): {}",
                        consecutive_failures, e
                    )
                    if consecutive_failures >= max_consecutive_failures:
                        logger.error("Too many camera reopening failures; giving up")
                        self._running = False
                        break
                    time.sleep(1)
                    continue

            ret, frame = self._cap.read()
            if not ret:
                consecutive_failures += 1
                if consecutive_failures % 20 == 0:  # Log every 20 failures
                    logger.warning(
                        "cap.read() failed ({} consecutive); camera may be disconnected",
                        consecutive_failures
                    )
                if consecutive_failures >= max_consecutive_failures:
                    logger.error("Camera read failed too many times; stopping")
                    self._running = False
                    break
                time.sleep(0.05)
                continue
            
            consecutive_failures = 0  # Reset on successful read

            self._frame_count += 1
            self._fps_frame_counter += 1

            # Optional: scale down before queueing (saves downstream work)
            scale = self._cfg.vision.scale_factor
            if scale != 1.0:
                new_w = int(self._actual_width * scale)
                new_h = int(self._actual_height * scale)
                frame = cv2.resize(frame, (new_w, new_h))

            # Non-blocking put; drop if consumer is slow (newest-frame policy)
            try:
                self._queue.put_nowait(frame)
            except Full:
                # Drain one old frame, replace with new
                try:
                    self._queue.get_nowait()
                except Empty:
                    pass
                try:
                    self._queue.put_nowait(frame)
                except Full:
                    self._dropped_frames += 1

            # Update measured FPS every second
            now = time.time()
            if now - self._last_fps_time >= 1.0:
                self._fps_measured = self._fps_frame_counter / (
                    now - self._last_fps_time
                )
                self._fps_frame_counter = 0
                self._last_fps_time = now

        logger.debug("Capture loop exited (total frames: {})", self._frame_count)

    def get_frame_dimensions(self) -> Tuple[int, int]:
        """Return (width, height) of captured frames."""
        if hasattr(self, "_actual_width"):
            scale = self._cfg.vision.scale_factor
            return (
                int(self._actual_width * scale),
                int(self._actual_height * scale),
            )
        return (self._cam_cfg.width, self._cam_cfg.height)
