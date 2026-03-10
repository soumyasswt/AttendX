"""
config/settings.py
==================
Centralized configuration for the Smart Attendance System.
All tuneable parameters live here; no magic numbers in application code.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Tuple


# ─────────────────────────────────────────────
# Sub-config dataclasses
# ─────────────────────────────────────────────

@dataclass
class CameraConfig:
    """Hardware capture settings."""
    device_id: int = 0
    width: int = 1280
    height: int = 720
    fps: int = 30
    buffer_size: int = 2          # OS capture buffer (lower = less latency)
    fourcc: str = "MJPG"          # Codec; MJPG gives best USB bandwidth


@dataclass
class VisionConfig:
    """Computer vision pipeline tuning."""
    detection_interval: int = 6          # Run heavy detector every N frames
    detection_confidence: float = 0.65   # RetinaFace / YOLO face score threshold
    min_face_size: int = 60              # Pixels; smaller faces ignored
    nms_threshold: float = 0.4           # Non-maximum suppression IoU threshold
    alignment_size: Tuple[int, int] = (112, 112)  # ArcFace canonical input size
    scale_factor: float = 1.0            # Resize input before detection (0.5 = half)
    max_faces: int = 10                  # Cap simultaneous tracked faces


@dataclass
class TrackerConfig:
    """Centroid / IoU tracker parameters."""
    max_disappeared: int = 40   # Frames before a track is dropped
    max_distance: int = 120     # Max pixel centroid movement per frame
    iou_min: float = 0.25       # Minimum IoU for assignment


@dataclass
class RecognitionConfig:
    """Face recognition / embedding settings."""
    embedding_dim: int = 512
    similarity_threshold: float = 0.55   # Cosine sim; above = same person
    model_name: str = "buffalo_l"        # insightface model pack
    use_gpu: bool = False                 # Set True if CUDA available
    ctx_id: int = -1                      # -1 = CPU; 0+ = GPU id


@dataclass
class SecurityConfig:
    """Anti-spoofing and liveness settings."""
    enabled: bool = True
    liveness_threshold: float = 0.60     # Combined score to pass
    ear_blink_threshold: float = 0.22    # Eye Aspect Ratio for blink
    blink_required: bool = True           # Require at least one blink
    texture_lbp_threshold: float = 45.0  # LBP variance; screens score low
    min_liveness_frames: int = 3         # Consecutive passing frames required


@dataclass
class AttendanceConfig:
    """Attendance business logic."""
    cooldown_minutes: int = 5            # Ignore same person within N minutes
    late_arrival_hour: int = 9           # Hour (24h) beyond which = late
    working_hours_start: int = 8
    working_hours_end: int = 18
    min_recognition_confidence: float = 0.60


@dataclass
class DatabaseConfig:
    """SQLite data tier settings."""
    db_path: str = "data/attendance.db"
    wal_mode: bool = True                # WAL journal for concurrency
    pool_timeout: int = 10               # Seconds to wait for connection
    backup_interval_hours: int = 24


@dataclass
class UIConfig:
    """Presentation tier settings."""
    appearance_mode: str = "dark"        # "dark" | "light" | "system"
    color_theme: str = "blue"            # customtkinter theme
    window_title: str = "Smart Attendance System v1.0"
    window_min_width: int = 1280
    window_min_height: int = 760
    sidebar_width: int = 380
    feed_update_ms: int = 30             # ~33 fps UI refresh
    event_panel_max_rows: int = 50


@dataclass
class PerformanceConfig:
    """Threading and optimization knobs."""
    frame_queue_maxsize: int = 4         # Bounded queue to drop stale frames
    result_queue_maxsize: int = 20
    inference_thread_count: int = 1      # Inference workers
    db_write_batch_size: int = 5         # Batch DB writes
    embedding_cache_ttl_seconds: int = 300  # In-memory embedding cache TTL


@dataclass
class LoggingConfig:
    level: str = "INFO"
    log_dir: str = "logs"
    rotation: str = "50 MB"
    retention: str = "10 days"
    format: str = (
        "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
        "<level>{message}</level>"
    )


# ─────────────────────────────────────────────
# Root application config
# ─────────────────────────────────────────────

@dataclass
class AppConfig:
    """Single root configuration object injected into every service."""

    # Sub-configs
    camera: CameraConfig = field(default_factory=CameraConfig)
    vision: VisionConfig = field(default_factory=VisionConfig)
    tracker: TrackerConfig = field(default_factory=TrackerConfig)
    recognition: RecognitionConfig = field(default_factory=RecognitionConfig)
    security: SecurityConfig = field(default_factory=SecurityConfig)
    attendance: AttendanceConfig = field(default_factory=AttendanceConfig)
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    ui: UIConfig = field(default_factory=UIConfig)
    performance: PerformanceConfig = field(default_factory=PerformanceConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)

    # Derived paths (resolved at runtime)
    base_dir: Path = field(default_factory=lambda: Path(__file__).resolve().parent.parent)

    def __post_init__(self) -> None:
        """Resolve all path-based config fields relative to base_dir."""
        self.models_dir: Path = self.base_dir / "models"
        self.data_dir: Path = self.base_dir / "data"
        self.logs_dir: Path = self.base_dir / "logs"
        self.exports_dir: Path = self.base_dir / "exports"

        # Override db path to be absolute
        self.database.db_path = str(self.data_dir / "attendance.db")

        # Ensure dirs exist
        for d in (self.models_dir, self.data_dir, self.logs_dir, self.exports_dir):
            d.mkdir(parents=True, exist_ok=True)

        # Allow environment variable overrides for key settings
        if val := os.getenv("CAMERA_DEVICE_ID"):
            self.camera.device_id = int(val)
        if val := os.getenv("RECOGNITION_THRESHOLD"):
            self.recognition.similarity_threshold = float(val)
        if val := os.getenv("COOLDOWN_MINUTES"):
            self.attendance.cooldown_minutes = int(val)
        if val := os.getenv("DISABLE_ANTISPOOFING"):
            self.security.enabled = val.lower() not in ("1", "true", "yes")


# ─────────────────────────────────────────────
# Module-level singleton
# ─────────────────────────────────────────────

config = AppConfig()
