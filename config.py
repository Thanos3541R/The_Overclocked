"""Central configuration for the Driver Monitoring System pipeline.

All tunable thresholds, filter parameters, and gating limits are defined here.
Modify these values to adapt the pipeline to different cameras, vehicles, or
detection sensitivity requirements.

WARNING: All feature extraction and oculomotor thresholds are uncalibrated placeholders.
"""
from dataclasses import dataclass, field
from typing import Tuple


@dataclass(frozen=True)
class PipelineConfig:
    """Immutable pipeline configuration."""

    # ── Frame Capture ──
    camera_source: int = 0  # 0 = default webcam, or path to video file
    frame_width: int = 1280
    frame_height: int = 720
    # Note: 30 fps is an acceptable desktop/webcam fallback. For forensic/clinical
    # oculomotor velocity measurement (eyelid downstroke 50-100ms, saccades 20-50ms),
    # >= 60 fps (preferably 90-120 fps) global-shutter camera hardware is required
    # to eliminate derivative quantization noise.
    target_fps: float = 30.0

    # ── Preprocessing (CLAHE) ──
    clahe_clip_limit: float = 2.0
    clahe_tile_grid: Tuple[int, int] = (8, 8)
    low_light_threshold: float = 60.0  # mean luminance below this triggers gamma
    gamma_dark: float = 0.5   # gamma exponent for very dark frames (< 40 lum)
    gamma_dim: float = 0.7    # gamma exponent for dim frames (40-60 lum)

    # ── MediaPipe Face Mesh ──
    max_faces: int = 1
    refine_landmarks: bool = True  # enables iris landmarks (468-477)
    min_detection_confidence: float = 0.35
    min_tracking_confidence: float = 0.35
    enable_adaptive_roi: bool = True
    roi_margin: float = 2.0

    # ── 1-Euro Filter ──
    landmark_min_cutoff: float = 1.0   # smoothing at rest
    landmark_beta: float = 0.007       # speed coefficient
    landmark_d_cutoff: float = 1.0     # derivative cutoff
    ear_min_cutoff: float = 1.5        # less smoothing for EAR signal
    ear_beta: float = 0.003

    # ── Head Pose Gating ──
    yaw_limit: float = 35.0       # degrees, ±
    pitch_down_limit: float = -30.0  # degrees
    pitch_up_limit: float = 25.0     # degrees

    # ── Feature Extraction ──
    ear_blink_threshold: float = 0.20  # [PLACEHOLDER]
    mar_yawn_threshold: float = 0.60  # [PLACEHOLDER]
    perclos_window_sec: float = 60.0    # rolling window for PERCLOS
    perclos_alert_threshold: float = 0.15  # fraction of time eyes closed  # [PLACEHOLDER]
    microsleep_frames: int = 15        # at 30fps = 500ms  # [PLACEHOLDER]
    yawn_frames: int = 30              # at 30fps = 1 second
    gaze_dispersion_window_sec: float = 15.0  # rolling σ_yaw window (seconds)
    saccade_onset_velocity: float = 30.0      # deg/s: movement above this flags saccade start  # [PLACEHOLDER]
    saccade_offset_velocity: float = 15.0     # deg/s: movement below this flags saccade end (hysteresis)  # [PLACEHOLDER]

    # ── Advanced Oculomotor Impairment Signals ──
    eccentric_gaze_threshold: float = 0.35    # normalized iris_x threshold for GEN evaluation (~15-20 deg)  # [PLACEHOLDER]
    gen_drift_min_vel: float = 2.0            # deg/s: min centripetal drift speed  # [PLACEHOLDER]
    gen_drift_max_vel: float = 18.0           # deg/s: max centripetal drift speed  # [PLACEHOLDER]
    gen_reset_min_vel: float = 30.0           # deg/s: min outward corrective beat speed  # [PLACEHOLDER]
    loc_vergence_threshold_deg: float = -3.5  # degrees: negative vergence indicates divergent strabismus / LOC  # [PLACEHOLDER]
    vor_min_head_speed: float = 2.0           # deg/s: min head rotation to evaluate VOR gain  # [PLACEHOLDER]
    vor_max_head_speed: float = 25.0          # deg/s: max head rotation for micro-compensatory VOR  # [PLACEHOLDER]
    pursuit_window_sec: float = 5.0           # rolling window for pursuit fragmentation ratio  # [PLACEHOLDER]

    # ── Software IMU Gate ──
    head_accel_threshold: float = 2.5  # m/s^2  # [PLACEHOLDER]
    imu_freeze_ms: float = 150.0      # freeze window after shock  # [PLACEHOLDER]
    imu_accel_window: int = 5          # 5-frame Savitzky-Golay polynomial window

    # ── Output ──
    save_csv: bool = False
    csv_output_path: str = "output_features.csv"
    headless: bool = False  # if True, skip OpenCV visualization window
    show_mesh: bool = True
    show_iris: bool = True
    show_pose_axes: bool = True


# Default configuration instance
DEFAULT_CONFIG = PipelineConfig()
