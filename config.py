"""Central configuration for the Driver Monitoring System pipeline.

All tunable thresholds, filter parameters, and gating limits are defined here.
Modify these values to adapt the pipeline to different cameras, vehicles, or
detection sensitivity requirements.

WARNING: All feature extraction and oculomotor thresholds are uncalibrated placeholders.
"""
import json
from dataclasses import dataclass, field, fields
from typing import Optional, Tuple


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
    target_fps: float = 60.0
    anti_motion_blur: bool = False

    # ── Camera Intrinsics & Angular Scaling ──
    camera_calibration_path: Optional[str] = None  # Optional path to JSON/NPZ with camera_matrix & dist_coeffs
    focal_length_px: Optional[float] = None        # Optional explicit focal length in pixels (None = frame_width)
    deg_per_iris_unit: float = 15.0                # Standard angular scale (deg per normalized iris unit)  # [PLACEHOLDER]
    saccade_deg_per_unit: float = 22.0             # Saccadic angular scale (deg per normalized iris unit)  # [PLACEHOLDER]

    # ── Saccade, Blink & Sway Internal Thresholds (Formerly Inline Magic Numbers) ──
    saccade_subpixel_deadzone_dx: float = 0.026    # Normalized horizontal sub-pixel noise floor (~0.5 px)
    saccade_subpixel_deadzone_hypot: float = 0.030 # Combined 2D sub-pixel noise floor
    saccade_head_motion_dx: float = 0.055          # Min iris shift required during active head movement
    eye_socket_vel_threshold: float = 0.75         # Normalized eye-socket translation speed for head-motion guard
    blink_phase_timeout_frames: int = 8            # Max frames allowed in closing/opening phase before timeout
    sway_window_sec: float = 3.0                   # Rolling window (s) for high-pass postural sway detrending
    sway_yaw_gate_deg_s: float = 5.0               # Yaw angular speed (deg/s) above which postural sway pauses
    sway_nod_reset_deg_s: float = 35.0             # Pitch/roll angular speed (deg/s) that clears sway history
    rigid_head_sway_threshold: float = 0.12        # Sway residual (deg) below which rigid neck locking is flagged

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
    roi_alpha: float = 0.15
    max_occlusion_hold_frames: int = 15  # max frame dropouts tolerated before state wipeout (0.5s at 30fps)

    # ── 1-Euro Filter ──
    landmark_min_cutoff: float = 0.8   # smoothing at rest
    landmark_beta: float = 0.04       # speed coefficient
    landmark_d_cutoff: float = 2.0     # derivative cutoff
    eyelid_min_cutoff: float = 2.0     # smoothing for fast eyelid landmarks
    eyelid_beta: float = 0.50          # speed coefficient for eyelids
    eyelid_d_cutoff: float = 8.0       # derivative cutoff for eyelids
    iris_min_cutoff: float = 1.5       # smoothing for iris landmarks at rest
    iris_beta: float = 0.55            # fast response for relative iris saccades
    iris_d_cutoff: float = 8.0         # fast derivative cutoff for 1-2 frame saccades
    corner_min_cutoff: float = 0.6     # low cutoff for stabilized eye corners
    corner_beta: float = 0.01          # low speed coefficient to prevent eye-corner jitter injection
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
    saccade_onset_velocity: float = 35.0      # deg/s: movement above this flags saccade start  # [PLACEHOLDER]
    saccade_offset_velocity: float = 20.0     # deg/s: movement below this flags saccade end (hysteresis)  # [PLACEHOLDER]

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
    head_accel_threshold: float = 4.0  # m/s^2  # [PLACEHOLDER]
    imu_freeze_ms: float = 150.0      # freeze window after shock  # [PLACEHOLDER]
    imu_accel_window: int = 5          # 5-frame Savitzky-Golay polynomial window
    continuous_vibration_accel_min: float = 1.5   # m/s^2 min acceleration for continuous vibration  # [PLACEHOLDER]
    continuous_vibration_duration_s: float = 0.5  # duration (s) to activate continuous vibration mode  # [PLACEHOLDER]
    max_freeze_duty_cycle: float = 0.25           # maximum freeze duty cycle over rolling 2.0s window  # [PLACEHOLDER]

    # ── Output ──
    save_csv: bool = False
    csv_output_path: str = "output_features.csv"
    headless: bool = False  # if True, skip OpenCV visualization window
    show_mesh: bool = True
    show_iris: bool = True
    show_pose_axes: bool = True

    @classmethod
    def from_json(cls, json_path: str) -> "PipelineConfig":
        """Load pipeline configuration from a JSON file, overriding defaults.

        Unknown keys in the JSON are safely ignored.
        """
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        valid_fields = {f.name for f in fields(cls)}
        kwargs = {k: v for k, v in data.items() if k in valid_fields}
        return cls(**kwargs)


# Default configuration instance
DEFAULT_CONFIG = PipelineConfig()
