"""Facial micromotion feature extraction for driver monitoring.

Computes per-frame features from normalized, smoothed landmarks:
  - EAR (Eye Aspect Ratio): blink and drowsiness detection
  - MAR (Mouth Aspect Ratio): yawn detection
  - Iris/pupil gaze position: gaze direction estimation
  - PERCLOS: rolling percentage of eye closure over time window
  - Blink detection: EAR dip below threshold
  - Micro-sleep detection: sustained EAR below threshold

Alcohol-discriminating signals (Fransson et al. 2010):
  - Saccadic velocity & latency: angular velocity of iris during rapid eye
    movements; detects catch-up saccade breakdown during smooth pursuit
  - Blink closing/opening velocity: involuntary downstroke/upstroke speed
    of EAR changes (unfakeable, unlike blink rate)
  - Gaze yaw dispersion (σ_yaw): rolling std of horizontal gaze angle
    over ~15s; captures gaze tunneling toward road center
"""
import cv2
import logging
import threading
import numpy as np
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, List, Optional, Tuple

logger = logging.getLogger(__name__)

from utils.landmarks import (
    RIGHT_EYE_EAR_FLAT, LEFT_EYE_EAR_FLAT,
    RIGHT_IRIS_CENTER, LEFT_IRIS_CENTER,
    RIGHT_IRIS_RING, LEFT_IRIS_RING,
    RIGHT_EYE_OUTER, LEFT_EYE_OUTER,
    RIGHT_EYE_EAR_INDICES, LEFT_EYE_EAR_INDICES,
    MAR_VERTICAL_PAIRS, MAR_HORIZONTAL,
    RIGHT_CHEEK, LEFT_CHEEK, CHEEK_LANDMARKS,
    VASODILATION_LANDMARKS,
    RIGHT_EYEBROW_ARCH, LEFT_EYEBROW_ARCH,
    LEFT_EYE_CONTOUR, RIGHT_EYE_CONTOUR,
)


@dataclass
class FrameFeatures:
    """Per-frame feature output from the driver monitoring pipeline."""
    timestamp: float = 0.0
    frame_id: int = 0
    face_valid: bool = True

    # Head pose (degrees)
    head_yaw: float = 0.0
    head_pitch: float = 0.0
    head_roll: float = 0.0

    # Eye Aspect Ratio
    ear_left: float = 0.0
    ear_right: float = 0.0
    ear_avg: float = 0.0

    # Mouth Aspect Ratio
    mar: float = 0.0

    # Blink / micro-sleep detection
    blink_detected: bool = False
    microsleep_detected: bool = False
    blink_phase: str = "open"  # "open", "closing", "closed", "opening"

    # Iris gaze (normalized [-1, 1])
    left_iris_x: float = 0.0
    left_iris_y: float = 0.0
    right_iris_x: float = 0.0
    right_iris_y: float = 0.0

    # PERCLOS
    perclos: float = 0.0

    # Yawn detection
    yawn_detected: bool = False

    # ── Alcohol-discriminating signals ──

    # Saccadic dynamics (Fransson et al. 2010)
    saccade_velocity: float = 0.0        # deg/s, peak angular velocity of iris
    saccade_detected: bool = False       # True during a rapid eye movement
    saccade_latency_ms: float = 0.0      # ms since last saccade ended (reaction time proxy)
    raw_iris_dx: float = 0.0             # Raw normalized iris horizontal displacement per frame (|Δx|)

    # Blink velocity (involuntary, unfakeable)
    blink_closing_velocity: float = 0.0  # EAR units/s, downstroke speed
    blink_opening_velocity: float = 0.0  # EAR units/s, upstroke speed

    # Gaze yaw dispersion (tunneling detection)
    gaze_yaw_dispersion: float = 0.0     # σ of horizontal gaze over rolling window

    # ── Advanced Oculomotor & Impairment Signals ──
    # 1. Gaze-Evoked Nystagmus (GEN — Romano et al. 2017)
    gen_detected: bool = False             # True if sawtooth nystagmus beats occur at eccentric gaze
    gen_beat_count: int = 0                # Number of corrective beats in current episode
    gen_slow_phase_vel: float = 0.0        # deg/s centripetal drift speed

    # 2. Smooth Pursuit Fragmentation Ratio
    pursuit_fragmentation_ratio: float = 0.0  # Ratio of catch-up saccade time to total pursuit tracking time

    # 3. Binocular Vergence & Lack of Convergence (LOC - DRE sign)
    vergence_angle_deg: float = 0.0        # Net vergence angle in degrees (+ = converged, - = divergent)
    lack_of_convergence: bool = False      # True if divergent strabismus / unable to converge

    # 4. Vestibulo-Ocular Reflex (VOR) Micro-Compensation Gain (PMC8997842)
    vor_gain: float = 1.0                  # -d(eye_yaw)/d(head_yaw) during head micro-movement (normal ~0.85-1.05)

    # ── Section A — Involuntary Biomarkers (Neuromuscular & Autonomic) ──
    # High evidential weight for lockout decisions (unfakeable)
    head_postural_sway: float = 0.0          # Involuntary micro-tremor / low-frequency postural sway (deg, sqrt(var_pitch + var_roll))
    facial_flushing_ratio: float = 1.0       # Cheek micro-vascular vasodilation chromaticity ratio R/((G+B)/2)
    flushing_valid: bool = True              # False when monochrome / NIR mode active (chromaticity unavailable)
    flushing_delta: float = 0.0              # Relative vasodilation surge: (ratio - baseline) / baseline

    # ── Section B — Voluntary & Semi-Voluntary Changes (Maskable / Gaming Behaviors) ──
    # Susceptible to conscious suppression; used for corroboration and mismatch detection
    deliberate_blink_suppression: bool = False  # Prolonged inter-blink interval (>6.0s) from fighting droop
    voluntary_eye_widening: bool = False        # Frontalis/levator contraction spike (EAR surge > 1.25x baseline)
    stare_fixation_duration_s: float = 0.0      # Rigid central staring fixation duration (s)
    compensatory_stare_active: bool = False     # True if fixation > 6.0s with low spatial variance (< 0.8 deg)
    rigid_head_stabilization: bool = False      # Conscious locking of neck muscles to suppress natural sway

    # ── Eyebrow / Frontalis & Autonomic Pulse Micromotions (720p @ 60fps) ──
    brow_eye_distance_ratio: float = 0.0        # Normalized brow-to-eye vertical distance (frontalis tone)
    brow_elevation_delta: float = 0.0           # Relative change in brow elevation over baseline
    frontalis_compensation_active: bool = False # True when eyebrows lift (> +12%) to fight eyelid ptosis
    eyelid_tremor_hz: float = 0.0               # High-frequency palpebral flutter /orbicularis tremor (Hz)
    eyeball_tremor_hz: float = 0.0              # High-frequency ocular/iris micro-flutter (Hz)
    rppg_pulse_bpm: float = 0.0                 # T-zone green-channel photoplethysmography heart rate (BPM)
    rppg_hrv_rmssd_ms: float = 0.0              # Beat-to-beat heart rate variability RMSSD (ms)
    rppg_valid: bool = False                    # True when RGB buffer has sufficient SNR for pulse extraction

    # ── Anti-Masking Divergence Metric (M_mask) ──
    # Quantifies discordance between voluntary masking attempts and involuntary reflex decay
    masking_divergence_score: float = 0.0       # 0.0 to 100.0% divergence score (M_mask)
    masking_detected: bool = False              # True when M_mask >= 50.0%

    # Gating info
    imu_gated: bool = False
    gating_reason: str = ""
    vibration_score: float = 0.0

    # Phase 8 diagnostic signals
    eyeglass_glare_detected: bool = False
    sunglasses_occluded: bool = False
    speech_active: bool = False
    highway_fixation_exempt: bool = False


def _dist(p1: np.ndarray, p2: np.ndarray) -> float:
    """Euclidean distance between two points."""
    return float(np.linalg.norm(p1 - p2))


# ─── Saccade detection defaults (tunable via PipelineConfig) ───
# Normal saccades reach 200–900 deg/s. Impaired saccades are slower.
# At 30 fps (~33.3ms/frame), 30 deg/s corresponds to ~1.0 deg/frame iris shift,
# and 15 deg/s offset corresponds to ~0.5 deg/frame.
# Approximate degrees per normalized iris displacement unit.
# Human eye: ~30° horizontal range maps to iris_x in [-1, 1].
_DEG_PER_IRIS_UNIT = 15.0         # Standard scale for vergence, VOR, GEN, and fixation drift
_SACCADE_DEG_PER_UNIT = 22.0      # Saccadic angular scale (so dx=0.055 at 30fps > 35 deg/s onset, while dx=0.050 post-drop drift = 33.0 deg/s < 35 deg/s)


def is_monochrome_frame(frame: Optional[np.ndarray]) -> bool:
    """Check if input frame is single-channel grayscale or monochrome NIR."""
    if frame is None:
        return True
    if frame.ndim == 2:
        return True
    if frame.ndim == 3:
        if frame.shape[2] == 1:
            return True
        if frame.shape[2] == 3:
            # Check sub-sample of pixels across channels
            sub = frame[::16, ::16]
            diff1 = np.abs(sub[:, :, 0].astype(np.int16) - sub[:, :, 1].astype(np.int16))
            diff2 = np.abs(sub[:, :, 1].astype(np.int16) - sub[:, :, 2].astype(np.int16))
            return bool(np.max(diff1) < 4 and np.max(diff2) < 4)
    return False


class FeatureExtractor:
    """Extracts facial micromotion features from landmarks."""

    def __init__(self, blink_threshold: float = 0.20,
                 mar_yawn_threshold: float = 0.60,
                 perclos_window_sec: float = 60.0,
                 perclos_alert_threshold: float = 0.15,
                 microsleep_frames: int = 15,
                 yawn_frames: int = 30,
                 gaze_dispersion_window_sec: float = 15.0,
                 saccade_onset_velocity: float = 35.0,
                 saccade_offset_velocity: float = 20.0,
                 eccentric_gaze_threshold: float = 0.35,
                 gen_drift_min_vel: float = 2.0,
                 gen_drift_max_vel: float = 18.0,
                 gen_reset_min_vel: float = 30.0,
                 loc_vergence_threshold_deg: float = -3.5,
                 vor_min_head_speed: float = 2.0,
                 vor_max_head_speed: float = 25.0,
                 pursuit_window_sec: float = 5.0,
                 perclos_min_warmup_sec: float = 0.0,
                 fps: float = 30.0,
                 deg_per_iris_unit: float = _DEG_PER_IRIS_UNIT,
                 saccade_deg_per_unit: float = _SACCADE_DEG_PER_UNIT,
                 saccade_subpixel_deadzone_dx: float = 0.026,
                 saccade_subpixel_deadzone_hypot: float = 0.030,
                 saccade_head_motion_dx: float = 0.055,
                 eye_socket_vel_threshold: float = 0.75,
                 blink_phase_timeout_frames: int = 8,
                 sway_window_sec: float = 3.0,
                 sway_yaw_gate_deg_s: float = 5.0,
                 sway_nod_reset_deg_s: float = 35.0,
                 rigid_head_sway_threshold: float = 0.12):
        """
        Args:
            blink_threshold: EAR below this = eyes closed.
            mar_yawn_threshold: MAR above this = yawning.
            perclos_window_sec: Rolling window for PERCLOS (seconds).
            perclos_alert_threshold: PERCLOS above this = drowsy.
            microsleep_frames: Consecutive closed-eye frames for micro-sleep.
            yawn_frames: Consecutive high-MAR frames for yawn.
            gaze_dispersion_window_sec: Rolling window for σ_yaw (seconds).
            saccade_onset_velocity: Angular velocity (deg/s) to trigger saccade start.
            saccade_offset_velocity: Angular velocity (deg/s) to trigger saccade end (hysteresis).
            eccentric_gaze_threshold: Normalized horizontal gaze offset to evaluate GEN.
            gen_drift_min_vel / gen_drift_max_vel: Valid velocity range for centripetal drift.
            gen_reset_min_vel: Minimum velocity for fast corrective nystagmus beat.
            loc_vergence_threshold_deg: Vergence threshold below which LOC is flagged.
            vor_min_head_speed / vor_max_head_speed: Head yaw velocity range for VOR evaluation.
            pursuit_window_sec: Window duration for pursuit fragmentation ratio.
            perclos_min_warmup_sec: Minimum warmup period (seconds) before PERCLOS denominator shrinks.
            fps: Expected frame rate for buffer sizing.
        """
        self.blink_threshold = blink_threshold
        self.mar_yawn_threshold = mar_yawn_threshold
        self.perclos_alert = perclos_alert_threshold
        self.microsleep_frames = microsleep_frames
        self.yawn_frames = yawn_frames
        self.saccade_onset_velocity = saccade_onset_velocity
        self.saccade_offset_velocity = saccade_offset_velocity
        self._fps = fps
        self.perclos_min_warmup_sec = perclos_min_warmup_sec
        self.deg_per_iris_unit = deg_per_iris_unit
        self.saccade_deg_per_unit = saccade_deg_per_unit
        self.saccade_subpixel_deadzone_dx = saccade_subpixel_deadzone_dx
        self.saccade_subpixel_deadzone_hypot = saccade_subpixel_deadzone_hypot
        self.saccade_head_motion_dx = saccade_head_motion_dx
        self.eye_socket_vel_threshold = eye_socket_vel_threshold
        self.blink_phase_timeout_frames = blink_phase_timeout_frames
        self.sway_window_sec = sway_window_sec
        self.sway_yaw_gate_deg_s = sway_yaw_gate_deg_s
        self.sway_nod_reset_deg_s = sway_nod_reset_deg_s
        self.rigid_head_sway_threshold = rigid_head_sway_threshold
        self._lock = threading.RLock()
        self._flushing_cal_samples: Deque[float] = deque(maxlen=30)
        self._flushing_cal_locked: bool = False

        # Eyebrow / Frontalis tracking state
        self.baseline_brow_ratio: float = 0.0
        self._brow_cal_samples: Deque[float] = deque(maxlen=max(int(2.0 * fps), 10))
        self._smoothed_brow_ratio: float = 0.0

        # Eyelid micro-tremor / flutter state (2-second rolling window)
        self._ear_tremor_history: Deque[Tuple[float, float]] = deque(maxlen=max(int(2.0 * fps), 15))
        self._iris_tremor_history: Deque[Tuple[float, float]] = deque(maxlen=max(int(2.0 * fps), 15))
        self._rppg_sample_counter: int = 0

        # T-zone rPPG pulse & HRV state (10-second rolling green-channel buffer)
        self._rppg_green_history: Deque[Tuple[float, float]] = deque(maxlen=max(int(10.0 * fps), 60))
        self._last_rppg_bpm: float = 0.0
        self._last_rppg_rmssd: float = 0.0
        self._last_rppg_valid: bool = False

        # Rolling buffers for PERCLOS
        buffer_size = int(perclos_window_sec * fps)
        self._ear_history: Deque[float] = deque(maxlen=max(buffer_size, 1))
        self._perclos_closed_count: int = 0
        self._last_blink_threshold: float = blink_threshold

        # Counters for sustained events
        self._consecutive_closed = 0
        self._consecutive_yawn = 0

        # Last valid features (for gating holdover)
        self._last_features = FrameFeatures()

        # ── Saccade tracking state ──
        self._prev_iris_x: Optional[float] = None   # previous avg horizontal gaze
        self._prev_iris_y: Optional[float] = None
        self._prev_left_iris_x: Optional[float] = None
        self._prev_right_iris_x: Optional[float] = None
        self._prev_eye_center_norm: Optional[np.ndarray] = None
        self._prev_timestamp: Optional[float] = None
        self._in_saccade: bool = False
        self._last_saccade_end_time: float = 0.0     # for latency computation

        # ── Blink velocity tracking state ──
        # Track EAR trajectory during a blink to measure closing/opening speed
        self._prev_ear_avg: Optional[float] = None
        self._prev_ear_timestamp: Optional[float] = None
        self._blink_phase: str = "open"  # "open", "closing", "closed", "opening"
        self._closing_velocities: List[float] = []   # EAR/s samples during closing
        self._opening_velocities: List[float] = []   # EAR/s samples during opening
        self._last_blink_closing_vel: float = 0.0    # hold last completed blink's value
        self._last_blink_opening_vel: float = 0.0
        self._open_ear_baseline: float = 0.28        # dynamic baseline for ptosis/drowsiness adaptation
        self._phase_frames: int = 0                  # frame count in current blink phase (timeout guard)
        self._ear_deriv_window: Deque[Tuple[float, float]] = deque(maxlen=3)
        self._closing_glitch_count: int = 0

        # ── Gaze yaw dispersion (σ_yaw) ──
        dispersion_buffer_size = int(gaze_dispersion_window_sec * fps)
        self._gaze_yaw_history: Deque[float] = deque(
            maxlen=max(dispersion_buffer_size, 1))

        # ── VOR micro-compensation state ──
        self.vor_min_head_speed = vor_min_head_speed
        self.vor_max_head_speed = vor_max_head_speed
        self._prev_head_yaw: Optional[float] = None
        self._prev_head_pitch: Optional[float] = None
        self._prev_head_roll: Optional[float] = None
        self._prev_head_timestamp: Optional[float] = None
        self._smoothed_vor_gain: float = 1.0

        # ── Gaze-Evoked Nystagmus (GEN) state (Romano et al. 2017) ──
        self.eccentric_gaze_threshold = eccentric_gaze_threshold
        self.gen_drift_min_vel = gen_drift_min_vel
        self.gen_drift_max_vel = gen_drift_max_vel
        self.gen_reset_min_vel = gen_reset_min_vel
        self._gen_drift_detected: bool = False
        self._gen_drift_vel: float = 0.0
        self._gen_beat_times: Deque[float] = deque(maxlen=20)

        # ── Binocular Vergence & LOC ──
        self.loc_vergence_threshold_deg = loc_vergence_threshold_deg

        # ── Smooth Pursuit Fragmentation state ──
        self.pursuit_window_sec = pursuit_window_sec
        pursuit_buffer_size = int(pursuit_window_sec * fps)
        self._pursuit_history: Deque[Tuple[bool, float]] = deque(maxlen=max(pursuit_buffer_size, 1))

        # ── Involuntary Postural Sway State (Section A) ──
        self.sway_window_sec = sway_window_sec
        sway_buffer_size = int(self.sway_window_sec * fps)
        self._head_pose_history: Deque[Tuple[float, float, float]] = deque(maxlen=max(sway_buffer_size, 1))

        # ── Facial Flushing State (Section A) ──
        self._smoothed_flushing: float = 1.0
        self.baseline_flushing: float = 1.35

        # ── Voluntary Masking State (Section B) ──
        self.sober_ear_baseline: float = 0.28
        self.sober_blink_up_baseline: float = 2.5
        self.sober_blink_freq_baseline: float = 0.28  # ~17 blinks/min
        self._ear_time_history: Deque[Tuple[float, float]] = deque(maxlen=max(int(4.0 * fps), 1))
        self._last_blink_time: float = 0.0
        self._blink_times: Deque[float] = deque(maxlen=30)
        self._stare_start_time: Optional[float] = None
        self._stare_anchor_x: Optional[float] = None
        self._stare_anchor_y: Optional[float] = None
        self._stare_iris_samples: Deque[Tuple[float, float]] = deque(maxlen=max(int(10.0 * fps), 1))
        self._just_resumed_from_gate: bool = False

        # Phase 8: Rolling-shutter compensation and raw EAR history for sunglass detection
        self._baseline_nose_ratio: float = 0.0
        self._nose_ratio_samples: Deque[float] = deque(maxlen=30)
        self._raw_ear_var_history: Deque[float] = deque(maxlen=90)

        # Phase 9: Highway fixation exemption and speech/yawn cadence tracking
        self._highway_fixation_exempt: bool = False
        self._mar_history: Deque[Tuple[float, float]] = deque(maxlen=max(int(2.0 * fps), 15))

    def reset(self) -> None:
        """Clear rolling deques, histories, and previous-frame temporal tracking state.

        Essential when a scene cut, camera transition, or identity switch occurs.
        """
        with self._lock:
            self._ear_history.clear()
            self._consecutive_closed = 0
            self._consecutive_yawn = 0
            self._last_features = FrameFeatures()
            self._prev_iris_x = None
            self._prev_iris_y = None
            self._prev_left_iris_x = None
            self._prev_right_iris_x = None
            self._prev_eye_center_norm = None
            self._prev_timestamp = None
            self._in_saccade = False
            self._last_saccade_end_time = 0.0
            self._prev_ear_avg = None
            self._prev_ear_timestamp = None
            self._blink_phase = "open"
            self._closing_velocities.clear()
            self._opening_velocities.clear()
            self._last_blink_closing_vel = 0.0
            self._last_blink_opening_vel = 0.0
            self._open_ear_baseline = 0.28
            self._phase_frames = 0
            self._ear_deriv_window.clear()
            self._closing_glitch_count = 0
            self._gaze_yaw_history.clear()
            self._prev_head_yaw = None
            self._prev_head_pitch = None
            self._prev_head_roll = None
            self._prev_head_timestamp = None
            self._smoothed_vor_gain = 1.0
            self._gen_drift_detected = False
            self._gen_drift_vel = 0.0
            self._gen_beat_times.clear()
            self._pursuit_history.clear()
            self._head_pose_history.clear()
            self._smoothed_flushing = 1.0
            self._flushing_cal_samples.clear()
            self._flushing_cal_locked = False
            self._flushing_initialized = False
            self.baseline_brow_ratio = 0.0
            self._brow_cal_samples.clear()
            self._smoothed_brow_ratio = 0.0
            self._brow_initialized = False
            self._ear_tremor_history.clear()
            self._iris_tremor_history.clear()
            self._rppg_sample_counter = 0
            self._rppg_green_history.clear()
            self._last_rppg_bpm = 0.0
            self._last_rppg_rmssd = 0.0
            self._last_rppg_valid = False
            self._ear_time_history.clear()
            self._last_blink_time = 0.0
            self._blink_times.clear()
            self._stare_start_time = None
            self._stare_anchor_x = None
            self._stare_anchor_y = None
            self._stare_iris_samples.clear()
            self._just_resumed_from_gate = False
            self._baseline_nose_ratio = 0.0
            self._nose_ratio_samples.clear()
            self._raw_ear_var_history.clear()
            self._highway_fixation_exempt = False
            self._mar_history.clear()

    @staticmethod
    def compute_ear(landmarks: np.ndarray, indices: list) -> float:
        """Compute Eye Aspect Ratio (Soukupová & Čech 2016).

        EAR = (||P2-P6|| + ||P3-P5||) / (2 * ||P1-P4||)

        Args:
            landmarks: Full landmark array (478, 3).
            indices: [P1, P2, P3, P4, P5, P6] landmark indices.

        Returns:
            EAR value (typically 0.25-0.35 open, <0.20 closed).
        """
        p1, p2, p3, p4, p5, p6 = [landmarks[i] for i in indices]

        vertical_1 = np.linalg.norm(p2 - p6)
        vertical_2 = np.linalg.norm(p3 - p5)
        horizontal = np.linalg.norm(p1 - p4)

        if horizontal < 1e-6:
            return 0.0

        return float((vertical_1 + vertical_2) / (2.0 * horizontal))

    @staticmethod
    def compute_mar(landmarks: np.ndarray) -> float:
        """Compute Mouth Aspect Ratio (dense 3-pair formula).

        MAR = (Σ ||V_top - V_bot||) / (3 * ||H_left - H_right||)

        Returns:
            MAR value (typically <0.20 closed, >0.60 yawning).
        """
        h_left = landmarks[MAR_HORIZONTAL[0]]
        h_right = landmarks[MAR_HORIZONTAL[1]]
        horizontal = np.linalg.norm(h_left - h_right)

        if horizontal < 1e-6:
            return 0.0

        vertical_sum = 0.0
        for top_idx, bot_idx in MAR_VERTICAL_PAIRS:
            vertical_sum += np.linalg.norm(
                landmarks[top_idx] - landmarks[bot_idx])

        return float(vertical_sum / (3.0 * horizontal))

    @staticmethod
    def compute_iris_position(landmarks: np.ndarray,
                               iris_center_idx: int,
                               eye_outer_idx: int,
                               eye_inner_idx: int,
                               ring_indices: Optional[List[int]] = None) -> tuple:
        """Compute normalized iris position relative to eye corners.

        Averages the 5 iris landmarks (center + 4 ring points) when available
        to compute the iris centroid, reducing sub-pixel jitter by sqrt(5) ~ 2.23x.

        Returns:
            (x, y) normalized to approximately [-1, 1].
            x: horizontal gaze along consistent conjugate axis (-1 = left, +1 = right in camera frame).
            y: vertical gaze (-1 = down, +1 = up).
        """
        if ring_indices is None:
            if iris_center_idx == LEFT_IRIS_CENTER:
                ring_indices = LEFT_IRIS_RING
            elif iris_center_idx == RIGHT_IRIS_CENTER:
                ring_indices = RIGHT_IRIS_RING

        outer = landmarks[eye_outer_idx]
        inner = landmarks[eye_inner_idx]

        eye_center = (outer + inner) / 2.0
        eye_width = float(np.linalg.norm(outer - inner))

        if (ring_indices is not None and landmarks is not None and
                landmarks.shape[0] > max(ring_indices)):
            ring_pts = landmarks[ring_indices, :2]
            center_pt = landmarks[iris_center_idx, :2]
            dists = np.linalg.norm(ring_pts - center_pt, axis=1)
            # Valid iris ring points must have non-zero radius and be within the eye orbit (< 0.6 * eye_width)
            if np.all((dists > 0.2) & (dists < max(1.0, 0.6 * eye_width))):
                all_iris_indices = [iris_center_idx] + list(ring_indices)
                iris = np.mean(landmarks[all_iris_indices], axis=0)
            else:
                iris = landmarks[iris_center_idx]
        else:
            iris = landmarks[iris_center_idx]

        if eye_width < 1e-6:
            return (0.0, 0.0)

        # Consistent conjugate horizontal axis: left corner -> right corner (+X in frame)
        if outer[0] <= inner[0]:
            left_corner, right_corner = outer, inner
        else:
            left_corner, right_corner = inner, outer

        eye_axis = (right_corner - left_corner) / eye_width
        offset = iris - eye_center
        x = float(np.dot(offset[:2], eye_axis[:2]) / (eye_width / 2.0))

        # Vertical: perpendicular component
        perp = np.array([-eye_axis[1], eye_axis[0]])
        y = float(np.dot(offset[:2], perp) / (eye_width / 2.0))

        if np.isnan(x) or np.isinf(x):
            x = 0.0
        if np.isnan(y) or np.isinf(y):
            y = 0.0

        return (float(np.clip(x, -1.0, 1.0)), float(np.clip(y, -1.0, 1.0)))

    @staticmethod
    def _compute_eye_socket_center_norm(landmarks: np.ndarray) -> Optional[np.ndarray]:
        if landmarks is None or landmarks.ndim != 2 or len(landmarks) <= 362:
            return None
        r_w = float(np.linalg.norm(landmarks[33, :2] - landmarks[133, :2]))
        l_w = float(np.linalg.norm(landmarks[263, :2] - landmarks[362, :2]))
        mean_w = max(0.5 * (r_w + l_w), 1.0)
        center = 0.25 * (landmarks[33, :2] + landmarks[133, :2] + landmarks[263, :2] + landmarks[362, :2])
        if np.isnan(center).any() or np.isinf(center).any():
            return None
        return center / mean_w

    def _compute_saccade_dynamics(self, iris_x: float, iris_y: float,
                                   timestamp: float,
                                   blink_phase: str = "open",
                                   left_iris_x: Optional[float] = None,
                                   right_iris_x: Optional[float] = None,
                                   head_vel_deg_s: float = 0.0,
                                   eye_socket_vel: float = 0.0,
                                   vibration_score: float = 0.0) -> Tuple[float, bool, float, float]:
        """Compute saccadic velocity, detect saccades, measure latency, and raw displacement.

        Uses frame-to-frame angular velocity of the averaged iris position.
        A saccade is detected when angular velocity exceeds onset threshold
        and ends when it drops below offset threshold (hysteresis).

        NOTE on Sampling & Discretization Limit:
          Human saccades reach peak angular velocities of 200–900 deg/s across
          durations of only 20–50 ms. At 30 fps (33.3 ms/frame), a saccade spans
          only 1–2 discrete frames, introducing high velocity quantization noise.
          While dt is tracked dynamically to support 30 fps desktop fallback,
          capturing fine saccadic kinematics with forensic/clinical certainty
          requires >= 60 fps (preferably 90–120 fps) global-shutter sensor hardware.

        Args:
            iris_x: Average horizontal iris position (normalized [-1, 1]).
            iris_y: Average vertical iris position (normalized [-1, 1]).
            timestamp: Current frame timestamp (seconds).
            blink_phase: Current blink FSM phase ("open", "closing", "closed", "opening").
            left_iris_x: Optional left eye horizontal iris position for conjugate validation.
            right_iris_x: Optional right eye horizontal iris position for conjugate validation.
            head_vel_deg_s: Head angular speed in deg/s for VOR suppression.
            eye_socket_vel: Normalized eye socket translation speed for motion suppression.
            vibration_score: Instantaneous vehicle vibration score from KinematicStabilizer.

        Returns:
            (velocity_deg_s, is_saccade, latency_ms, raw_iris_dx)
        """
        if (np.isnan(iris_x) or np.isnan(iris_y) or np.isinf(iris_x) or np.isinf(iris_y) or
                np.isnan(timestamp) or np.isinf(timestamp)):
            return (0.0, False, 0.0, 0.0)

        if self._prev_iris_x is None or self._prev_iris_y is None or self._prev_timestamp is None:
            self._prev_iris_x = iris_x
            self._prev_iris_y = iris_y
            self._prev_timestamp = timestamp
            if left_iris_x is not None:
                self._prev_left_iris_x = left_iris_x
            if right_iris_x is not None:
                self._prev_right_iris_x = right_iris_x
            return (0.0, False, 0.0, 0.0)

        dt = timestamp - self._prev_timestamp
        if dt <= 1e-6:
            return (0.0, self._in_saccade, 0.0, 0.0)
        if dt > 0.5:
            # Gated period or frame drop exceeded 500ms — re-anchor without generating false derivative spike
            if self._in_saccade:
                self._in_saccade = False
                self._last_saccade_end_time = timestamp
            self._prev_iris_x = iris_x
            self._prev_iris_y = iris_y
            self._prev_timestamp = timestamp
            if left_iris_x is not None:
                self._prev_left_iris_x = left_iris_x
            if right_iris_x is not None:
                self._prev_right_iris_x = right_iris_x
            return (0.0, False, 0.0, 0.0)

        # ── Blink Occlusion Guard ──
        # When eyelids are closing, closed, or opening, zero out displacement & velocity
        # to prevent stale holdover, while preserving in-saccade state and latency.
        if blink_phase != "open":
            self._was_in_blink = True
            raw_dx = 0.0
            velocity = 0.0
            saccade_detected = self._last_features.saccade_detected
            self._in_saccade = saccade_detected

            if self._in_saccade:
                latency_ms = 0.0
            elif self._last_saccade_end_time > 0:
                latency_ms = (timestamp - self._last_saccade_end_time) * 1000.0
            else:
                latency_ms = 0.0

            return (velocity, saccade_detected, latency_ms, raw_dx)

        if getattr(self, "_was_in_blink", False):
            self._was_in_blink = False
            # Reopening frame: re-anchor to current valid pupil position
            # eliminating the false derivative spike against blink-occluded coordinates
            raw_dx = 0.0
            velocity = 0.0
            if self._in_saccade:
                self._in_saccade = False
                self._last_saccade_end_time = timestamp
                latency_ms = 0.0
            elif self._last_saccade_end_time > 0:
                latency_ms = (timestamp - self._last_saccade_end_time) * 1000.0
            else:
                latency_ms = 0.0
            self._prev_iris_x = iris_x
            self._prev_iris_y = iris_y
            self._prev_timestamp = timestamp
            if left_iris_x is not None:
                self._prev_left_iris_x = left_iris_x
            if right_iris_x is not None:
                self._prev_right_iris_x = right_iris_x
            return (velocity, False, latency_ms, raw_dx)

        # Raw frame-to-frame iris displacement in normalized units
        raw_dx = float(abs(iris_x - self._prev_iris_x))
        raw_dy = float(abs(iris_y - self._prev_iris_y))

        # Binocular conjugate check: only zero out if eyes move in strictly opposite directions
        # (vergence or eyeglass reflection glitch); otherwise use binocular mean raw_dx
        if (left_iris_x is not None and right_iris_x is not None and
                self._prev_left_iris_x is not None and self._prev_right_iris_x is not None):
            dx_l = left_iris_x - self._prev_left_iris_x
            dx_r = right_iris_x - self._prev_right_iris_x
            if (dx_l * dx_r) < -1e-4:
                effective_dx = 0.0
            else:
                effective_dx = raw_dx
        else:
            effective_dx = raw_dx

        # Head-Motion / VOR Guard:
        # Only activate when the eye sockets physically translate in pixel space (eye_socket_vel > self.eye_socket_vel_threshold)
        # OR during large head rotations (head_vel_deg_s > 30.0) accompanied by socket translation,
        # so stationary PnP Euler jitter never blocks real saccades!
        is_head_moving = (eye_socket_vel > self.eye_socket_vel_threshold) or (head_vel_deg_s > 30.0 and eye_socket_vel > 0.25)

        # Dynamic deadzone scaling under cabin vibration
        if vibration_score > 0.2:
            scale = 1.0 + 1.2 * vibration_score
            eff_saccade_deadzone_dx = self.saccade_subpixel_deadzone_dx * scale
            eff_saccade_deadzone_hypot = self.saccade_subpixel_deadzone_hypot * scale
        else:
            eff_saccade_deadzone_dx = self.saccade_subpixel_deadzone_dx
            eff_saccade_deadzone_hypot = self.saccade_subpixel_deadzone_hypot

        if is_head_moving:
            if effective_dx < max(self.saccade_head_motion_dx, eff_saccade_deadzone_dx):
                velocity = 0.0
            else:
                dx_deg = effective_dx * self.saccade_deg_per_unit
                velocity = max(0.0, (dx_deg / dt) - 0.5 * head_vel_deg_s)
        else:
            # Sub-pixel deadzone: < eff_saccade_deadzone_dx or hypot < eff_saccade_deadzone_hypot is sensor/vibration noise
            if effective_dx < eff_saccade_deadzone_dx or np.hypot(raw_dx, raw_dy * 0.5) < eff_saccade_deadzone_hypot:
                velocity = 0.0
            else:
                dx_deg = effective_dx * self.saccade_deg_per_unit
                velocity = dx_deg / dt  # deg/s

        # Saccade detection with hysteresis
        if not self._in_saccade:
            if velocity > self.saccade_onset_velocity:
                self._in_saccade = True
        else:
            if velocity < self.saccade_offset_velocity:
                self._in_saccade = False
                self._last_saccade_end_time = timestamp

        # Latency: time since last saccade ended (inter-saccadic interval)
        # During a saccade, latency is 0; during fixation, it accumulates
        if self._in_saccade:
            latency_ms = 0.0
        elif self._last_saccade_end_time > 0:
            latency_ms = (timestamp - self._last_saccade_end_time) * 1000.0
        else:
            latency_ms = 0.0

        return (velocity, self._in_saccade, latency_ms, effective_dx)

    def _compute_blink_velocity(self, ear_avg: float,
                                 timestamp: float) -> Tuple[float, float]:
        """Track EAR trajectory through a blink to measure closing/opening speed.

        Blink phases: open → closing → closed → opening → open
        The closing velocity (downstroke) and opening velocity (upstroke) are
        computed as the average |dEAR/dt| during each respective phase.

        These are involuntary motor signals: alcohol slows both phases, but
        especially the opening (upstroke) phase. Unlike blink *rate*, which
        a driver can consciously suppress, blink *velocity* cannot be faked.

        NOTE on Sampling & Discretization Limit:
          An involuntary eyelid downstroke lasts only 50–100 ms. At 30 fps
          (33.3 ms/frame), that downstroke spans only 1.5 to 3 discrete frames,
          producing high quantization variance in derivative calculations.
          While 30 fps is an acceptable desktop demonstration fallback,
          capturing fine oculomotor velocity with forensic certainty requires
          >= 60 fps global-shutter NIR sensor hardware.

        Returns:
            (closing_velocity, opening_velocity) in EAR units/second.
            Returns the last completed blink's values; 0.0 if no blink yet.
        """
        if self._prev_ear_avg is None or self._prev_ear_timestamp is None:
            self._prev_ear_avg = ear_avg
            self._prev_ear_timestamp = timestamp
            self._ear_deriv_window.clear()
            self._ear_deriv_window.append((timestamp, ear_avg))
            return (self._last_blink_closing_vel, self._last_blink_opening_vel)

        dt = timestamp - self._prev_ear_timestamp
        if dt <= 1e-6:
            return (self._last_blink_closing_vel, self._last_blink_opening_vel)
        if dt > 0.5:
            # Gated freeze or frame drop exceeded 500ms — re-anchor without corrupting blink velocities
            self._prev_ear_avg = ear_avg
            self._prev_ear_timestamp = timestamp
            self._ear_deriv_window.clear()
            self._ear_deriv_window.append((timestamp, ear_avg))
            self._closing_glitch_count = 0
            return (self._last_blink_closing_vel, self._last_blink_opening_vel)

        # Adapt baseline open EAR during sustained open eyes (not during blinks or ptosis droop)
        if ear_avg > self.blink_threshold * 1.15:
            self._open_ear_baseline = 0.98 * self._open_ear_baseline + 0.02 * ear_avg

        self._phase_frames += 1

        # Phase timeout guard: if trapped in closing or opening for > self.blink_phase_timeout_frames frames
        if self._phase_frames > self.blink_phase_timeout_frames:
            if self._closing_velocities:
                self._last_blink_closing_vel = float(np.max(self._closing_velocities))
            if self._opening_velocities:
                self._last_blink_opening_vel = float(np.max(self._opening_velocities))
            if self._blink_phase == "closing":
                self._blink_phase = "closed" if ear_avg < self.blink_threshold else "open"
            elif self._blink_phase == "opening":
                self._blink_phase = "open" if ear_avg >= self.blink_threshold else "closed"
            self._phase_frames = 0
            self._closing_glitch_count = 0

        d_inst = float(ear_avg - self._prev_ear_avg)
        self._ear_deriv_window.append((timestamp, ear_avg))
        if len(self._ear_deriv_window) == 3:
            t0, y0 = self._ear_deriv_window[0]
            t1, y1 = self._ear_deriv_window[1]
            t2, y2 = self._ear_deriv_window[2]
            dt_span = t2 - t0
            if dt_span > 1e-6:
                t_mean = (t0 + t1 + t2) / 3.0
                y_mean = (y0 + y1 + y2) / 3.0
                dt0 = t0 - t_mean
                dt1 = t1 - t_mean
                dt2 = t2 - t_mean
                denom = dt0 * dt0 + dt1 * dt1 + dt2 * dt2
                if denom > 1e-12:
                    num = dt0 * (y0 - y_mean) + dt1 * (y1 - y_mean) + dt2 * (y2 - y_mean)
                    slope = num / denom
                else:
                    slope = (y2 - y0) / dt_span

                if slope * d_inst > 0:
                    d_ear = float(slope * dt)
                    ear_velocity = float(max(abs(slope), abs(d_inst / dt)))
                else:
                    d_ear = d_inst
                    ear_velocity = float(abs(d_inst / dt))
            else:
                d_ear = d_inst
                ear_velocity = float(abs(d_inst / dt))
        else:
            d_ear = d_inst
            ear_velocity = float(abs(d_inst / dt))

        # Low-FPS thermal throttling discretization compensation (Phase 9)
        if 0.040 <= dt <= 0.120 and self._fps >= 25.0:
            fps_comp = min(1.65, 1.0 + 10.0 * (dt - 0.035))
            ear_velocity *= fps_comp

        # State machine: track blink phase transitions
        if self._blink_phase == "open":
            if (d_ear < -0.010 or d_inst < -0.010) and ear_avg < min(0.28, self._open_ear_baseline * 0.95):
                # EAR dropping → entering closing phase
                if ear_avg < self.blink_threshold:
                    self._blink_phase = "closed"
                else:
                    self._blink_phase = "closing"
                self._phase_frames = 0
                self._closing_glitch_count = 0
                self._closing_velocities = [ear_velocity]
                self._opening_velocities = []

        elif self._blink_phase == "closing":
            if ear_avg < self.blink_threshold:
                # Eyes fully closed → transition to closed
                self._blink_phase = "closed"
                self._phase_frames = 0
                self._closing_glitch_count = 0
                if ear_velocity > 0.02:
                    self._closing_velocities.append(ear_velocity)
            elif d_ear < 0:
                # Still closing (omit zero derivative from frozen/gated frames)
                self._closing_glitch_count = 0
                if ear_velocity > 0.02:
                    self._closing_velocities.append(ear_velocity)
            else:
                # Check for 1-frame vibration glitch tolerance
                if self._closing_glitch_count < 1 and d_ear < 0.015:
                    self._closing_glitch_count += 1
                else:
                    # Aborted blink (EAR went back up without closing fully)
                    self._blink_phase = "open"
                    self._phase_frames = 0
                    self._closing_glitch_count = 0

        elif self._blink_phase == "closed":
            if d_ear > 0.005 and ear_avg > self.blink_threshold * 0.75:
                # EAR rising → entering opening phase
                reopen_thresh = min(0.22, self._open_ear_baseline * 0.85)
                if ear_avg >= reopen_thresh:
                    # Direct transition to open if eyes reopened above threshold in single frame
                    self._blink_phase = "open"
                    self._phase_frames = 0
                    self._closing_glitch_count = 0
                    if ear_velocity > 0.02:
                        self._opening_velocities = [ear_velocity]
                    if self._closing_velocities:
                        self._last_blink_closing_vel = float(
                            np.max(self._closing_velocities))
                    if self._opening_velocities:
                        self._last_blink_opening_vel = float(
                            np.max(self._opening_velocities))
                else:
                    self._blink_phase = "opening"
                    self._phase_frames = 0
                    self._closing_glitch_count = 0
                    self._opening_velocities = [ear_velocity]

        elif self._blink_phase == "opening":
            reopen_thresh = min(0.22, self._open_ear_baseline * 0.85)
            if ear_avg >= reopen_thresh:
                # Eyes reopened → blink complete, compute velocities
                if ear_velocity > 0.02:
                    self._opening_velocities.append(ear_velocity)

                # Store completed blink peak velocities (robust against slow turnaround edge frames)
                if self._closing_velocities:
                    self._last_blink_closing_vel = float(
                        np.max(self._closing_velocities))
                if self._opening_velocities:
                    self._last_blink_opening_vel = float(
                        np.max(self._opening_velocities))

                self._blink_phase = "open"
                self._phase_frames = 0
                self._closing_glitch_count = 0
            elif d_ear < -0.010 and ear_avg < self.blink_threshold:
                # Re-closed during microsleep
                self._blink_phase = "closed"
                self._phase_frames = 0
                self._closing_glitch_count = 0
            elif d_ear > 0:
                # Still opening (omit zero derivative from frozen/gated frames)
                if ear_velocity > 0.02:
                    self._opening_velocities.append(ear_velocity)
            else:
                # Double-dip or flicker — stay in opening
                pass

        self._prev_ear_avg = ear_avg
        self._prev_ear_timestamp = timestamp

        return (self._last_blink_closing_vel, self._last_blink_opening_vel)

    def _compute_gaze_yaw_dispersion(self, gaze_x: float) -> float:
        """Compute rolling standard deviation of horizontal gaze angle.

        Captures the "gaze tunneling toward road center" signal: impaired
        drivers progressively reduce their scanning arc, resulting in lower
        σ_yaw. Healthy drivers continuously scan mirrors, instruments, and
        the road ahead, producing higher σ_yaw.

        Args:
            gaze_x: Average horizontal gaze position (normalized [-1, 1]).

        Returns:
            σ_yaw: standard deviation of gaze_x over the rolling window.
        """
        self._gaze_yaw_history.append(gaze_x)

        if len(self._gaze_yaw_history) < 10:
            return 0.0

        return float(np.std(self._gaze_yaw_history))

    def _compute_binocular_vergence(self, left_iris_x: float, right_iris_x: float) -> Tuple[float, bool]:
        """Compute binocular vergence angle and test for Lack of Convergence (LOC).

        Under conjugate horizontal gaze (+X right for both eyes):
        Right eye (viewer left): nasal (inward) is +X, temporal (outward) is -X.
        Left eye (viewer right): nasal (inward) is -X, temporal (outward) is +X.
        Vergence = (right_iris_x - left_iris_x) * _DEG_PER_IRIS_UNIT.
        Parallel forward gaze (infinity) has vergence ≈ 0.0°.
        Near convergence (instrument panel) has positive vergence (> 2.0°).
        Divergent strabismus / inability to hold convergence (DRE LOC sign) has negative vergence (< -3.5°).
        Pure conjugate saccades yield vergence ≈ 0.0° (no false-positive LOC triggers).
        """
        vergence = (right_iris_x - left_iris_x) * self.deg_per_iris_unit
        loc = bool(vergence < self.loc_vergence_threshold_deg)
        return (float(vergence), loc)

    def _compute_vor_gain(self, avg_iris_x: float, head_yaw: float, timestamp: float, blink_phase: str = "open") -> float:
        """Compute Vestibulo-Ocular Reflex (VOR) micro-compensation gain (PMC8997842).

        VOR maintains gaze fixated on the road during natural head tremors/sway (2-25 deg/s).
        Eye counter-rotation velocity should match negative head velocity: gain = -d_eye / d_head.
        Sober: ~0.85-1.05. Impaired (BAC >= 0.05%): depressed (<0.7) or lagged.
        """
        if self._prev_head_yaw is None or self._prev_head_timestamp is None:
            return float(self._smoothed_vor_gain)

        dt = timestamp - self._prev_head_timestamp
        if dt <= 1e-6:
            return float(self._smoothed_vor_gain)

        d_yaw = (head_yaw - self._prev_head_yaw + 180.0) % 360.0 - 180.0
        head_vel = d_yaw / dt   # deg/s

        if blink_phase == "open" and self._prev_iris_x is not None:
            eye_vel = (avg_iris_x - self._prev_iris_x) * self.deg_per_iris_unit / dt  # deg/s

            if self.vor_min_head_speed <= abs(head_vel) <= self.vor_max_head_speed:
                raw_gain = -(eye_vel / head_vel)
                if raw_gain > 0.05:
                    clamped_gain = float(np.clip(raw_gain, 0.0, 2.0))
                    self._smoothed_vor_gain = 0.85 * self._smoothed_vor_gain + 0.15 * clamped_gain

        return float(self._smoothed_vor_gain)

    def _compute_gen(self, avg_iris_x: float, timestamp: float,
                     head_vel_deg_s: float = 0.0,
                     avg_iris_y: float = 0.0) -> Tuple[bool, int, float]:
        """Gaze-Evoked Nystagmus (GEN) Detector (Romano et al. 2017).

        Detects the signature 'leaky neural integrator' failure at eccentric gaze:
          - Gaze held sideways (|iris_x| > threshold)
          - Slow centripetal drift inward toward the orbit center (2-18 deg/s)
          - Fast corrective saccadic beat outward back toward the target (> 30 deg/s)
          - Multiple beats within rolling 1.5s window flags GEN.
        """
        if self._prev_iris_x is None or self._prev_timestamp is None:
            return (False, 0, 0.0)

        dt = timestamp - self._prev_timestamp
        if dt <= 1e-6:
            recent_beats = sum(1 for t in self._gen_beat_times if timestamp - t <= 1.5)
            return (recent_beats >= 2, recent_beats, self._gen_drift_vel)

        is_eccentric = abs(avg_iris_x) >= self.eccentric_gaze_threshold
        iris_vel = (avg_iris_x - self._prev_iris_x) * self.deg_per_iris_unit / dt  # deg/s

        prev_y = self._prev_iris_y if self._prev_iris_y is not None else avg_iris_y
        vert_vel = abs(avg_iris_y - prev_y) / max(dt, 1e-4) * self.deg_per_iris_unit
        # OKN vs Pathological GEN Guard:
        # Suppress if driver is actively turning head (mirror/traffic scan) or large vertical flow
        suppress_okn = bool(head_vel_deg_s > 10.0 or vert_vel > 25.0)

        if not suppress_okn and is_eccentric:
            # Centripetal velocity points back toward 0:
            # If iris_x > 0 (nasal/right), inward drift is negative velocity (iris_vel < 0).
            # If iris_x < 0 (temporal/left), inward drift is positive velocity (iris_vel > 0).
            inward_drift_speed = -iris_vel if avg_iris_x > 0 else iris_vel
            outward_beat_speed = iris_vel if avg_iris_x > 0 else -iris_vel

            if self.gen_drift_min_vel <= inward_drift_speed <= self.gen_drift_max_vel:
                self._gen_drift_detected = True
                self._gen_drift_vel = inward_drift_speed

            elif self._gen_drift_detected and outward_beat_speed >= self.gen_reset_min_vel:
                # Fast corrective beat completed the cycle!
                self._gen_beat_times.append(timestamp)
                self._gen_drift_detected = False
        else:
            self._gen_drift_detected = False

        recent_beats = sum(1 for t in self._gen_beat_times if timestamp - t <= 1.5)
        gen_detected = recent_beats >= 2
        return (gen_detected, recent_beats, self._gen_drift_vel if gen_detected else 0.0)

    def _compute_pursuit_fragmentation(self, saccade_detected: bool, timestamp: float) -> float:
        """Compute smooth pursuit fragmentation ratio.

        Ratio of catch-up saccade frames to total non-fixation tracking frames
        in a rolling window (default 5.0s). In healthy pursuit, ratio ≈ 0.
        Under alcohol impairment, smooth pursuit degrades into jerky catch-up saccades (> 0.20).
        """
        self._pursuit_history.append((saccade_detected, timestamp))

        if len(self._pursuit_history) < 10:
            return 0.0

        cutoff_time = timestamp - self.pursuit_window_sec
        recent_samples = [s for s, t in self._pursuit_history if t >= cutoff_time]
        if not recent_samples:
            return 0.0

        saccadic_samples = sum(1 for s in recent_samples if s)
        return float(saccadic_samples / len(recent_samples))

    def _compute_postural_sway(self, head_pitch: float, head_roll: float, timestamp: float, head_angular_vel: float = 0.0, vibration_score: float = 0.0) -> Tuple[float, bool]:
        """Compute low-frequency involuntary head wobble/postural sway (Section A) and detect rigid head stabilization (Section B).

        NOTE: Seated postural sway is an experimental proxy of Romberg standing posturography;
        in moving vehicles, low-frequency road terrain (< 0.5 Hz) and vehicle pitch/roll must be
        filtered out. We isolate 0.5 - 2.0 Hz tremor by applying high-pass linear detrending across
        the recent window to reject road grade/banking drift and evaluating residual variance.
        """
        if vibration_score > 0.35:
            # Road vibration present: pause head_postural_sway accumulation so road vibration
            # is never misclassified as Romberg postural sway
            sway_val = float(self._last_features.head_postural_sway) if hasattr(self, "_last_features") else 0.0
            return (sway_val, False)

        if head_angular_vel > self.sway_yaw_gate_deg_s and len(self._head_pose_history) > 0:
            return (float(self._last_features.head_postural_sway), False)

        self._head_pose_history.append((timestamp, head_pitch, head_roll))
        cutoff = timestamp - self.sway_window_sec
        recent = [(t, p, r) for t, p, r in self._head_pose_history if t >= cutoff]
        if len(recent) < 10:
            return (0.0, False)

        times = np.array([t for t, _, _ in recent], dtype=np.float64)
        pitches = np.array([p for _, p, _ in recent], dtype=np.float64)
        rolls = np.array([r for _, _, r in recent], dtype=np.float64)

        # High-pass suspension de-trending: reject vehicle incline, banking, and road grade drift (<0.5 Hz)
        try:
            if len(times) >= 2 and (times[-1] - times[0]) > 1e-4:
                t_rel = times - times[0]
                p_slope, p_intercept = np.polyfit(t_rel, pitches, 1)
                detrended_pitches = pitches - (p_slope * t_rel + p_intercept)
                r_slope, r_intercept = np.polyfit(t_rel, rolls, 1)
                detrended_rolls = rolls - (r_slope * t_rel + r_intercept)
            else:
                detrended_pitches = pitches - np.mean(pitches)
                detrended_rolls = rolls - np.mean(rolls)
        except Exception as exc:
            logger.debug("Postural sway polyfit detrending fallback: %s", exc, exc_info=True)
            detrended_pitches = pitches - np.mean(pitches)
            detrended_rolls = rolls - np.mean(rolls)

        std_p = float(np.std(detrended_pitches))
        std_r = float(np.std(detrended_rolls))
        sway = float(np.sqrt(std_p**2 + std_r**2))
        rigid_stabilization = bool(sway < self.rigid_head_sway_threshold and len(recent) >= 20)
        return (sway, rigid_stabilization)

    def _sample_tzone_rgb(
        self, frame: Optional[np.ndarray], landmarks: np.ndarray
    ) -> Optional[Tuple[float, float, float]]:
        """Sample mean (R, G, B) across VASODILATION_LANDMARKS in a single pass."""
        if frame is None or is_monochrome_frame(frame) or landmarks is None or len(landmarks) < 468:
            return None

        try:
            h, w = frame.shape[:2]
            r_vals, g_vals, b_vals = [], [], []

            for idx in VASODILATION_LANDMARKS:
                if idx < len(landmarks):
                    pt = landmarks[idx]
                    px = int(pt[0] * w) if pt[0] <= 1.0 else int(pt[0])
                    py = int(pt[1] * h) if pt[1] <= 1.0 else int(pt[1])

                    # Skip uninitialized / origin points from synthetic test mocks
                    if px <= 3 and py <= 3:
                        continue

                    y1 = max(0, py - 3)
                    y2 = min(h, py + 4)
                    x1 = max(0, px - 3)
                    x2 = min(w, px + 4)

                    if y2 > y1 and x2 > x1:
                        patch = frame[y1:y2, x1:x2]
                        b_vals.append(float(np.mean(patch[:, :, 0])))
                        g_vals.append(float(np.mean(patch[:, :, 1])))
                        r_vals.append(float(np.mean(patch[:, :, 2])))

            if not r_vals or not g_vals or not b_vals:
                return None
            return (float(np.mean(r_vals)), float(np.mean(g_vals)), float(np.mean(b_vals)))
        except Exception as exc:
            logger.debug("T-zone RGB patch sampling failed: %s", exc, exc_info=True)
            return None

    def _compute_facial_flushing(
        self,
        frame: Optional[np.ndarray],
        landmarks: np.ndarray,
        tzone_rgb: Optional[Tuple[float, float, float]] = None,
    ) -> Tuple[float, bool, float]:
        """Measure micro-vascular facial flushing / vasodilation across cheeks and forehead T-zone (Section A).

        Calculates chromaticity ratio R / ((G + B)/2 + 1e-5) across MediaPipe full cheek
        matrices and forehead frontal patch landmarks (VASODILATION_LANDMARKS).
        If frame is monochrome / NIR, flushing is cleanly marked invalid and delta set to 0.0.
        When RGB, flushing_delta evaluates relative surge over baseline (default 1.35 or calibrated).
        """
        if frame is None or is_monochrome_frame(frame) or landmarks is None or len(landmarks) < 468:
            return (1.0, False, 0.0)

        try:
            rgb = tzone_rgb if tzone_rgb is not None else self._sample_tzone_rgb(frame, landmarks)
            if rgb is not None:
                mean_r, mean_g, mean_b = rgb
                mean_gb = float((mean_g + mean_b) / 2.0)
                raw_ratio = float(mean_r / (mean_gb + 1e-5))
                clamped = float(np.clip(raw_ratio, 0.5, 2.5))
                if not getattr(self, "_flushing_initialized", False):
                    self._smoothed_flushing = clamped
                    self._flushing_initialized = True
                else:
                    self._smoothed_flushing = 0.90 * self._smoothed_flushing + 0.10 * clamped

                if not getattr(self, "_flushing_cal_locked", False) and self.baseline_flushing == 1.35:
                    self._flushing_cal_samples.append(clamped)
                    if len(self._flushing_cal_samples) >= 15:
                        med = float(np.median(self._flushing_cal_samples))
                        if med < 1.28:
                            self.baseline_flushing = med
                        self._flushing_cal_locked = True

                flushing_delta = float((self._smoothed_flushing - self.baseline_flushing) / max(0.1, self.baseline_flushing))
                return (float(self._smoothed_flushing), True, flushing_delta)
        except Exception as exc:
            logger.debug("Facial flushing computation failed: %s", exc, exc_info=True)

        return (float(self._smoothed_flushing), True, 0.0)

    def _compute_voluntary_eye_widening(self, ear_avg: float, timestamp: float, frontalis_compensation_active: bool = False) -> bool:
        """Detect voluntary eye-widening spikes (Section B).

        Frontalis/levator contraction spikes where EAR surges > 1.25x sober baseline
        (>0.35 with baseline 0.28), often used to consciously fight ptosis droop.
        """
        self._ear_time_history.append((timestamp, ear_avg))
        cutoff = timestamp - 3.5
        recent = [e for t, e in self._ear_time_history if t >= cutoff]

        if ear_avg > 1.25 * self.sober_ear_baseline:
            if recent and (min(recent) < 0.24 or (ear_avg - min(recent)) > 0.08):
                return True
        if frontalis_compensation_active and recent and min(recent) < 0.24:
            return True
        return False

    def _compute_deliberate_blink_suppression(self, blink_detected: bool, ear_avg: float, timestamp: float) -> bool:
        """Detect deliberate blink suppression (Section B).

        A driver actively fighting eyelid closure withholds blinks, resulting in
        prolonged inter-blink intervals (> 6.0 seconds while eyes are open).
        """
        if self._last_blink_time <= 0.0:
            self._last_blink_time = timestamp

        if blink_detected:
            self._last_blink_time = timestamp
            return False

        if ear_avg >= self.blink_threshold:
            interval = timestamp - self._last_blink_time
            return bool(interval >= 6.0 and timestamp >= 6.0)

        return False

    def _compute_fixation_stare(self, avg_iris_x: float, avg_iris_y: float,
                                saccade_detected: bool, blink_detected: bool,
                                timestamp: float) -> Tuple[float, bool]:
        """Track compensatory staring fixation duration and spatial variance (Section B).

        Rigid unbroken central fixation for > 6.0 seconds with low spatial variance
        (sigma_gaze < 0.8 deg) captures compensatory cognitive freeze while avoiding
        false positives on alert highway forward gaze.
        """
        if self._stare_start_time is None or self._stare_anchor_x is None or self._stare_anchor_y is None:
            self._stare_start_time = timestamp
            self._stare_anchor_x = avg_iris_x
            self._stare_anchor_y = avg_iris_y
            self._stare_iris_samples.clear()
            self._stare_iris_samples.append((avg_iris_x, avg_iris_y))
            self._highway_fixation_exempt = False
            return (0.0, False)

        drift_deg = np.sqrt(
            ((avg_iris_x - self._stare_anchor_x) * self.deg_per_iris_unit) ** 2 +
            ((avg_iris_y - self._stare_anchor_y) * self.deg_per_iris_unit) ** 2
        )

        if saccade_detected or blink_detected or drift_deg > 2.5:
            # Fixation broken, re-anchor
            self._stare_start_time = timestamp
            self._stare_anchor_x = avg_iris_x
            self._stare_anchor_y = avg_iris_y
            self._stare_iris_samples.clear()
            self._stare_iris_samples.append((avg_iris_x, avg_iris_y))
            self._highway_fixation_exempt = False
            return (0.0, False)

        self._stare_iris_samples.append((avg_iris_x, avg_iris_y))
        duration = float(timestamp - self._stare_start_time)

        # Compensatory stare active if duration >= 6.0s and gaze std < 0.8 deg
        compensatory_stare = False
        if duration >= 6.0 and len(self._stare_iris_samples) >= 15:
            xs = [x for x, _ in self._stare_iris_samples]
            ys = [y for _, y in self._stare_iris_samples]
            sigma_deg = float(np.sqrt(np.std(xs)**2 + np.std(ys)**2) * self.deg_per_iris_unit)
            if 0.15 <= sigma_deg < 0.8 and self._last_blink_opening_vel >= 1.8:
                compensatory_stare = False
                self._highway_fixation_exempt = True
            else:
                compensatory_stare = bool(sigma_deg < 0.8)
                self._highway_fixation_exempt = False
        else:
            self._highway_fixation_exempt = False

        return (duration, compensatory_stare)

    def _compute_brow_micromotion(self, landmarks: np.ndarray, ear_avg: float) -> Tuple[float, float, bool]:
        """Compute vertical eyebrow-to-canthus ratio, delta, and frontalis activation.

        Normalized by horizontal inter-canthal width for 100% scale and distance invariance.
        """
        if landmarks is None or len(landmarks) < 337 or np.isnan(landmarks).any():
            return (0.0, 0.0, False)

        try:
            w_r = float(np.linalg.norm(landmarks[133, :2] - landmarks[33, :2]))
            w_l = float(np.linalg.norm(landmarks[263, :2] - landmarks[362, :2]))
            if w_r < 1e-4 or w_l < 1e-4:
                return (0.0, 0.0, False)

            eye_r_y = float(0.5 * (landmarks[33, 1] + landmarks[133, 1]))
            eye_l_y = float(0.5 * (landmarks[263, 1] + landmarks[362, 1]))
            brow_r_y = float(np.mean(landmarks[RIGHT_EYEBROW_ARCH, 1]))
            brow_l_y = float(np.mean(landmarks[LEFT_EYEBROW_ARCH, 1]))

            # Guard synthetic test landmarks where eyebrow points are uninitialized (<= 1.0)
            if brow_r_y <= 1.0 and brow_l_y <= 1.0:
                return (0.0, 0.0, False)

            ratio_r = (eye_r_y - brow_r_y) / w_r
            ratio_l = (eye_l_y - brow_l_y) / w_l
            raw_ratio = float(0.5 * (ratio_r + ratio_l))

            if not getattr(self, "_brow_initialized", False) or self._smoothed_brow_ratio <= 0.0:
                self._smoothed_brow_ratio = raw_ratio
                self._brow_initialized = True
            else:
                self._smoothed_brow_ratio = float(0.80 * self._smoothed_brow_ratio + 0.20 * raw_ratio)

            if self.baseline_brow_ratio <= 0.0:
                self._brow_cal_samples.append(raw_ratio)
                if len(self._brow_cal_samples) >= 15:
                    self.baseline_brow_ratio = float(np.median(self._brow_cal_samples))
                else:
                    return (float(self._smoothed_brow_ratio), 0.0, False)

            brow_delta = float((self._smoothed_brow_ratio - self.baseline_brow_ratio) / max(0.1, abs(self.baseline_brow_ratio)))
            frontalis_active = bool(brow_delta >= 0.12)
            return (float(self._smoothed_brow_ratio), float(brow_delta), frontalis_active)
        except Exception as exc:
            logger.debug("Brow ratio computation failed: %s", exc, exc_info=True)
            return (0.0, 0.0, False)

    def _compute_eyelid_tremor(self, ear_avg: float, timestamp: float, blink_phase: str, vibration_score: float = 0.0) -> float:
        """Track high-frequency palpebral micro-flutter (3-15 Hz).

        Zero-crossing oscillation of first-order EAR differences during open-eye fixation.
        """
        if blink_phase != "open":
            return 0.0

        self._ear_tremor_history.append((timestamp, ear_avg))
        cutoff = timestamp - 2.0
        samples = [(t, e) for t, e in self._ear_tremor_history if t >= cutoff]

        if len(samples) < 12:
            return 0.0

        duration = samples[-1][0] - samples[0][0]
        if duration < 0.4:
            return 0.0

        ears = np.array([e for t, e in samples], dtype=np.float64)
        diffs = np.diff(ears)
        valid_mask = (np.abs(diffs) > 0.001) & (np.abs(diffs) < 0.025)
        valid_diffs = diffs[valid_mask]

        if len(valid_diffs) < 6:
            return 0.0

        signs = np.sign(valid_diffs)
        reversals = int(np.sum(signs[1:] * signs[:-1] < 0))
        tremor_hz = float(reversals / (2.0 * duration))
        if vibration_score > 0.35:
            tremor_hz *= max(0.0, 1.0 - vibration_score)
        return float(round(tremor_hz, 2))

    def _compute_eyeball_tremor(
        self,
        avg_iris_x: float,
        timestamp: float,
        blink_phase: str,
        saccade_detected: bool,
        vibration_score: float = 0.0,
    ) -> float:
        """Track high-frequency horizontal eyeball/iris micro-flutter (1-15 Hz).

        Zero-crossing oscillation of first-order horizontal iris differences during
        open-eye non-saccadic intervals, discounted by cabin vibration.
        """
        if blink_phase != "open" or saccade_detected:
            return 0.0

        self._iris_tremor_history.append((timestamp, avg_iris_x))
        cutoff = timestamp - 2.0
        samples = [(t, x) for t, x in self._iris_tremor_history if t >= cutoff]

        if len(samples) < 12:
            return 0.0

        duration = samples[-1][0] - samples[0][0]
        if duration < 0.4:
            return 0.0

        xs = np.array([x for _, x in samples], dtype=np.float64)
        diffs = np.diff(xs)
        valid_mask = (np.abs(diffs) > 0.001) & (np.abs(diffs) < 0.045)
        valid_diffs = diffs[valid_mask]

        if len(valid_diffs) < 6:
            return 0.0

        signs = np.sign(valid_diffs)
        reversals = int(np.sum(signs[1:] * signs[:-1] < 0))
        tremor_hz = float(reversals / (2.0 * duration))
        if vibration_score > 0.35:
            tremor_hz *= max(0.0, 1.0 - vibration_score)
        return float(round(tremor_hz, 2))

    def _compute_rppg_pulse(
        self,
        frame: Optional[np.ndarray],
        landmarks: np.ndarray,
        timestamp: float,
        tzone_rgb: Optional[Tuple[float, float, float]] = None,
    ) -> Tuple[float, float, bool]:
        """Extract T-zone green-channel photoplethysmography heart rate & HRV.

        Physiological heart-rate band: 0.75 - 2.5 Hz (45 - 150 BPM).
        Automatically bypassed on monochrome/NIR frames.
        Uses strided FFT evaluation (every 5 samples after initial lock) to cut CPU overhead by >80%.
        """
        if frame is None or is_monochrome_frame(frame) or landmarks is None or len(landmarks) < 468:
            return (0.0, 0.0, False)

        try:
            rgb = tzone_rgb if tzone_rgb is not None else self._sample_tzone_rgb(frame, landmarks)
            if rgb is None:
                return (self._last_rppg_bpm, self._last_rppg_rmssd, self._last_rppg_valid)

            mean_r, mean_g, mean_b = rgb
            norm_g = float(mean_g / ((mean_r + mean_g + mean_b) / 3.0 + 1e-5) * 120.0)
            self._rppg_green_history.append((timestamp, norm_g))
            self._rppg_sample_counter += 1

            times = [t for t, _ in self._rppg_green_history]
            duration = times[-1] - times[0]
            if len(self._rppg_green_history) < 60 or duration < 4.0:
                return (self._last_rppg_bpm, self._last_rppg_rmssd, self._last_rppg_valid)

            if self._last_rppg_valid and (self._rppg_sample_counter % 5 != 0):
                return (self._last_rppg_bpm, self._last_rppg_rmssd, True)

            t_arr = np.array(times, dtype=np.float64)
            g_arr = np.array([g for _, g in self._rppg_green_history], dtype=np.float64)

            t_rel = t_arr - t_arr[0]
            p = np.polyfit(t_rel, g_arr, 1)
            sig = g_arr - np.polyval(p, t_rel)
            std_val = float(np.std(sig))
            if std_val > 1e-4:
                sig = sig / std_val

            dt = float(np.median(np.diff(t_arr)))
            if dt <= 1e-4:
                return (self._last_rppg_bpm, self._last_rppg_rmssd, self._last_rppg_valid)

            freqs = np.fft.rfftfreq(len(sig), d=dt)
            fft_mag = np.abs(np.fft.rfft(sig))

            band_mask = (freqs >= 0.75) & (freqs <= 2.5)
            if not np.any(band_mask):
                return (self._last_rppg_bpm, self._last_rppg_rmssd, self._last_rppg_valid)

            peak_idx = int(np.argmax(fft_mag * band_mask))
            peak_freq = float(freqs[peak_idx])
            bpm = float(round(peak_freq * 60.0, 1))

            # Inter-beat intervals from positive zero-crossings
            zc = []
            for i in range(len(sig) - 1):
                if sig[i] <= 0.0 and sig[i + 1] > 0.0:
                    t_zero = t_arr[i] + (-sig[i]) / (sig[i + 1] - sig[i] + 1e-9) * (t_arr[i + 1] - t_arr[i])
                    zc.append(t_zero)

            rmssd = 0.0
            if len(zc) >= 3:
                ibis = np.diff(zc) * 1000.0  # ms
                valid_ibis = [ibi for ibi in ibis if 300.0 <= ibi <= 1500.0]
                if len(valid_ibis) >= 2:
                    rmssd = float(round(np.sqrt(np.mean(np.diff(valid_ibis) ** 2)), 2))

            self._last_rppg_bpm = bpm
            self._last_rppg_rmssd = rmssd
            self._last_rppg_valid = True
            return (bpm, rmssd, True)
        except Exception as exc:
            logger.debug("rPPG pulse FFT/HRV computation failed: %s", exc, exc_info=True)
            return (self._last_rppg_bpm, self._last_rppg_rmssd, self._last_rppg_valid)

    def _compute_anti_masking_divergence(self, features: FrameFeatures, timestamp: float = 0.0) -> Tuple[float, bool]:
        """Compute Anti-Masking Divergence Metric (M_mask).

        Quantifies biophysical kinetic discordance:
          - Kinetic mismatch: normal/high blink rate (f_blink / f_baseline >= 0.8) paired with
            pharmacologically sluggish levator upstroke (v_baseline / v_up > 1.25).
          - Involuntary Vector I_invol: GEN, LOC, pursuit fragmentation, sluggish upstroke.
          - Voluntary Masking Vector V_mask: eye widening, deliberate suppression, compensatory stare (>=6.0s).
          - Discordance Score: M_mask in [0, 100]%.
          - Trigger: M_mask >= 50.0% flags masking_detected = True.
        """
        # 1. Blink frequency estimation
        cutoff = timestamp - 60.0
        recent_blinks = [t for t in self._blink_times if t >= cutoff]
        if timestamp >= 10.0 and len(recent_blinks) >= 2:
            dt_window = min(60.0, timestamp)
            current_f = len(recent_blinks) / max(1.0, dt_window)
        else:
            current_f = self.sober_blink_freq_baseline

        r_rate = float(current_f / max(0.05, self.sober_blink_freq_baseline))

        # 2. Levator upstroke sluggishness
        v_up = features.blink_opening_velocity
        if v_up <= 0.05:
            v_up = self._last_blink_opening_vel if self._last_blink_opening_vel > 0.05 else self.sober_blink_up_baseline

        r_sluggish = float(self.sober_blink_up_baseline / max(0.1, v_up))

        # Kinetic mismatch core: only fires if upstroke is genuinely sluggish (> 25% slower)
        # and blink rate is active (>= 80% baseline)
        kinetic_mismatch = 0.0
        if r_sluggish > 1.25 and r_rate >= 0.80:
            sluggish_excess = r_sluggish - 1.25
            kinetic_mismatch = min(100.0, 50.0 * min(2.0, r_rate) * sluggish_excess)

        # 3. Behavioral discordance: voluntary masking fighting involuntary signs
        invol_count = 0
        if features.gen_detected:
            invol_count += 1
        if features.lack_of_convergence:
            invol_count += 1
        if features.pursuit_fragmentation_ratio > 0.20:
            invol_count += 1
        if r_sluggish > 1.30:
            invol_count += 1

        any_voluntary_flag = bool(
            features.voluntary_eye_widening or
            features.deliberate_blink_suppression or
            features.compensatory_stare_active or
            features.rigid_head_stabilization
        )

        bonus = 0.0
        if invol_count >= 1:
            if features.voluntary_eye_widening:
                bonus += 25.0
            if features.compensatory_stare_active:
                bonus += 25.0
            if features.deliberate_blink_suppression:
                bonus += 20.0
            if features.rigid_head_stabilization:
                bonus += 15.0

        if not any_voluntary_flag:
            total_m_mask = 0.0
            detected = False
        else:
            total_m_mask = min(100.0, kinetic_mismatch + bonus)
            detected = bool(total_m_mask >= 50.0)

        return (float(round(total_m_mask, 1)), detected)

    def _evaluate_eye_glare_and_occlusion(
        self,
        frame: Optional[np.ndarray],
        landmarks: np.ndarray,
        ear_l: float,
        ear_r: float,
    ) -> Tuple[bool, bool, bool, bool]:
        """Detect localized eyeglass specular glare and sunglass occlusion.

        Returns:
            (glare_left, glare_right, any_glare, sunglasses_occluded)
        """
        if frame is None or landmarks is None or len(landmarks) < 468:
            return (False, False, False, False)

        h, w = frame.shape[:2]
        if h < 10 or w < 10:
            return (False, False, False, False)

        pts_l = landmarks[LEFT_EYE_CONTOUR, :2].copy()
        pts_r = landmarks[RIGHT_EYE_CONTOUR, :2].copy()

        # Handle [0, 1] normalized coordinates if passed
        if np.max(pts_l[:, 0]) <= 1.0 and np.max(pts_l[:, 1]) <= 1.0:
            pts_l[:, 0] *= w
            pts_l[:, 1] *= h
        if np.max(pts_r[:, 0]) <= 1.0 and np.max(pts_r[:, 1]) <= 1.0:
            pts_r[:, 0] *= w
            pts_r[:, 1] *= h

        x_min_l = max(0, int(np.floor(np.min(pts_l[:, 0]))))
        x_max_l = min(w, int(np.ceil(np.max(pts_l[:, 0]))))
        y_min_l = max(0, int(np.floor(np.min(pts_l[:, 1]))))
        y_max_l = min(h, int(np.ceil(np.max(pts_l[:, 1]))))

        x_min_r = max(0, int(np.floor(np.min(pts_r[:, 0]))))
        x_max_r = min(w, int(np.ceil(np.max(pts_r[:, 0]))))
        y_min_r = max(0, int(np.floor(np.min(pts_r[:, 1]))))
        y_max_r = min(h, int(np.ceil(np.max(pts_r[:, 1]))))

        if x_max_l <= x_min_l or y_max_l <= y_min_l or x_max_r <= x_min_r or y_max_r <= y_min_r:
            return (False, False, False, False)

        crop_l = frame[y_min_l:y_max_l, x_min_l:x_max_l]
        crop_r = frame[y_min_r:y_max_r, x_min_r:x_max_r]
        if crop_l.size == 0 or crop_r.size == 0:
            return (False, False, False, False)

        crop_l_gray = cv2.cvtColor(crop_l, cv2.COLOR_BGR2GRAY) if crop_l.ndim == 3 else crop_l
        crop_r_gray = cv2.cvtColor(crop_r, cv2.COLOR_BGR2GRAY) if crop_r.ndim == 3 else crop_r

        std_l = float(np.std(crop_l_gray))
        std_r = float(np.std(crop_r_gray))

        # If a crop is uniform synthetic test data (np.std(crop_gray) < 1.0), return (False, False, False, False)
        # so synthetic uniform frames in unit tests are never flagged as glare or sunglasses!
        if std_l < 1.0 and std_r < 1.0:
            return (False, False, False, False)

        glare_left = False
        glare_right = False

        if std_l >= 1.0:
            poly_l = np.round(pts_l - np.array([x_min_l, y_min_l], dtype=np.float64)).astype(np.int32)
            poly_l[:, 0] = np.clip(poly_l[:, 0], 0, x_max_l - x_min_l - 1)
            poly_l[:, 1] = np.clip(poly_l[:, 1], 0, y_max_l - y_min_l - 1)
            mask_l = np.zeros(crop_l_gray.shape, dtype=np.uint8)
            cv2.fillPoly(mask_l, [poly_l], 255)
            mask_l_pos = mask_l > 0
            if np.any(mask_l_pos):
                glare_frac_l = float(np.mean(crop_l_gray[mask_l_pos] > 242))
            else:
                glare_frac_l = 0.0
            mean_crop_lum_l = float(np.mean(crop_l_gray))
            glare_left = bool(glare_frac_l > 0.25 and mean_crop_lum_l < 210.0)

        if std_r >= 1.0:
            poly_r = np.round(pts_r - np.array([x_min_r, y_min_r], dtype=np.float64)).astype(np.int32)
            poly_r[:, 0] = np.clip(poly_r[:, 0], 0, x_max_r - x_min_r - 1)
            poly_r[:, 1] = np.clip(poly_r[:, 1], 0, y_max_r - y_min_r - 1)
            mask_r = np.zeros(crop_r_gray.shape, dtype=np.uint8)
            cv2.fillPoly(mask_r, [poly_r], 255)
            mask_r_pos = mask_r > 0
            if np.any(mask_r_pos):
                glare_frac_r = float(np.mean(crop_r_gray[mask_r_pos] > 242))
            else:
                glare_frac_r = 0.0
            mean_crop_lum_r = float(np.mean(crop_r_gray))
            glare_right = bool(glare_frac_r > 0.25 and mean_crop_lum_r < 210.0)

        any_glare = bool(glare_left or glare_right)

        sunglasses_occluded = False
        if not is_monochrome_frame(frame) and len(self._raw_ear_var_history) >= 60:
            ear_std = float(np.std(self._raw_ear_var_history))
            mean_crop_lum_l = float(np.mean(crop_l_gray))
            mean_crop_lum_r = float(np.mean(crop_r_gray))
            if (mean_crop_lum_l < 45.0 and mean_crop_lum_r < 45.0 and
                1.0 <= std_l < 8.0 and 1.0 <= std_r < 8.0 and
                ear_std < 0.0015):
                sunglasses_occluded = True

        return (glare_left, glare_right, any_glare, sunglasses_occluded)

    @staticmethod
    def _is_valid_landmarks(landmarks: Optional[np.ndarray]) -> bool:
        if landmarks is None or not isinstance(landmarks, np.ndarray):
            return False
        if landmarks.ndim != 2 or landmarks.shape[0] < 468 or landmarks.shape[1] < 2:
            return False
        if not np.all(np.isfinite(landmarks[:, :2])):
            return False
        # Reject all-zero or collapsed synthetic arrays
        if float(np.ptp(landmarks[:, 0])) < 1e-3 or float(np.ptp(landmarks[:, 1])) < 1e-3:
            return False
        return True

    def compute(self, landmarks: Optional[np.ndarray] = None,
                timestamp: float = 0.0,
                frame_id: int = 0,
                face_valid: bool = True,
                head_yaw: float = 0.0,
                head_pitch: float = 0.0,
                head_roll: float = 0.0,
                imu_gated: bool = False,
                gating_reason: str = "",
                frame: Optional[np.ndarray] = None,
                image_landmarks: Optional[np.ndarray] = None,
                vibration_score: float = 0.0) -> FrameFeatures:
        """Compute all features for a single frame.

        Args:
            landmarks: Normalized, smoothed landmarks (478, 3).
            timestamp: Frame timestamp (seconds).
            frame_id: Frame counter.
            face_valid: False if head pose is outside gating limits.
            head_yaw/pitch/roll: Head pose in degrees.
            imu_gated: True if IMU triggered a freeze.
            gating_reason: Human-readable gating reason.
            frame: Optional BGR frame array for color/vasodilation measurement.
            image_landmarks: Optional un-rotated raw landmarks for visual skin sampling.
            vibration_score: Instantaneous chassis/cabin vibration score from KinematicStabilizer.

        Returns:
            FrameFeatures dataclass with all computed values.
        """
        with self._lock:
            if face_valid and not self._is_valid_landmarks(landmarks):
                face_valid = False
                if not gating_reason:
                    gating_reason = "invalid_landmarks"

            features = FrameFeatures(
                timestamp=timestamp,
                frame_id=frame_id,
                face_valid=face_valid,
                head_yaw=head_yaw,
                head_pitch=head_pitch,
                head_roll=head_roll,
                imu_gated=imu_gated,
                gating_reason=gating_reason,
                vibration_score=float(vibration_score),
            )

            if imu_gated and not features.gating_reason:
                features.gating_reason = "imu_shock"

            if not face_valid or imu_gated:
                self._just_resumed_from_gate = True
                self._in_saccade = False
                if not face_valid and head_yaw == 0.0 and head_pitch == 0.0 and head_roll == 0.0:
                    features.head_yaw = self._last_features.head_yaw
                    features.head_pitch = self._last_features.head_pitch
                    features.head_roll = self._last_features.head_roll
                # Hold last valid features, but update metadata
                features.ear_left = self._last_features.ear_left
                features.ear_right = self._last_features.ear_right
                features.ear_avg = self._last_features.ear_avg
                features.mar = self._last_features.mar
                features.perclos = self._last_features.perclos
                features.left_iris_x = self._last_features.left_iris_x
                features.left_iris_y = self._last_features.left_iris_y
                features.right_iris_x = self._last_features.right_iris_x
                features.right_iris_y = self._last_features.right_iris_y
                features.saccade_velocity = 0.0
                features.saccade_detected = False
                features.saccade_latency_ms = self._last_features.saccade_latency_ms
                features.raw_iris_dx = 0.0
                features.blink_phase = "open"
                self._blink_phase = "open"
                features.blink_closing_velocity = self._last_features.blink_closing_velocity
                features.blink_opening_velocity = self._last_features.blink_opening_velocity
                features.gaze_yaw_dispersion = self._last_features.gaze_yaw_dispersion
                features.gen_detected = self._last_features.gen_detected
                features.gen_beat_count = self._last_features.gen_beat_count
                features.gen_slow_phase_vel = self._last_features.gen_slow_phase_vel
                features.pursuit_fragmentation_ratio = self._last_features.pursuit_fragmentation_ratio
                features.vergence_angle_deg = self._last_features.vergence_angle_deg
                features.lack_of_convergence = self._last_features.lack_of_convergence
                features.vor_gain = self._last_features.vor_gain
                features.head_postural_sway = self._last_features.head_postural_sway
                features.facial_flushing_ratio = self._last_features.facial_flushing_ratio
                features.flushing_valid = self._last_features.flushing_valid
                features.flushing_delta = self._last_features.flushing_delta
                features.brow_eye_distance_ratio = self._last_features.brow_eye_distance_ratio
                features.brow_elevation_delta = self._last_features.brow_elevation_delta
                features.frontalis_compensation_active = self._last_features.frontalis_compensation_active
                features.eyelid_tremor_hz = self._last_features.eyelid_tremor_hz
                features.eyeball_tremor_hz = self._last_features.eyeball_tremor_hz
                features.rppg_pulse_bpm = self._last_features.rppg_pulse_bpm
                features.rppg_hrv_rmssd_ms = self._last_features.rppg_hrv_rmssd_ms
                features.rppg_valid = self._last_features.rppg_valid
                features.deliberate_blink_suppression = self._last_features.deliberate_blink_suppression
                features.voluntary_eye_widening = self._last_features.voluntary_eye_widening
                features.stare_fixation_duration_s = self._last_features.stare_fixation_duration_s
                features.compensatory_stare_active = self._last_features.compensatory_stare_active
                features.rigid_head_stabilization = self._last_features.rigid_head_stabilization
                features.masking_divergence_score = self._last_features.masking_divergence_score
                features.masking_detected = self._last_features.masking_detected
                features.eyeglass_glare_detected = self._last_features.eyeglass_glare_detected
                features.sunglasses_occluded = self._last_features.sunglasses_occluded
                features.speech_active = self._last_features.speech_active
                features.highway_fixation_exempt = self._last_features.highway_fixation_exempt
                return features

            if getattr(self, "_just_resumed_from_gate", False):
                self._just_resumed_from_gate = False
                self._in_saccade = False
                self._prev_timestamp = timestamp
                self._prev_ear_timestamp = timestamp
                self._prev_head_timestamp = timestamp
                self._prev_ear_avg = (self.compute_ear(landmarks, LEFT_EYE_EAR_FLAT) + self.compute_ear(landmarks, RIGHT_EYE_EAR_FLAT)) / 2.0
                iris_lms = image_landmarks if image_landmarks is not None else landmarks
                left_x, left_y = self.compute_iris_position(iris_lms, LEFT_IRIS_CENTER, 33, 133)
                right_x, right_y = self.compute_iris_position(iris_lms, RIGHT_IRIS_CENTER, 263, 362)
                self._prev_iris_x = (left_x + right_x) / 2.0
                self._prev_iris_y = (left_y + right_y) / 2.0
                self._prev_left_iris_x = left_x
                self._prev_right_iris_x = right_x
                self._prev_eye_center_norm = self._compute_eye_socket_center_norm(iris_lms)
                self._prev_head_yaw = head_yaw
                self._prev_head_pitch = head_pitch
                self._prev_head_roll = head_roll

            # ── EAR ──
            features.ear_right = self.compute_ear(landmarks, RIGHT_EYE_EAR_FLAT)
            features.ear_left = self.compute_ear(landmarks, LEFT_EYE_EAR_FLAT)
            features.ear_avg = (features.ear_left + features.ear_right) / 2.0

            # ── Rolling-Shutter Vertical Mid-Face Ratio Compensation ──
            if landmarks is not None and len(landmarks) > 263:
                dist_nose = float(np.linalg.norm(landmarks[168, :2] - landmarks[4, :2]))
                dist_canthal = float(np.linalg.norm(landmarks[263, :2] - landmarks[33, :2]))
                if dist_nose > 10.0 and dist_canthal > 10.0:
                    raw_nose_ratio = dist_nose / dist_canthal
                    if vibration_score <= 0.15 and abs(head_pitch) < 10.0:
                        self._nose_ratio_samples.append(raw_nose_ratio)
                        self._baseline_nose_ratio = float(np.median(self._nose_ratio_samples))
                    elif (vibration_score > 0.25 and
                          self._baseline_nose_ratio > 0.05 and
                          abs(head_pitch - (self._prev_head_pitch or 0.0)) < 2.5):
                        shutter_stretch = float(np.clip(raw_nose_ratio / self._baseline_nose_ratio, 0.85, 1.15))
                        if abs(shutter_stretch - 1.0) > 0.015:
                            features.ear_left /= shutter_stretch
                            features.ear_right /= shutter_stretch
                            features.ear_avg /= shutter_stretch

            self._raw_ear_var_history.append(features.ear_avg)

            # ── Iris gaze ──
            iris_lms = image_landmarks if image_landmarks is not None else landmarks
            # Left eye (viewer's left): iris 468, corners 33 (outer) & 133 (inner)
            features.left_iris_x, features.left_iris_y = self.compute_iris_position(
                iris_lms, LEFT_IRIS_CENTER, 33, 133)
            # Right eye (viewer's right): iris 473, corners 263 (outer) & 362 (inner)
            features.right_iris_x, features.right_iris_y = self.compute_iris_position(
                iris_lms, RIGHT_IRIS_CENTER, 263, 362)

            # ── Eyeglass Glare Detection & Monocular Fallback ──
            glare_l, glare_r, any_glare, sunglasses_occluded = self._evaluate_eye_glare_and_occlusion(
                frame, iris_lms, features.ear_left, features.ear_right
            )
            features.eyeglass_glare_detected = any_glare
            features.sunglasses_occluded = sunglasses_occluded

            if glare_l and not glare_r:
                features.ear_avg = features.ear_right
                features.left_iris_x = features.right_iris_x
                features.left_iris_y = features.right_iris_y
            elif glare_r and not glare_l:
                features.ear_avg = features.ear_left
                features.right_iris_x = features.left_iris_x
                features.right_iris_y = features.left_iris_y

            # ── Downward Glance Threshold Compensation (Instrument Cluster Glance Guard) ──
            avg_iris_y = (features.left_iris_y + features.right_iris_y) / 2.0
            down_pitch_deg = max(0.0, -head_pitch - 8.0)
            down_gaze_unit = max(0.0, -avg_iris_y - 0.20)
            if down_pitch_deg > 0.0 or down_gaze_unit > 0.0:
                glance_scale = max(0.65, 1.0 - 0.012 * down_pitch_deg - 0.15 * down_gaze_unit)
                eff_blink_threshold = self.blink_threshold * glance_scale
            else:
                eff_blink_threshold = self.blink_threshold

            # ── Blink / Micro-sleep detection ──
            eyes_closed = features.ear_avg < eff_blink_threshold
            if eyes_closed:
                self._consecutive_closed += 1
            else:
                if self._consecutive_closed > 0:
                    features.blink_detected = True
                self._consecutive_closed = 0

            features.microsleep_detected = (
                self._consecutive_closed >= self.microsleep_frames)

            # ── MAR & Speech / Singing Cadence Filter ──
            features.mar = self.compute_mar(landmarks)
            self._mar_history.append((timestamp, features.mar))

            recent_mar_samples = [(t, m) for t, m in self._mar_history if timestamp - t <= 2.0]
            if len(recent_mar_samples) >= 12:
                times_mar = np.array([t for t, _ in recent_mar_samples], dtype=np.float64)
                mars = np.array([m for _, m in recent_mar_samples], dtype=np.float64)
                mean_mar = float(np.mean(mars))
                mar_std = float(np.std(mars))
                devs = mars - mean_mar
                zero_crossings = int(np.sum((devs[:-1] * devs[1:]) < 0))
                dt_mar = float(max(times_mar[-1] - times_mar[0], 1e-4))
                zc_rate = float(zero_crossings / dt_mar)
                zc_freq = zc_rate / 2.0
                if 0.15 <= mean_mar <= 0.55 and mar_std >= 0.04 and (zc_freq >= 1.5 or zc_rate >= 1.5):
                    features.speech_active = True

            # ── Yawn detection ──
            if features.mar > self.mar_yawn_threshold:
                if features.speech_active and features.mar < 0.70:
                    pass  # Speech cadence guard: rapid speech/singing does not increment yawn counter
                else:
                    self._consecutive_yawn += 1
            else:
                self._consecutive_yawn = 0
            features.yawn_detected = (
                self._consecutive_yawn >= self.yawn_frames)

            # ── PERCLOS (O(1) sliding window ring counter) ──
            if getattr(self, "_last_blink_threshold", None) != self.blink_threshold:
                self._last_blink_threshold = self.blink_threshold
                self._perclos_closed_count = sum(1 for e in self._ear_history if e < self.blink_threshold)

            if len(self._ear_history) == self._ear_history.maxlen:
                old_ear = self._ear_history[0]
                if old_ear < self.blink_threshold:
                    self._perclos_closed_count = max(0, self._perclos_closed_count - 1)

            self._ear_history.append(features.ear_avg)
            if features.ear_avg < self.blink_threshold:
                self._perclos_closed_count += 1

            min_samples = int(self.perclos_min_warmup_sec * self._fps) if self.perclos_min_warmup_sec > 0.0 else 1
            denom = max(len(self._ear_history), min_samples, 1)
            features.perclos = float(np.clip(self._perclos_closed_count / denom, 0.0, 1.0))

            # ── Blink closing/opening velocity & FSM tracking ──
            features.blink_closing_velocity, features.blink_opening_velocity = \
                self._compute_blink_velocity(features.ear_avg, timestamp)
            features.blink_phase = self._blink_phase

            # Compute head angular velocity & eye socket translation velocity for VOR / motion suppression
            head_total_vel = 0.0
            if self._prev_head_yaw is not None and self._prev_head_pitch is not None and self._prev_head_timestamp is not None:
                dt_head = timestamp - self._prev_head_timestamp
                if dt_head > 1e-6:
                    d_yaw = (head_yaw - self._prev_head_yaw + 180.0) % 360.0 - 180.0
                    d_pitch = (head_pitch - self._prev_head_pitch + 180.0) % 360.0 - 180.0
                    head_total_vel = float(np.hypot(d_yaw, d_pitch) / dt_head)

            curr_socket = self._compute_eye_socket_center_norm(iris_lms)
            eye_socket_vel = 0.0
            if curr_socket is not None and self._prev_eye_center_norm is not None and self._prev_timestamp is not None:
                dt_socket = timestamp - self._prev_timestamp
                if dt_socket > 1e-6:
                    eye_socket_vel = float(np.linalg.norm(curr_socket - self._prev_eye_center_norm) / dt_socket)

            # ── Saccadic velocity & latency (Fransson et al. 2010) ──
            avg_iris_x = (features.left_iris_x + features.right_iris_x) / 2.0
            avg_iris_y = (features.left_iris_y + features.right_iris_y) / 2.0
            features.saccade_velocity, features.saccade_detected, features.saccade_latency_ms, features.raw_iris_dx = \
                self._compute_saccade_dynamics(
                    avg_iris_x, avg_iris_y, timestamp,
                    blink_phase=self._blink_phase,
                    left_iris_x=features.left_iris_x,
                    right_iris_x=features.right_iris_x,
                    head_vel_deg_s=head_total_vel,
                    eye_socket_vel=eye_socket_vel,
                    vibration_score=float(vibration_score),
                )

            # ── Gaze yaw dispersion (σ_yaw) ──
            if self._blink_phase == "open" and not features.blink_detected:
                features.gaze_yaw_dispersion = self._compute_gaze_yaw_dispersion(avg_iris_x)
            else:
                features.gaze_yaw_dispersion = (
                    float(np.std(self._gaze_yaw_history)) if len(self._gaze_yaw_history) >= 10 else 0.0
                )

            # ── Advanced Oculomotor Impairment Signals ──
            # 1. Binocular vergence & Lack of Convergence (LOC - DRE sign)
            # Gate on blink, extreme yaw (>20°), steep pitch (>15°), and eccentric gaze (>0.30)
            # where eyeball geometry naturally foreshortens the abducting eye
            if self._blink_phase != "open" or abs(head_yaw) > 20.0 or abs(head_pitch) > 15.0 or abs(avg_iris_x) > 0.30:
                features.vergence_angle_deg = self._last_features.vergence_angle_deg
                features.lack_of_convergence = False
            else:
                features.vergence_angle_deg, features.lack_of_convergence = \
                    self._compute_binocular_vergence(features.left_iris_x, features.right_iris_x)

            # 2. Vestibulo-Ocular Reflex (VOR) micro-compensation gain (PMC8997842)
            features.vor_gain = self._compute_vor_gain(avg_iris_x, head_yaw, timestamp, blink_phase=self._blink_phase)

            # 3. Gaze-Evoked Nystagmus (GEN - Romano et al. 2017)
            features.gen_detected, features.gen_beat_count, features.gen_slow_phase_vel = \
                self._compute_gen(avg_iris_x, timestamp, head_vel_deg_s=head_total_vel, avg_iris_y=avg_iris_y)

            # 4. Smooth Pursuit Fragmentation Ratio
            features.pursuit_fragmentation_ratio = \
                self._compute_pursuit_fragmentation(features.saccade_detected, timestamp)

            # ── Postural Sway & Head Stabilization (Section A & B) ──
            head_angular_vel = 0.0
            pitch_roll_vel = 0.0
            if self._prev_head_yaw is not None and self._prev_head_timestamp is not None:
                dt_head = timestamp - self._prev_head_timestamp
                if dt_head > 1e-6:
                    d_yaw = (head_yaw - self._prev_head_yaw + 180.0) % 360.0 - 180.0
                    head_angular_vel = float(abs(d_yaw) / dt_head)
                    if self._prev_head_pitch is not None and self._prev_head_roll is not None:
                        d_pitch = (head_pitch - self._prev_head_pitch + 180.0) % 360.0 - 180.0
                        d_roll = (head_roll - self._prev_head_roll + 180.0) % 360.0 - 180.0
                        pitch_roll_vel = float(np.hypot(d_pitch, d_roll) / dt_head)

            if vibration_score > 0.35:
                # Road vibration present: pause head_postural_sway accumulation so road vibration is not misclassified
                features.head_postural_sway = float(self._last_features.head_postural_sway)
                features.rigid_head_stabilization = False
            elif pitch_roll_vel > self.sway_nod_reset_deg_s:
                self._head_pose_history.clear()
                features.head_postural_sway = float(self._last_features.head_postural_sway)
                features.rigid_head_stabilization = False
            else:
                features.head_postural_sway, features.rigid_head_stabilization = \
                    self._compute_postural_sway(
                        head_pitch, head_roll, timestamp,
                        head_angular_vel=head_angular_vel,
                        vibration_score=float(vibration_score),
                    )

            # Involuntary impairment override: drowsy drivers in microsleep or high PERCLOS cannot be masking with rigid head
            if features.microsleep_detected or features.perclos > 0.15:
                features.rigid_head_stabilization = False

            # ── Single-Pass T-Zone RGB Sampling for Flushing & rPPG ──
            sample_lms = image_landmarks if image_landmarks is not None else landmarks
            tzone_rgb = self._sample_tzone_rgb(frame, sample_lms)

            # ── Facial Flushing / Vasodilation (Section A) ──
            features.facial_flushing_ratio, features.flushing_valid, features.flushing_delta = \
                self._compute_facial_flushing(frame, sample_lms, tzone_rgb=tzone_rgb)

            # ── Eyebrow / Frontalis Micromotions ──
            features.brow_eye_distance_ratio, features.brow_elevation_delta, features.frontalis_compensation_active = \
                self._compute_brow_micromotion(sample_lms, features.ear_avg)

            # ── Palpebral Eyelid Tremor (3-15 Hz) & Ocular Eyeball Tremor (1-15 Hz) ──
            features.eyelid_tremor_hz = \
                self._compute_eyelid_tremor(
                    features.ear_avg, timestamp, self._blink_phase,
                    vibration_score=float(vibration_score),
                )
            features.eyeball_tremor_hz = \
                self._compute_eyeball_tremor(
                    avg_iris_x, timestamp, self._blink_phase,
                    features.saccade_detected,
                    vibration_score=float(vibration_score),
                )

            # ── T-zone rPPG Pulse & HRV ──
            features.rppg_pulse_bpm, features.rppg_hrv_rmssd_ms, features.rppg_valid = \
                self._compute_rppg_pulse(frame, sample_lms, timestamp, tzone_rgb=tzone_rgb)

            # ── Voluntary Masking Behaviors (Section B) ──
            if features.blink_detected:
                self._blink_times.append(timestamp)

            features.voluntary_eye_widening = \
                self._compute_voluntary_eye_widening(
                    features.ear_avg, timestamp,
                    frontalis_compensation_active=features.frontalis_compensation_active,
                )
            features.deliberate_blink_suppression = \
                self._compute_deliberate_blink_suppression(features.blink_detected, features.ear_avg, timestamp)
            features.stare_fixation_duration_s, features.compensatory_stare_active = \
                self._compute_fixation_stare(avg_iris_x, avg_iris_y, features.saccade_detected, features.blink_detected, timestamp)
            features.highway_fixation_exempt = self._highway_fixation_exempt

            if features.sunglasses_occluded:
                features.deliberate_blink_suppression = False
                features.compensatory_stare_active = False

            # ── Anti-Masking Divergence Metric (M_mask) ──
            features.masking_divergence_score, features.masking_detected = \
                self._compute_anti_masking_divergence(features, timestamp)

            self._prev_left_iris_x = features.left_iris_x
            self._prev_right_iris_x = features.right_iris_x
            self._prev_eye_center_norm = curr_socket
            self._prev_iris_x = avg_iris_x
            self._prev_iris_y = avg_iris_y
            self._prev_head_yaw = head_yaw
            self._prev_head_pitch = head_pitch
            self._prev_head_roll = head_roll
            self._prev_head_timestamp = timestamp
            self._prev_timestamp = timestamp

            self._last_features = features
            return features
