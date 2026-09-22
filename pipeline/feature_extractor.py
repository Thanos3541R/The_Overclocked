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
import numpy as np
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, List, Optional, Tuple

from utils.landmarks import (
    RIGHT_EYE_EAR_FLAT, LEFT_EYE_EAR_FLAT,
    RIGHT_IRIS_CENTER, LEFT_IRIS_CENTER,
    RIGHT_EYE_OUTER, LEFT_EYE_OUTER,
    RIGHT_EYE_EAR_INDICES, LEFT_EYE_EAR_INDICES,
    MAR_VERTICAL_PAIRS, MAR_HORIZONTAL,
    RIGHT_CHEEK, LEFT_CHEEK, CHEEK_LANDMARKS,
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

    # ── Section B — Voluntary & Semi-Voluntary Changes (Maskable / Gaming Behaviors) ──
    # Susceptible to conscious suppression; used for corroboration and mismatch detection
    deliberate_blink_suppression: bool = False  # Prolonged inter-blink interval (>6.0s) from fighting droop
    voluntary_eye_widening: bool = False        # Frontalis/levator contraction spike (EAR surge > 1.25x baseline)
    stare_fixation_duration_s: float = 0.0      # Rigid central staring fixation without micro-saccades
    rigid_head_stabilization: bool = False      # Conscious locking of neck muscles to suppress natural sway

    # ── Anti-Masking Divergence Metric (M_mask) ──
    # Quantifies discordance between voluntary masking attempts and involuntary reflex decay
    masking_divergence_score: float = 0.0       # 0.0 to 100.0% divergence score (M_mask)
    masking_detected: bool = False              # True when M_mask >= 50.0%

    # Gating info
    imu_gated: bool = False
    gating_reason: str = ""


def _dist(p1: np.ndarray, p2: np.ndarray) -> float:
    """Euclidean distance between two points."""
    return float(np.linalg.norm(p1 - p2))


# ─── Saccade detection defaults (tunable via PipelineConfig) ───
# Normal saccades reach 200–900 deg/s. Impaired saccades are slower.
# At 30 fps (~33.3ms/frame), 30 deg/s corresponds to ~1.0 deg/frame iris shift,
# and 15 deg/s offset corresponds to ~0.5 deg/frame.
# Approximate degrees per normalized iris displacement unit.
# Human eye: ~30° horizontal range maps to iris_x in [-1, 1].
_DEG_PER_IRIS_UNIT = 15.0         # 1 iris unit ≈ 15° of eye rotation


class FeatureExtractor:
    """Extracts facial micromotion features from landmarks."""

    def __init__(self, blink_threshold: float = 0.20,
                 mar_yawn_threshold: float = 0.60,
                 perclos_window_sec: float = 60.0,
                 perclos_alert_threshold: float = 0.15,
                 microsleep_frames: int = 15,
                 yawn_frames: int = 30,
                 gaze_dispersion_window_sec: float = 15.0,
                 saccade_onset_velocity: float = 30.0,
                 saccade_offset_velocity: float = 15.0,
                 eccentric_gaze_threshold: float = 0.35,
                 gen_drift_min_vel: float = 2.0,
                 gen_drift_max_vel: float = 18.0,
                 gen_reset_min_vel: float = 30.0,
                 loc_vergence_threshold_deg: float = -3.5,
                 vor_min_head_speed: float = 2.0,
                 vor_max_head_speed: float = 25.0,
                 pursuit_window_sec: float = 5.0,
                 fps: float = 30.0):
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

        # Rolling buffers for PERCLOS
        buffer_size = int(perclos_window_sec * fps)
        self._ear_history: Deque[float] = deque(maxlen=max(buffer_size, 1))

        # Counters for sustained events
        self._consecutive_closed = 0
        self._consecutive_yawn = 0

        # Last valid features (for gating holdover)
        self._last_features = FrameFeatures()

        # ── Saccade tracking state ──
        self._prev_iris_x: Optional[float] = None   # previous avg horizontal gaze
        self._prev_iris_y: Optional[float] = None
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

        # ── Gaze yaw dispersion (σ_yaw) ──
        dispersion_buffer_size = int(gaze_dispersion_window_sec * fps)
        self._gaze_yaw_history: Deque[float] = deque(
            maxlen=max(dispersion_buffer_size, 1))

        # ── VOR micro-compensation state ──
        self.vor_min_head_speed = vor_min_head_speed
        self.vor_max_head_speed = vor_max_head_speed
        self._prev_head_yaw: Optional[float] = None
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
        self.sway_window_sec = 3.0
        sway_buffer_size = int(self.sway_window_sec * fps)
        self._head_pose_history: Deque[Tuple[float, float, float]] = deque(maxlen=max(sway_buffer_size, 1))

        # ── Facial Flushing State (Section A) ──
        self._smoothed_flushing: float = 1.0

        # ── Voluntary Masking State (Section B) ──
        self.sober_ear_baseline: float = 0.28
        self._ear_time_history: Deque[Tuple[float, float]] = deque(maxlen=max(int(4.0 * fps), 1))
        self._last_blink_time: float = 0.0
        self._stare_start_time: Optional[float] = None
        self._stare_anchor_x: Optional[float] = None
        self._stare_anchor_y: Optional[float] = None

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
                               eye_inner_idx: int) -> tuple:
        """Compute normalized iris position relative to eye corners.

        Returns:
            (x, y) normalized to approximately [-1, 1].
            x: -1 = looking outward, +1 = looking inward.
            y: -1 = looking down, +1 = looking up.
        """
        iris = landmarks[iris_center_idx]
        outer = landmarks[eye_outer_idx]
        inner = landmarks[eye_inner_idx]

        eye_center = (outer + inner) / 2.0
        eye_width = np.linalg.norm(outer - inner)

        if eye_width < 1e-6:
            return (0.0, 0.0)

        # Horizontal: project iris offset onto eye axis
        eye_axis = (inner - outer) / eye_width
        offset = iris - eye_center
        x = float(np.dot(offset[:2], eye_axis[:2]) / (eye_width / 2.0))

        # Vertical: perpendicular component
        perp = np.array([-eye_axis[1], eye_axis[0]])
        y = float(np.dot(offset[:2], perp) / (eye_width / 2.0))

        return (np.clip(x, -1.0, 1.0), np.clip(y, -1.0, 1.0))

    def _compute_saccade_dynamics(self, iris_x: float, iris_y: float,
                                   timestamp: float) -> Tuple[float, bool, float]:
        """Compute saccadic velocity, detect saccades, and measure latency.

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

        Returns:
            (velocity_deg_s, is_saccade, latency_ms)
        """
        if self._prev_iris_x is None or self._prev_timestamp is None:
            self._prev_iris_x = iris_x
            self._prev_iris_y = iris_y
            self._prev_timestamp = timestamp
            return (0.0, False, 0.0)

        dt = timestamp - self._prev_timestamp
        if dt <= 1e-6:
            return (0.0, self._in_saccade, 0.0)

        # Angular displacement in degrees
        dx_deg = (iris_x - self._prev_iris_x) * _DEG_PER_IRIS_UNIT
        dy_deg = (iris_y - self._prev_iris_y) * _DEG_PER_IRIS_UNIT
        angular_displacement = np.sqrt(dx_deg**2 + dy_deg**2)

        velocity = angular_displacement / dt  # deg/s

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

        return (velocity, self._in_saccade, latency_ms)

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
            return (self._last_blink_closing_vel, self._last_blink_opening_vel)

        dt = timestamp - self._prev_ear_timestamp
        if dt <= 1e-6:
            return (self._last_blink_closing_vel, self._last_blink_opening_vel)

        d_ear = ear_avg - self._prev_ear_avg
        ear_velocity = abs(d_ear / dt)   # EAR units / second

        # State machine: track blink phase transitions
        if self._blink_phase == "open":
            if d_ear < -0.005 and ear_avg < 0.28:
                # EAR dropping → entering closing phase
                self._blink_phase = "closing"
                self._closing_velocities = [ear_velocity]
                self._opening_velocities = []

        elif self._blink_phase == "closing":
            if ear_avg < self.blink_threshold:
                # Eyes fully closed → transition to closed
                self._blink_phase = "closed"
                self._closing_velocities.append(ear_velocity)
            elif d_ear < 0:
                # Still closing
                self._closing_velocities.append(ear_velocity)
            else:
                # Aborted blink (EAR went back up without closing fully)
                self._blink_phase = "open"

        elif self._blink_phase == "closed":
            if d_ear > 0.005:
                # EAR rising → entering opening phase
                self._blink_phase = "opening"
                self._opening_velocities = [ear_velocity]

        elif self._blink_phase == "opening":
            if ear_avg > 0.22:
                # Eyes reopened → blink complete, compute velocities
                self._opening_velocities.append(ear_velocity)

                # Store completed blink velocities
                if self._closing_velocities:
                    self._last_blink_closing_vel = float(
                        np.mean(self._closing_velocities))
                if self._opening_velocities:
                    self._last_blink_opening_vel = float(
                        np.mean(self._opening_velocities))

                self._blink_phase = "open"
            elif d_ear > 0:
                # Still opening
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

        In normalized iris coordinates, +1 = nasal (adducted) and -1 = temporal (abducted).
        Vergence = (left_x + right_x) * _DEG_PER_IRIS_UNIT.
        Parallel forward gaze (infinity) has vergence ≈ 0.0°.
        Near convergence (instrument panel) has positive vergence (> 2.0°).
        Divergent strabismus / inability to hold convergence (DRE LOC sign) has negative vergence (< -3.5°).
        """
        vergence = (left_iris_x + right_iris_x) * _DEG_PER_IRIS_UNIT
        loc = bool(vergence < self.loc_vergence_threshold_deg)
        return (float(vergence), loc)

    def _compute_vor_gain(self, avg_iris_x: float, head_yaw: float, timestamp: float) -> float:
        """Compute Vestibulo-Ocular Reflex (VOR) micro-compensation gain (PMC8997842).

        VOR maintains gaze fixated on the road during natural head tremors/sway (2-25 deg/s).
        Eye counter-rotation velocity should match negative head velocity: gain = -d_eye / d_head.
        Sober: ~0.85-1.05. Impaired (BAC >= 0.05%): depressed (<0.7) or lagged.
        """
        if self._prev_head_yaw is None or self._prev_head_timestamp is None:
            self._prev_head_yaw = head_yaw
            self._prev_head_timestamp = timestamp
            return self._smoothed_vor_gain

        dt = timestamp - self._prev_head_timestamp
        if dt <= 1e-6:
            return self._smoothed_vor_gain

        head_vel = (head_yaw - self._prev_head_yaw) / dt   # deg/s

        if self._prev_iris_x is not None:
            eye_vel = (avg_iris_x - self._prev_iris_x) * _DEG_PER_IRIS_UNIT / dt  # deg/s

            if self.vor_min_head_speed <= abs(head_vel) <= self.vor_max_head_speed:
                raw_gain = -(eye_vel / head_vel)
                clamped_gain = float(np.clip(raw_gain, 0.0, 2.0))
                self._smoothed_vor_gain = 0.85 * self._smoothed_vor_gain + 0.15 * clamped_gain

        self._prev_head_yaw = head_yaw
        self._prev_head_timestamp = timestamp
        return float(self._smoothed_vor_gain)

    def _compute_gen(self, avg_iris_x: float, timestamp: float) -> Tuple[bool, int, float]:
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
        iris_vel = (avg_iris_x - self._prev_iris_x) * _DEG_PER_IRIS_UNIT / dt  # deg/s

        if is_eccentric:
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

    def _compute_postural_sway(self, head_pitch: float, head_roll: float, timestamp: float) -> Tuple[float, bool]:
        """Compute low-frequency involuntary head wobble/postural sway (Section A) and detect rigid head stabilization (Section B).

        Involuntary postural sway: under cerebellar/vestibular alcohol depression, micro-tremor and
        sway increase (sway > 1.4 deg).
        Voluntary rigid head stabilization: driver consciously tenses neck to freeze head (sway < 0.12 deg).
        """
        self._head_pose_history.append((timestamp, head_pitch, head_roll))
        cutoff = timestamp - self.sway_window_sec
        recent = [(p, r) for t, p, r in self._head_pose_history if t >= cutoff]
        if len(recent) < 10:
            return (0.0, False)

        pitches = [p for p, _ in recent]
        rolls = [r for _, r in recent]
        std_p = float(np.std(pitches))
        std_r = float(np.std(rolls))
        sway = float(np.sqrt(std_p**2 + std_r**2))
        rigid_stabilization = bool(sway < 0.12 and len(recent) >= 20)
        return (sway, rigid_stabilization)

    def _compute_facial_flushing(self, frame: Optional[np.ndarray], landmarks: np.ndarray) -> float:
        """Measure micro-vascular facial flushing / vasodilation in cheek regions (Section A).

        Calculates chromaticity ratio R / ((G + B)/2 + 1e-5) across MediaPipe cheek landmarks.
        Normal: ~1.00. Vasodilation / flushing under alcohol: > 1.15.
        """
        if frame is None or landmarks is None or len(landmarks) < 468:
            return self._smoothed_flushing

        try:
            h, w = frame.shape[:2]
            r_vals, g_vals, b_vals = [], [], []

            for idx in CHEEK_LANDMARKS:
                if idx < len(landmarks):
                    pt = landmarks[idx]
                    px = int(pt[0] * w) if pt[0] <= 1.0 else int(pt[0])
                    py = int(pt[1] * h) if pt[1] <= 1.0 else int(pt[1])

                    y1 = max(0, py - 3)
                    y2 = min(h, py + 4)
                    x1 = max(0, px - 3)
                    x2 = min(w, px + 4)

                    if y2 > y1 and x2 > x1:
                        patch = frame[y1:y2, x1:x2]
                        b_vals.append(float(np.mean(patch[:, :, 0])))
                        g_vals.append(float(np.mean(patch[:, :, 1])))
                        r_vals.append(float(np.mean(patch[:, :, 2])))

            if r_vals and g_vals and b_vals:
                mean_r = float(np.mean(r_vals))
                mean_gb = float((np.mean(g_vals) + np.mean(b_vals)) / 2.0)
                raw_ratio = float(mean_r / (mean_gb + 1e-5))
                clamped = float(np.clip(raw_ratio, 0.5, 2.5))
                self._smoothed_flushing = 0.90 * self._smoothed_flushing + 0.10 * clamped
        except Exception:
            pass

        return float(self._smoothed_flushing)

    def _compute_voluntary_eye_widening(self, ear_avg: float, timestamp: float) -> bool:
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
                                timestamp: float) -> float:
        """Track compensatory staring fixation duration (Section B).

        Rigid unbroken central fixation within +/- 2.5 deg for > 3.5 seconds
        without natural exploratory micro-saccades.
        """
        if self._stare_start_time is None or self._stare_anchor_x is None or self._stare_anchor_y is None:
            self._stare_start_time = timestamp
            self._stare_anchor_x = avg_iris_x
            self._stare_anchor_y = avg_iris_y
            return 0.0

        drift_deg = np.sqrt(
            ((avg_iris_x - self._stare_anchor_x) * _DEG_PER_IRIS_UNIT) ** 2 +
            ((avg_iris_y - self._stare_anchor_y) * _DEG_PER_IRIS_UNIT) ** 2
        )

        if saccade_detected or blink_detected or drift_deg > 2.5:
            # Fixation broken, re-anchor
            self._stare_start_time = timestamp
            self._stare_anchor_x = avg_iris_x
            self._stare_anchor_y = avg_iris_y
            return 0.0

        return float(timestamp - self._stare_start_time)

    def _compute_anti_masking_divergence(self, features: FrameFeatures) -> Tuple[float, bool]:
        """Compute Anti-Masking Divergence Metric (M_mask).

        Quantifies discordance between voluntary masking attempts and involuntary reflex decay:
          - Involuntary Vector I_invol in [0, 1]: GEN, depressed VOR gain, sluggish upstroke,
            pursuit fragmentation, LOC, postural sway, ptosis droop.
          - Voluntary Vector V_mask in [0, 1]: eye-widening spikes, deliberate blink suppression,
            rigid compensatory stare, head stabilization.
          - Divergence Score: M_mask = 100 * min(1.0, 2.0 * I_invol * V_mask).
          - Trigger: M_mask >= 50.0% flags masking_detected = True.
        """
        inv = 0.0
        if features.gen_detected:
            inv += 0.25
        if features.vor_gain < 0.70:
            inv += 0.20
        elif features.vor_gain < 0.80:
            inv += 0.10
        if 0 < features.blink_opening_velocity < 1.2:
            inv += 0.20
        elif 0 < features.blink_opening_velocity < 1.8:
            inv += 0.10
        if features.pursuit_fragmentation_ratio > 0.25:
            inv += 0.15
        elif features.pursuit_fragmentation_ratio > 0.12:
            inv += 0.08
        if features.lack_of_convergence:
            inv += 0.10
        if features.head_postural_sway > 1.4:
            inv += 0.10
        if features.ear_avg < 0.22:
            inv += 0.10
        inv_vec = float(np.clip(inv, 0.0, 1.0))

        vol = 0.0
        if features.voluntary_eye_widening:
            vol += 0.35
        if features.deliberate_blink_suppression:
            vol += 0.35
        if features.stare_fixation_duration_s >= 3.5:
            vol += 0.20
        if features.rigid_head_stabilization:
            vol += 0.10
        vol_vec = float(np.clip(vol, 0.0, 1.0))

        m_mask = 100.0 * min(1.0, 2.0 * inv_vec * vol_vec)
        detected = bool(m_mask >= 50.0)
        return (float(round(m_mask, 1)), detected)

    def compute(self, landmarks: np.ndarray,
                timestamp: float,
                frame_id: int,
                face_valid: bool,
                head_yaw: float = 0.0,
                head_pitch: float = 0.0,
                head_roll: float = 0.0,
                imu_gated: bool = False,
                gating_reason: str = "",
                frame: Optional[np.ndarray] = None) -> FrameFeatures:
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

        Returns:
            FrameFeatures dataclass with all computed values.
        """
        features = FrameFeatures(
            timestamp=timestamp,
            frame_id=frame_id,
            face_valid=face_valid,
            head_yaw=head_yaw,
            head_pitch=head_pitch,
            head_roll=head_roll,
            imu_gated=imu_gated,
            gating_reason=gating_reason,
        )

        if not face_valid or imu_gated:
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
            features.saccade_velocity = self._last_features.saccade_velocity
            features.saccade_detected = self._last_features.saccade_detected
            features.saccade_latency_ms = self._last_features.saccade_latency_ms
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
            features.deliberate_blink_suppression = self._last_features.deliberate_blink_suppression
            features.voluntary_eye_widening = self._last_features.voluntary_eye_widening
            features.stare_fixation_duration_s = self._last_features.stare_fixation_duration_s
            features.rigid_head_stabilization = self._last_features.rigid_head_stabilization
            features.masking_divergence_score = self._last_features.masking_divergence_score
            features.masking_detected = self._last_features.masking_detected
            return features

        # ── EAR ──
        features.ear_right = self.compute_ear(landmarks, RIGHT_EYE_EAR_FLAT)
        features.ear_left = self.compute_ear(landmarks, LEFT_EYE_EAR_FLAT)
        features.ear_avg = (features.ear_left + features.ear_right) / 2.0

        # ── MAR ──
        features.mar = self.compute_mar(landmarks)

        # ── Blink / Micro-sleep detection ──
        eyes_closed = features.ear_avg < self.blink_threshold
        if eyes_closed:
            self._consecutive_closed += 1
        else:
            if self._consecutive_closed > 0:
                features.blink_detected = True
            self._consecutive_closed = 0

        features.microsleep_detected = (
            self._consecutive_closed >= self.microsleep_frames)

        # ── Yawn detection ──
        if features.mar > self.mar_yawn_threshold:
            self._consecutive_yawn += 1
        else:
            self._consecutive_yawn = 0
        features.yawn_detected = (
            self._consecutive_yawn >= self.yawn_frames)

        # ── PERCLOS ──
        self._ear_history.append(features.ear_avg)
        if len(self._ear_history) > 0:
            closed_count = sum(
                1 for e in self._ear_history if e < self.blink_threshold)
            features.perclos = closed_count / len(self._ear_history)

        # ── Iris gaze ──
        # Left eye (subject's left): iris 468, corners 263 (outer) & 362 (inner)
        features.left_iris_x, features.left_iris_y = self.compute_iris_position(
            landmarks, LEFT_IRIS_CENTER, 263, 362)
        # Right eye (subject's right): iris 473, corners 33 (outer) & 133 (inner)
        features.right_iris_x, features.right_iris_y = self.compute_iris_position(
            landmarks, RIGHT_IRIS_CENTER, 33, 133)

        # ── Saccadic velocity & latency (Fransson et al. 2010) ──
        avg_iris_x = (features.left_iris_x + features.right_iris_x) / 2.0
        avg_iris_y = (features.left_iris_y + features.right_iris_y) / 2.0
        features.saccade_velocity, features.saccade_detected, features.saccade_latency_ms = \
            self._compute_saccade_dynamics(avg_iris_x, avg_iris_y, timestamp)

        # ── Blink closing/opening velocity ──
        features.blink_closing_velocity, features.blink_opening_velocity = \
            self._compute_blink_velocity(features.ear_avg, timestamp)

        # ── Gaze yaw dispersion (σ_yaw) ──
        features.gaze_yaw_dispersion = self._compute_gaze_yaw_dispersion(avg_iris_x)

        # ── Advanced Oculomotor Impairment Signals ──
        # 1. Binocular vergence & Lack of Convergence (LOC - DRE sign)
        features.vergence_angle_deg, features.lack_of_convergence = \
            self._compute_binocular_vergence(features.left_iris_x, features.right_iris_x)

        # 2. Vestibulo-Ocular Reflex (VOR) micro-compensation gain (PMC8997842)
        features.vor_gain = self._compute_vor_gain(avg_iris_x, head_yaw, timestamp)

        # 3. Gaze-Evoked Nystagmus (GEN - Romano et al. 2017)
        features.gen_detected, features.gen_beat_count, features.gen_slow_phase_vel = \
            self._compute_gen(avg_iris_x, timestamp)

        # 4. Smooth Pursuit Fragmentation Ratio
        features.pursuit_fragmentation_ratio = \
            self._compute_pursuit_fragmentation(features.saccade_detected, timestamp)

        # ── Postural Sway & Head Stabilization (Section A & B) ──
        features.head_postural_sway, features.rigid_head_stabilization = \
            self._compute_postural_sway(head_pitch, head_roll, timestamp)

        # ── Facial Flushing / Vasodilation (Section A) ──
        features.facial_flushing_ratio = self._compute_facial_flushing(frame, landmarks)

        # ── Voluntary Masking Behaviors (Section B) ──
        features.voluntary_eye_widening = \
            self._compute_voluntary_eye_widening(features.ear_avg, timestamp)
        features.deliberate_blink_suppression = \
            self._compute_deliberate_blink_suppression(features.blink_detected, features.ear_avg, timestamp)
        features.stare_fixation_duration_s = \
            self._compute_fixation_stare(avg_iris_x, avg_iris_y, features.saccade_detected, features.blink_detected, timestamp)

        # ── Anti-Masking Divergence Metric (M_mask) ──
        features.masking_divergence_score, features.masking_detected = \
            self._compute_anti_masking_divergence(features)

        self._prev_iris_x = avg_iris_x
        self._prev_iris_y = avg_iris_y
        self._prev_timestamp = timestamp

        self._last_features = features
        return features
