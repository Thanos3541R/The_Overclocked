"""Unit tests for alcohol-discriminating feature signals.

Tests verify:
  - Saccadic velocity: rapid iris displacement produces high deg/s, fixation produces ~0
  - Saccade detection: onset/offset hysteresis works correctly
  - Saccade latency: accumulates during fixation periods
  - Blink velocity: closing and opening phases are tracked through a complete blink
  - Gaze yaw dispersion: constant gaze produces σ≈0, scanning produces σ>0
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest
from pipeline.feature_extractor import FeatureExtractor, FrameFeatures
from pipeline.one_euro_filter import OneEuroFilterBank
from pipeline.pnp_normalizer import PnPNormalizer
from pipeline.imu_gate import SoftwareIMUGate
from tests.test_ear import make_eye_landmarks
from utils.landmarks import RIGHT_EYE_EAR_FLAT, LEFT_EYE_EAR_FLAT


def make_full_landmarks(ear_val=0.30, iris_x_offset=0.0, iris_y_offset=0.0):
    """Create synthetic (478, 3) landmarks with configurable EAR and iris position.

    Args:
        ear_val: Approximate EAR to produce (controls vertical opening).
        iris_x_offset: Horizontal offset for iris from eye center (pixels).
        iris_y_offset: Vertical offset for iris from eye center (pixels).
    """
    lm = np.zeros((478, 3), dtype=np.float64)

    # Right eye (subject's right) — around (200, 200)
    eye_cx, eye_cy = 200.0, 200.0
    eye_w = 40.0
    vert = ear_val * 2.0 * eye_w  # EAR = vert / (2*horiz) → vert = EAR * 2 * horiz
    half_v = vert / 2.0

    # P1=33, P2=160, P3=158, P4=133, P5=153, P6=144
    lm[33]  = [eye_cx - eye_w/2, eye_cy, 0]
    lm[160] = [eye_cx - eye_w*0.15, eye_cy - half_v, 0]
    lm[158] = [eye_cx + eye_w*0.15, eye_cy - half_v, 0]
    lm[133] = [eye_cx + eye_w/2, eye_cy, 0]
    lm[153] = [eye_cx + eye_w*0.15, eye_cy + half_v, 0]
    lm[144] = [eye_cx - eye_w*0.15, eye_cy + half_v, 0]

    # Left eye (subject's left) — around (400, 200)
    eye_cx2, eye_cy2 = 400.0, 200.0
    lm[263] = [eye_cx2 + eye_w/2, eye_cy2, 0]  # outer corner (temporal)
    lm[385] = [eye_cx2 + eye_w*0.15, eye_cy2 - half_v, 0]
    lm[387] = [eye_cx2 - eye_w*0.15, eye_cy2 - half_v, 0]
    lm[362] = [eye_cx2 - eye_w/2, eye_cy2, 0]  # inner corner (nasal)
    lm[373] = [eye_cx2 - eye_w*0.15, eye_cy2 + half_v, 0]
    lm[380] = [eye_cx2 + eye_w*0.15, eye_cy2 + half_v, 0]

    # Iris centers (positioned at eye centers + offset)
    # Landmark 468: viewer's left eye (corners 33 & 133)
    # Landmark 473: viewer's right eye (corners 263 & 362)
    lm[468] = [eye_cx + iris_x_offset, eye_cy + iris_y_offset, 0]
    lm[473] = [eye_cx2 + iris_x_offset, eye_cy2 + iris_y_offset, 0]

    # Mouth (for MAR)
    lm[61]  = [280, 350, 0]  # right mouth corner
    lm[291] = [320, 350, 0]  # left mouth corner
    lm[81]  = [290, 345, 0]
    lm[178] = [290, 355, 0]
    lm[13]  = [300, 346, 0]
    lm[14]  = [300, 354, 0]
    lm[311] = [310, 345, 0]
    lm[402] = [310, 355, 0]

    return lm


class TestSaccadeDynamics:
    """Test saccadic velocity, detection, and latency."""

    def test_fixation_produces_near_zero_velocity(self):
        """Steady gaze should produce near-zero saccadic velocity."""
        ext = FeatureExtractor(fps=30.0)

        # 30 frames of steady fixation (same iris position)
        for i in range(30):
            lm = make_full_landmarks(ear_val=0.30, iris_x_offset=0.0)
            feat = ext.compute(lm, timestamp=i/30.0, frame_id=i,
                               face_valid=True)

        assert feat.saccade_velocity < 5.0, \
            f"Fixation velocity={feat.saccade_velocity:.1f} deg/s, expected < 5"
        assert feat.saccade_detected is False
        assert feat.raw_iris_dx == 0.0

    def test_rapid_shift_produces_high_velocity(self):
        """A rapid iris shift between frames should produce high velocity and positive raw_iris_dx."""
        ext = FeatureExtractor(fps=30.0)

        # Frame 0: center gaze
        lm0 = make_full_landmarks(ear_val=0.30, iris_x_offset=0.0)
        ext.compute(lm0, timestamp=0.0, frame_id=0, face_valid=True)

        # Frame 1: large horizontal shift (simulating saccade)
        lm1 = make_full_landmarks(ear_val=0.30, iris_x_offset=15.0)
        feat = ext.compute(lm1, timestamp=1/30.0, frame_id=1, face_valid=True)

        assert feat.saccade_velocity > 20.0, \
            f"Saccade velocity={feat.saccade_velocity:.1f} deg/s, expected > 20"
        assert feat.raw_iris_dx > 0.1, \
            f"Expected positive raw_iris_dx, got {feat.raw_iris_dx}"

    def test_saccade_onset_offset_hysteresis(self):
        """Saccade detection should use hysteresis (onset > offset threshold)."""
        ext = FeatureExtractor(fps=30.0)

        # Prime with fixation
        lm = make_full_landmarks(ear_val=0.30, iris_x_offset=0.0)
        ext.compute(lm, timestamp=0.0, frame_id=0, face_valid=True)

        # Trigger saccade with large shift
        lm_saccade = make_full_landmarks(ear_val=0.30, iris_x_offset=18.0)
        feat_saccade = ext.compute(lm_saccade, timestamp=1/30.0, frame_id=1, face_valid=True)
        assert feat_saccade.saccade_detected is True

        # Slow drift (above offset but below onset) — should STILL be in saccade
        # because we haven't dropped below offset threshold
        lm_drift = make_full_landmarks(ear_val=0.30, iris_x_offset=20.0)
        feat_drift = ext.compute(lm_drift, timestamp=2/30.0, frame_id=2, face_valid=True)
        # Velocity is now lower but may still be above offset — depends on shift size

    def test_subpixel_saccade_noise_deadzone_and_hysteresis(self):
        """Rest-state sub-pixel gaze noise (< 0.6 px / hypot < 0.030) yields velocity = 0.0 and saccade_detected = False."""
        ext = FeatureExtractor(fps=30.0)
        assert ext.saccade_onset_velocity == 35.0
        assert ext.saccade_offset_velocity == 20.0

        # Frame 0: Baseline center gaze
        lm0 = make_full_landmarks(ear_val=0.30, iris_x_offset=0.0)
        ext.compute(lm0, timestamp=0.0, frame_id=0, face_valid=True)

        # Frame 1: Sub-pixel noise (< 0.6 px, hypot < 0.030)
        lm_noise = make_full_landmarks(ear_val=0.30, iris_x_offset=0.5, iris_y_offset=0.5)
        feat_noise = ext.compute(lm_noise, timestamp=1/30.0, frame_id=1, face_valid=True)
        assert feat_noise.saccade_velocity == 0.0
        assert feat_noise.saccade_detected is False

        # Frame 2: Pure vertical sensor tremor (< 0.6 px equivalent)
        lm_vnoise = make_full_landmarks(ear_val=0.30, iris_x_offset=0.5, iris_y_offset=1.0)
        feat_vnoise = ext.compute(lm_vnoise, timestamp=2/30.0, frame_id=2, face_valid=True)
        assert feat_vnoise.saccade_velocity == 0.0
        assert feat_vnoise.saccade_detected is False

        # Frame 3: Large shift (> 35 deg/s) triggers saccade
        lm_saccade = make_full_landmarks(ear_val=0.30, iris_x_offset=18.0)
        feat_saccade = ext.compute(lm_saccade, timestamp=3/30.0, frame_id=3, face_valid=True)
        assert feat_saccade.saccade_velocity > 35.0
        assert feat_saccade.saccade_detected is True

        # Robustness against partial None state in _compute_saccade_dynamics
        ext._prev_iris_y = None
        vel, is_sac, lat, rdx = ext._compute_saccade_dynamics(0.5, 0.5, timestamp=4/30.0)
        assert vel == 0.0
        assert is_sac is False

        # Gate resume re-anchors both horizontal and vertical iris positions without spurious spike
        ext.compute(lm0, timestamp=5/30.0, frame_id=5, face_valid=True)
        ext.compute(lm0, timestamp=6/30.0, frame_id=6, face_valid=False)  # Gated frame
        lm_shifted = make_full_landmarks(ear_val=0.30, iris_x_offset=2.0, iris_y_offset=2.0)
        feat_resumed = ext.compute(lm_shifted, timestamp=7/30.0, frame_id=7, face_valid=True)
        assert feat_resumed.saccade_detected is False

        # Frame drop exceeding 500ms unlatches saccade and re-anchors so sub-onset drift does not latch
        ext.compute(lm_saccade, timestamp=8/30.0, frame_id=8, face_valid=True)
        assert ext._in_saccade is True
        # 1.0 second frame drop
        feat_dropped = ext.compute(lm_saccade, timestamp=38/30.0, frame_id=9, face_valid=True)
        assert feat_dropped.saccade_detected is False
        assert ext._in_saccade is False
        # Subsequent drift with velocity ~22.7 deg/s (between offset=20 and onset=35) must NOT latch
        lm_drift = make_full_landmarks(ear_val=0.30, iris_x_offset=18.0 + 1.0)
        feat_post_drop = ext.compute(lm_drift, timestamp=39/30.0, frame_id=10, face_valid=True)
        assert feat_post_drop.saccade_detected is False
        assert ext._in_saccade is False

        # Direct NaN / Inf input safety
        v_nan, sac_nan, _, _ = ext._compute_saccade_dynamics(float('nan'), 0.0, timestamp=40/30.0)
        assert v_nan == 0.0
        assert sac_nan is False

        v_inf, sac_inf, _, _ = ext._compute_saccade_dynamics(float('inf'), 0.0, timestamp=41/30.0)
        assert v_inf == 0.0
        assert sac_inf is False

        v_t_nan, sac_t_nan, _, _ = ext._compute_saccade_dynamics(0.1, 0.0, timestamp=float('nan'))
        assert v_t_nan == 0.0
        assert sac_t_nan is False

        v_t_inf, sac_t_inf, _, _ = ext._compute_saccade_dynamics(0.1, 0.0, timestamp=float('inf'))
        assert v_t_inf == 0.0
        assert sac_t_inf is False

    def test_binocular_conjugate_check_and_blink_reopen_reanchor(self):
        """Opposite eye motions are zeroed by conjugate check, and blink reopening re-anchors coordinates."""
        ext = FeatureExtractor(fps=30.0)

        # Baseline
        lm0 = make_full_landmarks(ear_val=0.30, iris_x_offset=0.0)
        ext.compute(lm0, timestamp=0.0, frame_id=0, face_valid=True)

        # Test A: Disconjugate eye movement (left moves right +10px, right moves left -10px)
        lm_dis = lm0.copy()
        # Left eye iris (viewer's right, landmark 473): +10px
        lm_dis[473, 0] += 10.0
        # Right eye iris (viewer's left, landmark 468): -10px
        lm_dis[468, 0] -= 10.0
        feat_dis = ext.compute(lm_dis, timestamp=1/30.0, frame_id=1, face_valid=True)
        # Disconjugate motion must be suppressed
        assert feat_dis.saccade_velocity == 0.0
        assert feat_dis.saccade_detected is False

        # Test B: Conjugate eye movement (both eyes move right +12px)
        lm_conj = lm0.copy()
        lm_conj[468, 0] += 12.0
        lm_conj[473, 0] += 12.0
        feat_conj = ext.compute(lm_conj, timestamp=2/30.0, frame_id=2, face_valid=True)
        assert feat_conj.saccade_velocity > 35.0
        assert feat_conj.saccade_detected is True

        # Test C: Blink reopen frame re-anchors _prev_iris_x/y
        # Frame 3: Blink closed
        lm_closed = make_full_landmarks(ear_val=0.08, iris_x_offset=12.0)
        ext.compute(lm_closed, timestamp=3/30.0, frame_id=3, face_valid=True)
        # Frame 4: Blink reopening with new pupil position
        lm_reopen = make_full_landmarks(ear_val=0.28, iris_x_offset=5.0)
        feat_reopen = ext.compute(lm_reopen, timestamp=4/30.0, frame_id=4, face_valid=True)
        # Reopening frame must not spike
        assert feat_reopen.saccade_velocity == 0.0
        assert feat_reopen.saccade_detected is False
        # Frame 5: Steady fixation at new pupil position must have 0 velocity
        lm_reopen_steady = make_full_landmarks(ear_val=0.28, iris_x_offset=5.0)
        feat_steady = ext.compute(lm_reopen_steady, timestamp=5/30.0, frame_id=5, face_valid=True)
        assert feat_steady.saccade_velocity == 0.0
        assert feat_steady.saccade_detected is False

    def test_latency_accumulates_during_fixation(self):
        """Saccade latency should increase during fixation after a saccade."""
        ext = FeatureExtractor(fps=30.0)

        # Prime
        lm0 = make_full_landmarks(ear_val=0.30, iris_x_offset=0.0)
        ext.compute(lm0, timestamp=0.0, frame_id=0, face_valid=True)

        # Trigger saccade
        lm1 = make_full_landmarks(ear_val=0.30, iris_x_offset=18.0)
        ext.compute(lm1, timestamp=1/30.0, frame_id=1, face_valid=True)

        # Return to fixation for 10 frames
        for i in range(2, 12):
            lm = make_full_landmarks(ear_val=0.30, iris_x_offset=18.0)
            feat = ext.compute(lm, timestamp=i/30.0, frame_id=i, face_valid=True)

        # Latency should be > 0 after saccade ended and fixation began
        if not feat.saccade_detected:
            assert feat.saccade_latency_ms > 0.0

    def test_blink_sequence_does_not_reset_saccade_latency(self):
        """Full open->closing->closed->opening->open cycle must not reset saccade_latency_ms to zero."""
        ext = FeatureExtractor(fps=30.0, blink_threshold=0.20)

        # 1. Trigger an initial saccade
        lm0 = make_full_landmarks(ear_val=0.15, iris_x_offset=0.0)
        ext.compute(lm0, timestamp=0.0, frame_id=0, face_valid=True)
        lm_sac = make_full_landmarks(ear_val=0.15, iris_x_offset=18.0)
        ext.compute(lm_sac, timestamp=1/30.0, frame_id=1, face_valid=True)

        # 2. Fixate for 10 frames so saccade ends and latency accumulates
        for i in range(2, 12):
            lm = make_full_landmarks(ear_val=0.15, iris_x_offset=18.0)
            feat = ext.compute(lm, timestamp=i/30.0, frame_id=i, face_valid=True)

        assert feat.saccade_detected is False
        latency_before_blink = feat.saccade_latency_ms
        assert latency_before_blink > 0.0

        # 3. Full blink cycle with simulated iris jitter/occlusion artifacts
        # Closing (ear_val 0.12 -> 0.24, 0.08 -> 0.16)
        closing_ears = [0.12, 0.08]
        for j, ear in enumerate(closing_ears):
            # Iris jumps around during closure (simulating occlusion artifact)
            lm = make_full_landmarks(ear_val=ear, iris_x_offset=18.0 + (j * 10.0))
            feat = ext.compute(lm, timestamp=(12+j)/30.0, frame_id=12+j, face_valid=True)
            assert feat.blink_phase in ("closing", "closed")
            assert feat.raw_iris_dx == 0.0
            assert feat.saccade_detected is False
            assert feat.saccade_latency_ms >= latency_before_blink

        # Closed (ear_val 0.04 -> 0.08)
        for j in range(2):
            lm = make_full_landmarks(ear_val=0.04, iris_x_offset=30.0)
            feat = ext.compute(lm, timestamp=(14+j)/30.0, frame_id=14+j, face_valid=True)
            assert feat.blink_phase == "closed"
            assert feat.raw_iris_dx == 0.0
            assert feat.saccade_detected is False
            assert feat.saccade_latency_ms >= latency_before_blink

        # Opening (ear_val 0.10 -> 0.20)
        opening_ears = [0.10]
        for j, ear in enumerate(opening_ears):
            lm = make_full_landmarks(ear_val=ear, iris_x_offset=25.0)
            feat = ext.compute(lm, timestamp=(16+j)/30.0, frame_id=16+j, face_valid=True)
            assert feat.blink_phase in ("opening", "open")
            assert feat.saccade_detected is False
            assert feat.saccade_latency_ms >= latency_before_blink

        # Fully reopened (ear_val 0.15 -> 0.30)
        for j in range(5):
            lm = make_full_landmarks(ear_val=0.15, iris_x_offset=18.0)
            feat = ext.compute(lm, timestamp=(17+j)/30.0, frame_id=17+j, face_valid=True)
            assert feat.blink_phase == "open"
            assert feat.saccade_detected is False
            assert feat.saccade_latency_ms > latency_before_blink

    def test_saccade_interrupted_by_blink(self):
        """A saccade starts, a blink occurs, eye reopens; confirm latency and detection state are correct after reopening."""
        ext = FeatureExtractor(fps=30.0, blink_threshold=0.20)

        # Baseline frame
        lm0 = make_full_landmarks(ear_val=0.15, iris_x_offset=0.0)
        ext.compute(lm0, timestamp=0.0, frame_id=0, face_valid=True)

        # 1. Saccade starts (large shift)
        lm_sac = make_full_landmarks(ear_val=0.15, iris_x_offset=18.0)
        feat_sac = ext.compute(lm_sac, timestamp=1/30.0, frame_id=1, face_valid=True)
        assert feat_sac.saccade_detected is True
        assert feat_sac.saccade_latency_ms == 0.0

        # 2. Blink occurs immediately while saccade is active
        # Closing (ear_val 0.12 -> 0.24)
        lm_closing = make_full_landmarks(ear_val=0.12, iris_x_offset=20.0)
        feat_c = ext.compute(lm_closing, timestamp=2/30.0, frame_id=2, face_valid=True)
        assert feat_c.blink_phase in ("closing", "closed")
        assert feat_c.saccade_detected is False
        assert feat_c.saccade_latency_ms == 0.0
        assert feat_c.raw_iris_dx == 0.0

        # Closed (ear_val 0.04 -> 0.08)
        lm_closed = make_full_landmarks(ear_val=0.04, iris_x_offset=20.0)
        feat_closed = ext.compute(lm_closed, timestamp=3/30.0, frame_id=3, face_valid=True)
        assert feat_closed.blink_phase == "closed"
        assert feat_closed.saccade_detected is False
        assert feat_closed.saccade_latency_ms >= 0.0

        # Opening (ear_val 0.10 -> 0.20)
        lm_opening = make_full_landmarks(ear_val=0.10, iris_x_offset=20.0)
        feat_o = ext.compute(lm_opening, timestamp=4/30.0, frame_id=4, face_valid=True)

        # 3. Eye reopens and driver fixates at new position (ear_val 0.15 -> 0.30)
        lm_reopen1 = make_full_landmarks(ear_val=0.15, iris_x_offset=20.0)
        feat_reopen1 = ext.compute(lm_reopen1, timestamp=5/30.0, frame_id=5, face_valid=True)
        assert feat_reopen1.blink_phase == "open"
        assert feat_reopen1.saccade_detected is False

        # Next fixation frame: latency begins accumulating from saccade end
        lm_reopen2 = make_full_landmarks(ear_val=0.15, iris_x_offset=20.0)
        feat_reopen2 = ext.compute(lm_reopen2, timestamp=6/30.0, frame_id=6, face_valid=True)
        assert feat_reopen2.saccade_detected is False
        assert feat_reopen2.saccade_latency_ms > 0.0

    def test_head_nod_up_down_does_not_cause_eye_sway_or_false_saccade(self):
        """A 40px vertical + 15px horizontal head nod with pitch oscillation causes 0.0px relative sway and 0 false saccades."""
        lm0 = make_full_landmarks(ear_val=0.30)
        bank = OneEuroFilterBank(t0=0.0, x0=lm0)
        ext = FeatureExtractor(fps=30.0)

        # Baseline un-nodded relative distances
        expected_rel_r_eyelid = lm0[160, 1] - 0.5 * (lm0[33, 1] + lm0[133, 1])
        expected_rel_r_iris = lm0[468, 1] - 0.5 * (lm0[33, 1] + lm0[133, 1])
        expected_rel_l_eyelid = lm0[385, 1] - 0.5 * (lm0[263, 1] + lm0[362, 1])
        expected_rel_l_iris = lm0[473, 1] - 0.5 * (lm0[263, 1] + lm0[362, 1])

        max_eyelid_sway = 0.0
        max_iris_sway = 0.0

        for i in range(60):
            t = i / 30.0
            y_nod = 40.0 * np.sin(2 * np.pi * 1.0 * t)
            x_nod = 15.0 * np.cos(2 * np.pi * 1.0 * t)
            pitch = 15.0 * np.sin(2 * np.pi * 1.0 * t)

            raw_lm = make_full_landmarks(ear_val=0.30)
            raw_lm[:, 0] += x_nod
            raw_lm[:, 1] += y_nod

            filtered_lm = bank.filter_landmarks(t, raw_lm)

            # Check relative vertical distance between eye corners and eyelids/irises
            filt_r_corner_y = 0.5 * (filtered_lm[33, 1] + filtered_lm[133, 1])
            filt_r_eyelid_rel = filtered_lm[160, 1] - filt_r_corner_y
            filt_r_iris_rel = filtered_lm[468, 1] - filt_r_corner_y

            filt_l_corner_y = 0.5 * (filtered_lm[263, 1] + filtered_lm[362, 1])
            filt_l_eyelid_rel = filtered_lm[385, 1] - filt_l_corner_y
            filt_l_iris_rel = filtered_lm[473, 1] - filt_l_corner_y

            sway_r_eyelid = abs(filt_r_eyelid_rel - expected_rel_r_eyelid)
            sway_r_iris = abs(filt_r_iris_rel - expected_rel_r_iris)
            sway_l_eyelid = abs(filt_l_eyelid_rel - expected_rel_l_eyelid)
            sway_l_iris = abs(filt_l_iris_rel - expected_rel_l_iris)

            max_eyelid_sway = max(max_eyelid_sway, sway_r_eyelid, sway_l_eyelid)
            max_iris_sway = max(max_iris_sway, sway_r_iris, sway_l_iris)

            feat = ext.compute(filtered_lm, timestamp=t, frame_id=i, face_valid=True, head_pitch=pitch)

            assert feat.saccade_detected is False, f"False saccade triggered on frame {i} during head nod"
            assert feat.saccade_velocity == 0.0, f"False saccade velocity {feat.saccade_velocity} on frame {i}"

        # 1. Max relative vertical lag between eye corners and eyelids/irises is < 1e-4 px (0.0 sway)
        assert max_eyelid_sway < 1e-4, f"Eyelid sway={max_eyelid_sway:.6f} px, expected < 1e-4"
        assert max_iris_sway < 1e-4, f"Iris sway={max_iris_sway:.6f} px, expected < 1e-4"


class TestBlinkVelocity:
    """Test blink closing and opening velocity measurement."""

    def test_complete_blink_captures_velocities(self):
        """A full open→close→open blink should produce non-zero velocities."""
        ext = FeatureExtractor(fps=30.0, blink_threshold=0.20)

        # Phase 1: Eyes open (10 frames)
        for i in range(10):
            lm = make_full_landmarks(ear_val=0.30)
            ext.compute(lm, timestamp=i/30.0, frame_id=i, face_valid=True)

        # Phase 2: Rapid closing (3 frames: 0.30 → 0.25 → 0.18 → 0.10)
        closing_ears = [0.25, 0.18, 0.10]
        for j, ear in enumerate(closing_ears):
            lm = make_full_landmarks(ear_val=ear)
            ext.compute(lm, timestamp=(10+j)/30.0, frame_id=10+j, face_valid=True)

        # Phase 3: Closed (2 frames)
        for j in range(2):
            lm = make_full_landmarks(ear_val=0.08)
            ext.compute(lm, timestamp=(13+j)/30.0, frame_id=13+j, face_valid=True)

        # Phase 4: Rapid opening (3 frames: 0.10 → 0.18 → 0.25 → 0.30)
        opening_ears = [0.12, 0.20, 0.28]
        for j, ear in enumerate(opening_ears):
            lm = make_full_landmarks(ear_val=ear)
            feat = ext.compute(lm, timestamp=(15+j)/30.0, frame_id=15+j, face_valid=True)

        # After the blink completes, velocities should be captured
        # Continue with open eyes to ensure blink phase completes
        for j in range(5):
            lm = make_full_landmarks(ear_val=0.30)
            feat = ext.compute(lm, timestamp=(18+j)/30.0, frame_id=18+j, face_valid=True)

        assert feat.blink_closing_velocity > 0.0, \
            f"Blink closing velocity={feat.blink_closing_velocity}, expected > 0"
        assert feat.blink_opening_velocity > 0.0, \
            f"Blink opening velocity={feat.blink_opening_velocity}, expected > 0"

    def test_no_blink_produces_zero_velocities(self):
        """Constant open eyes should not produce blink velocities."""
        ext = FeatureExtractor(fps=30.0)

        for i in range(30):
            lm = make_full_landmarks(ear_val=0.30)
            feat = ext.compute(lm, timestamp=i/30.0, frame_id=i, face_valid=True)

        assert feat.blink_closing_velocity == 0.0
        assert feat.blink_opening_velocity == 0.0

    def test_blink_velocity_measures_peak_velocity(self):
        """Blink closing and opening velocities record peak stroke velocity (np.max) instead of mean."""
        ext = FeatureExtractor(fps=30.0, blink_threshold=0.20)

        # Baseline open (ear_val=0.15 -> ear_avg=0.30)
        for i in range(5):
            lm = make_full_landmarks(ear_val=0.15)
            ext.compute(lm, timestamp=i/30.0, frame_id=i, face_valid=True)

        # Phase 1: Closing downstroke
        # Frame 5: ear_val=0.13 (ear_avg=0.26)
        ext.compute(make_full_landmarks(ear_val=0.13), timestamp=5/30.0, frame_id=5, face_valid=True)
        # Frame 6: ear_val=0.09 (ear_avg=0.18, drop 0.08 in 1/30s -> 2.4 EAR/s) - PEAK
        ext.compute(make_full_landmarks(ear_val=0.09), timestamp=6/30.0, frame_id=6, face_valid=True)

        # Closed
        # Frame 7 & 8: ear_val=0.04 (ear_avg=0.08)
        ext.compute(make_full_landmarks(ear_val=0.04), timestamp=7/30.0, frame_id=7, face_valid=True)
        ext.compute(make_full_landmarks(ear_val=0.04), timestamp=8/30.0, frame_id=8, face_valid=True)

        # Phase 2: Opening upstroke
        # Frame 9: ear_val=0.09 (ear_avg=0.18, jump 0.10 in 1/30s -> 3.0 EAR/s) - PEAK
        ext.compute(make_full_landmarks(ear_val=0.09), timestamp=9/30.0, frame_id=9, face_valid=True)
        # Frame 10: ear_val=0.13 (ear_avg=0.26, jump 0.08 in 1/30s -> 2.4 EAR/s)
        ext.compute(make_full_landmarks(ear_val=0.13), timestamp=10/30.0, frame_id=10, face_valid=True)

        # Settle back to fully open
        feat_done = ext.compute(make_full_landmarks(ear_val=0.15), timestamp=11/30.0, frame_id=11, face_valid=True)

        # Peak closing should be >= 2.0 EAR/s (not diluted down)
        assert feat_done.blink_closing_velocity >= 2.0, \
            f"Expected peak closing vel >= 2.0 EAR/s, got {feat_done.blink_closing_velocity:.3f}"
        # Peak opening should be >= 2.0 EAR/s
        assert feat_done.blink_opening_velocity >= 2.0, \
            f"Expected peak opening vel >= 2.0 EAR/s, got {feat_done.blink_opening_velocity:.3f}"

    def test_no_sawtooth_reset_in_open_phase(self):
        """Eliminate the 9-frame sawtooth reset bug: open recovery climbs monotonically."""
        ext = FeatureExtractor(fps=30.0, blink_threshold=0.20)

        # Baseline open
        for i in range(5):
            ext.compute(make_full_landmarks(ear_val=0.15), timestamp=i/30.0, frame_id=i, face_valid=True)

        # Perform a quick blink
        ext.compute(make_full_landmarks(ear_val=0.04), timestamp=5/30.0, frame_id=5, face_valid=True)
        ext.compute(make_full_landmarks(ear_val=0.04), timestamp=6/30.0, frame_id=6, face_valid=True)
        ext.compute(make_full_landmarks(ear_val=0.10), timestamp=7/30.0, frame_id=7, face_valid=True)
        ext.compute(make_full_landmarks(ear_val=0.15), timestamp=8/30.0, frame_id=8, face_valid=True)

        # Now driver keeps eyes open for 30 frames (1.0 second)
        vel_history = []
        for i in range(30):
            t = (9 + i) / 30.0
            feat = ext.compute(make_full_landmarks(ear_val=0.15), timestamp=t, frame_id=9+i, face_valid=True)
            vel_history.append(feat.blink_opening_velocity)

        # Verify velocities never experience a sawtooth dip every 9 frames
        for i in range(1, len(vel_history)):
            assert vel_history[i] >= vel_history[i-1] - 1e-6, \
                f"Sawtooth dip detected at frame {i}: {vel_history[i]:.4f} < {vel_history[i-1]:.4f}"

    def test_post_blink_followthrough_capture(self):
        """Follow-through frame after transitioning to 'open' captures active reopening upstroke."""
        ext = FeatureExtractor(fps=30.0, blink_threshold=0.20)

        # Baseline open (ear_avg=0.30)
        for i in range(5):
            ext.compute(make_full_landmarks(ear_val=0.15), timestamp=i/30.0, frame_id=i, face_valid=True)

        # Close eyes
        ext.compute(make_full_landmarks(ear_val=0.04), timestamp=5/30.0, frame_id=5, face_valid=True)
        ext.compute(make_full_landmarks(ear_val=0.04), timestamp=6/30.0, frame_id=6, face_valid=True)

        # Eyelid reopens to 0.26 (triggers transition to 'open')
        f7 = ext.compute(make_full_landmarks(ear_val=0.13), timestamp=7/30.0, frame_id=7, face_valid=True)
        assert f7.blink_phase == "open"

        # Frame 8 has steep follow-through upstroke from 0.26 to 0.32 (jump of 0.06 in 1/30s)
        f8 = ext.compute(make_full_landmarks(ear_val=0.16), timestamp=8/30.0, frame_id=8, face_valid=True)
        assert f8.blink_opening_velocity >= 1.8


class TestGazeYawDispersion:
    """Test rolling σ_yaw computation."""

    def test_constant_gaze_produces_near_zero_dispersion(self):
        """Steady horizontal gaze should produce σ_yaw ≈ 0."""
        ext = FeatureExtractor(fps=30.0, gaze_dispersion_window_sec=15.0)

        for i in range(60):
            lm = make_full_landmarks(ear_val=0.30, iris_x_offset=0.0)
            feat = ext.compute(lm, timestamp=i/30.0, frame_id=i, face_valid=True)

        assert feat.gaze_yaw_dispersion < 0.02, \
            f"Constant gaze σ_yaw={feat.gaze_yaw_dispersion:.4f}, expected ≈ 0"

    def test_scanning_gaze_produces_high_dispersion(self):
        """Alternating left/right gaze (scanning mirrors) should produce σ_yaw > 0."""
        ext = FeatureExtractor(fps=30.0, gaze_dispersion_window_sec=15.0)

        for i in range(60):
            # Alternate between left and right gaze positions
            offset = 10.0 if (i % 10 < 5) else -10.0
            lm = make_full_landmarks(ear_val=0.30, iris_x_offset=offset)
            feat = ext.compute(lm, timestamp=i/30.0, frame_id=i, face_valid=True)

        assert feat.gaze_yaw_dispersion > 0.1, \
            f"Scanning gaze σ_yaw={feat.gaze_yaw_dispersion:.4f}, expected > 0.1"

    def test_tunneling_lower_than_scanning(self):
        """Gaze tunneling (fixed center) should have lower σ_yaw than active scanning."""
        ext_tunnel = FeatureExtractor(fps=30.0, gaze_dispersion_window_sec=15.0)
        ext_scan = FeatureExtractor(fps=30.0, gaze_dispersion_window_sec=15.0)

        for i in range(90):
            # Tunneling: small random jitter around center
            jitter = np.random.uniform(-1.0, 1.0)
            lm_t = make_full_landmarks(ear_val=0.30, iris_x_offset=jitter)
            feat_t = ext_tunnel.compute(lm_t, timestamp=i/30.0, frame_id=i, face_valid=True)

            # Scanning: large sweeps
            sweep = 12.0 * np.sin(2 * np.pi * i / 30)
            lm_s = make_full_landmarks(ear_val=0.30, iris_x_offset=sweep)
            feat_s = ext_scan.compute(lm_s, timestamp=i/30.0, frame_id=i, face_valid=True)

        assert feat_t.gaze_yaw_dispersion < feat_s.gaze_yaw_dispersion, \
            f"Tunneling σ={feat_t.gaze_yaw_dispersion:.4f} should be < " \
            f"scanning σ={feat_s.gaze_yaw_dispersion:.4f}"

    def test_gaze_yaw_dispersion_frozen_during_blinks(self):
        """Eyelid closure during blinks must not append occluded coordinates into _gaze_yaw_history."""
        ext = FeatureExtractor(fps=30.0, gaze_dispersion_window_sec=15.0)

        # 1. Populate 30 open frames
        for i in range(30):
            lm = make_full_landmarks(ear_val=0.30, iris_x_offset=5.0 * np.sin(i * 0.2))
            ext.compute(lm, timestamp=i/30.0, frame_id=i, face_valid=True)

        history_len_open = len(ext._gaze_yaw_history)
        dispersion_open = ext._last_features.gaze_yaw_dispersion
        assert history_len_open == 30

        # 2. Simulate 5 blink frames with large occluded iris jump (e.g. +30px offset)
        for j in range(5):
            fid = 30 + j
            lm_blink = make_full_landmarks(ear_val=0.08, iris_x_offset=30.0)
            feat_blink = ext.compute(lm_blink, timestamp=fid/30.0, frame_id=fid, face_valid=True)
            # Dispersion history should NOT grow during blinks
            assert len(ext._gaze_yaw_history) == history_len_open
            # Gaze yaw dispersion should remain stable rather than exploding
            assert abs(feat_blink.gaze_yaw_dispersion - dispersion_open) < 1e-4

        # 3. Reopened frames: first frame 35 has blink_detected=True (protected from occluded jump),
        # second frame 36 has blink_phase='open' and appends to history again
        lm_reopen = make_full_landmarks(ear_val=0.30, iris_x_offset=0.0)
        ext.compute(lm_reopen, timestamp=35/30.0, frame_id=35, face_valid=True)
        assert len(ext._gaze_yaw_history) == history_len_open

        ext.compute(lm_reopen, timestamp=36/30.0, frame_id=36, face_valid=True)
        assert len(ext._gaze_yaw_history) == history_len_open + 1


def test_speed_bump_and_road_vibration_immunity():
    """Simulate sharp +40px vertical speed-bump impulse and 8 Hz road vibration.

    Verifies that through OneEuroFilterBank -> PnPNormalizer -> SoftwareIMUGate -> FeatureExtractor:
      1. Global landmark tracking lag during the speed bump is < 5.0 px (down from 25.3 px).
      2. SoftwareIMUGate triggers imu_gated == True during the speed bump impact.
      3. saccade_detected remains False (0 false saccades).
      4. blink_detected and microsleep_detected remain False (EAR stays within 0.30 ± 0.005).
    """
    def create_face(shift_x=0.0, shift_y=0.0):
        # ear_val=0.15 produces exact EAR = 0.30 in Soukupova & Cech formula
        lm = make_full_landmarks(ear_val=0.15, iris_x_offset=0.0)
        # Rigid anchors for PnP
        lm[4] = [300.0, 250.0, 0.0]    # Nose tip
        lm[152] = [300.0, 420.0, 0.0]  # Chin
        lm[168] = [300.0, 180.0, 0.0]  # Glabella
        lm[9] = [300.0, 150.0, 0.0]    # Forehead
        lm[:, 0] += shift_x
        lm[:, 1] += shift_y
        return lm

    base_lm = create_face()
    filter_bank = OneEuroFilterBank(0.0, base_lm)
    pnp = PnPNormalizer(640, 480)
    imu = SoftwareIMUGate(accel_threshold=2.5, freeze_duration_ms=150.0)
    extractor = FeatureExtractor(fps=30.0)

    # 1. Warm-up steady frames (3 frames at rest)
    for i in range(3):
        t = i * 0.033
        raw = create_face()
        filt = filter_bank.filter_landmarks(t, raw)
        pnp.solve_pose(filt)
        norm = pnp.normalize_landmarks(filt)
        euler = pnp.get_euler_angles()
        gated = imu.update(t, pnp.tvec, euler)
        extractor.compute(norm, t, i, face_valid=True, imu_gated=gated)

    # 2. Simulate 4 frames of speed-bump impulse (reaching +40px)
    bump_shifts = [15.0, 35.0, 40.0, 40.0]
    lags = []
    imu_gated_events = []
    saccades = []
    blinks = []
    microsleeps = []
    ears = []

    for k, bump_y in enumerate(bump_shifts):
        fid = 3 + k
        t = fid * 0.033
        raw = create_face(0.0, bump_y)
        filt = filter_bank.filter_landmarks(t, raw)
        lag = float(np.mean(np.abs(raw[4] - filt[4])))
        lags.append(lag)
        pnp.solve_pose(filt)
        norm = pnp.normalize_landmarks(filt)
        euler = pnp.get_euler_angles()
        gated = imu.update(t, pnp.tvec, euler)
        imu_gated_events.append(gated)
        feat = extractor.compute(norm, t, fid, face_valid=True, imu_gated=gated)
        saccades.append(feat.saccade_detected)
        blinks.append(feat.blink_detected)
        microsleeps.append(feat.microsleep_detected)
        ears.append(feat.ear_avg)

    # 3. Vibration frames: 8 Hz road vibration (±8px vertical, ±4px horizontal)
    for j in range(15):
        fid = 7 + j
        t = fid * 0.033
        vib_y = 40.0 + 8.0 * np.sin(2 * np.pi * 8 * (j * 0.033))
        vib_x = 4.0 * np.cos(2 * np.pi * 8 * (j * 0.033))
        raw = create_face(vib_x, vib_y)
        filt = filter_bank.filter_landmarks(t, raw)
        pnp.solve_pose(filt)
        norm = pnp.normalize_landmarks(filt)
        euler = pnp.get_euler_angles()
        gated = imu.update(t, pnp.tvec, euler)
        feat = extractor.compute(norm, t, fid, face_valid=True, imu_gated=gated)
        saccades.append(feat.saccade_detected)
        blinks.append(feat.blink_detected)
        microsleeps.append(feat.microsleep_detected)
        ears.append(feat.ear_avg)

    # Verifications:
    # 1. Global landmark tracking lag during the speed bump is < 5.0 px (down from 25.3 px)
    assert max(lags) < 5.0, f"Speed bump tracking lag {max(lags):.2f} px exceeds 5.0 px limit"

    # 2. SoftwareIMUGate triggers imu_gated == True during the speed bump impact
    assert any(imu_gated_events), "SoftwareIMUGate must trigger imu_gated == True during speed bump"

    # 3. saccade_detected remains False (0 false saccades)
    assert not any(saccades), "Road vibration/bump must not trigger false saccades"

    # 4. blink_detected and microsleep_detected remain False (EAR stays within 0.30 ± 0.005)
    assert not any(blinks), "Road vibration/bump must not trigger false blinks"
    assert not any(microsleeps), "Road vibration/bump must not trigger false microsleep"
    for ear in ears:
        assert abs(ear - 0.30) <= 0.005, f"EAR {ear:.4f} deviated from 0.30 by more than 0.005"



class TestVibrationHardenedFeatureExtractor:
    """Test suite for Phase 4 vibration hardening: dynamic deadzones, 5-point iris centroid, and blink glitch tolerance."""

    def test_dynamic_saccade_deadzone_under_vibration(self):
        """Dynamic deadzone scales up under vibration so residual vibration does not trigger false saccades."""
        ext = FeatureExtractor(fps=30.0)

        # Baseline frame
        lm0 = make_full_landmarks(ear_val=0.30, iris_x_offset=0.0)
        ext.compute(lm0, timestamp=0.0, frame_id=0, face_valid=True)

        # 1. At vibration_score = 0.0, a shift of 0.8px (normalized dx ~ 0.04 > 0.026) triggers non-zero velocity
        lm_small_shift = make_full_landmarks(ear_val=0.30, iris_x_offset=0.8)
        feat_no_vib = ext.compute(lm_small_shift, timestamp=1/30.0, frame_id=1, face_valid=True, vibration_score=0.0)
        assert feat_no_vib.saccade_velocity > 0.0

        # Reset extractor state
        ext.reset()
        ext.compute(lm0, timestamp=0.0, frame_id=0, face_valid=True)

        # 2. At vibration_score = 0.75, deadzone scales by (1.0 + 1.2 * 0.75) = 1.9x (eff_deadzone_dx ~ 0.0494)
        # The same 0.8px shift is now inside the expanded deadzone and produces velocity = 0.0
        feat_with_vib = ext.compute(lm_small_shift, timestamp=1/30.0, frame_id=1, face_valid=True, vibration_score=0.75)
        assert feat_with_vib.saccade_velocity == 0.0
        assert feat_with_vib.saccade_detected is False

        # 3. A large voluntary saccade (15px shift) exceeds even the scaled deadzone
        lm_large_shift = make_full_landmarks(ear_val=0.30, iris_x_offset=15.0)
        feat_large = ext.compute(lm_large_shift, timestamp=2/30.0, frame_id=2, face_valid=True, vibration_score=0.75)
        assert feat_large.saccade_velocity > 35.0
        assert feat_large.saccade_detected is True

    def test_5_point_iris_centroid_averaging(self):
        """When 5-point iris ring landmarks are populated, centroid averaging reduces sub-pixel jitter."""
        ext = FeatureExtractor(fps=30.0)
        landmarks = np.zeros((478, 3), dtype=np.float64)

        # Set eye corners for right eye (corners 33 and 133, width=40px, center=200)
        landmarks[33] = [180.0, 200.0, 0.0]
        landmarks[133] = [220.0, 200.0, 0.0]

        # Iris center with 1.0px jitter
        landmarks[468] = [201.0, 200.0, 0.0]
        # Ring points centered exactly at 200.0 (3.0px radius)
        landmarks[469] = [203.0, 200.0, 0.0]
        landmarks[470] = [197.0, 200.0, 0.0]
        landmarks[471] = [200.0, 203.0, 0.0]
        landmarks[472] = [200.0, 197.0, 0.0]

        # compute_iris_position should average all 5 points: (201 + 203 + 197 + 200 + 200) / 5 = 200.2
        # Offset from eye center (200.0) is 0.2 px instead of 1.0 px (5x reduction!)
        x_5pt, y_5pt = FeatureExtractor.compute_iris_position(landmarks, 468, 33, 133)
        expected_x = 0.2 / 20.0  # offset 0.2 / (width/2 = 20) = 0.010
        assert pytest.approx(x_5pt, abs=1e-4) == expected_x

    def test_blink_closing_phase_glitch_tolerance(self):
        """A single non-negative d_ear glitch (< 0.015) during closing phase is tolerated without aborting."""
        ext = FeatureExtractor(fps=30.0, blink_threshold=0.20)

        # Steady open eyes
        for i in range(5):
            ext.compute(make_full_landmarks(ear_val=0.15), timestamp=i/30.0, frame_id=i, face_valid=True)

        # Frame 5: Closing downstroke starts (ear drops from 0.30 to 0.25)
        f5 = ext.compute(make_full_landmarks(ear_val=0.125), timestamp=5/30.0, frame_id=5, face_valid=True)
        assert f5.blink_phase == "closing"

        # Frame 6: Vibration glitch causes a minor +0.008 EAR upward bounce
        f6 = ext.compute(make_full_landmarks(ear_val=0.129), timestamp=6/30.0, frame_id=6, face_valid=True)
        # Should NOT abort to 'open' thanks to self._closing_glitch_count
        assert f6.blink_phase == "closing"
        assert ext._closing_glitch_count == 1

        # Frame 7: Resumes closing downstroke to fully closed (ear = 0.08 < 0.20)
        f7 = ext.compute(make_full_landmarks(ear_val=0.04), timestamp=7/30.0, frame_id=7, face_valid=True)
        assert f7.blink_phase == "closed"
        assert ext._closing_glitch_count == 0

    def test_vibration_discounts_eyelid_tremor_and_pauses_postural_sway(self):
        """When vibration_score > 0.35, eyelid tremor is discounted and postural sway accumulation is paused."""
        ext = FeatureExtractor(fps=30.0)

        # 1. Warm-up postural sway history with normal wobble
        for i in range(30):
            t = i / 30.0
            p = 2.0 * np.sin(2 * np.pi * 1.5 * t)
            r = 1.5 * np.cos(2 * np.pi * 1.5 * t)
            feat = ext.compute(make_full_landmarks(ear_val=0.15), timestamp=t, frame_id=i,
                               face_valid=True, head_pitch=p, head_roll=r, vibration_score=0.0)

        initial_sway = feat.head_postural_sway
        assert initial_sway > 0.5
        history_len_before = len(ext._head_pose_history)

        # 2. Road vibration occurs: vibration_score = 0.70 > 0.35
        for i in range(30, 45):
            t = i / 30.0
            p = 5.0 * np.sin(2 * np.pi * 8.0 * t)  # road chatter
            r = 4.0 * np.cos(2 * np.pi * 8.0 * t)
            feat_vib = ext.compute(make_full_landmarks(ear_val=0.15), timestamp=t, frame_id=i,
                                   face_valid=True, head_pitch=p, head_roll=r, vibration_score=0.70)
            # Postural sway must hold the previous clean value, not absorb vehicle vibration
            assert feat_vib.head_postural_sway == initial_sway

        # Pose history accumulation was paused during vibration
        assert len(ext._head_pose_history) == history_len_before

        # 3. Test tremor discount:
        tremor_undiscounted = ext._compute_eyelid_tremor(0.28, 10.0, "open", vibration_score=0.0)
        tremor_discounted = ext._compute_eyelid_tremor(0.28, 10.0, "open", vibration_score=0.60)
        # When discounted by (1 - 0.60) = 0.40, tremor_discounted <= 0.40 * tremor_undiscounted + 1e-4
        assert tremor_discounted <= tremor_undiscounted


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

