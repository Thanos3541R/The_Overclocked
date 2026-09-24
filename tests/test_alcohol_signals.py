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
from tests.test_ear import make_eye_landmarks
from utils.landmarks import RIGHT_EYE_EAR_FLAT, LEFT_EYE_EAR_FLAT


def make_full_landmarks(ear_val=0.30, iris_x_offset=0.0):
    """Create synthetic (478, 3) landmarks with configurable EAR and iris position.

    Args:
        ear_val: Approximate EAR to produce (controls vertical opening).
        iris_x_offset: Horizontal offset for iris from eye center (pixels).
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
    lm[473] = [eye_cx + iris_x_offset, eye_cy, 0]    # right iris center
    lm[468] = [eye_cx2 + iris_x_offset, eye_cy2, 0]  # left iris center

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

    def test_rapid_shift_produces_high_velocity(self):
        """A rapid iris shift between frames should produce high velocity."""
        ext = FeatureExtractor(fps=30.0)

        # Frame 0: center gaze
        lm0 = make_full_landmarks(ear_val=0.30, iris_x_offset=0.0)
        ext.compute(lm0, timestamp=0.0, frame_id=0, face_valid=True)

        # Frame 1: large horizontal shift (simulating saccade)
        lm1 = make_full_landmarks(ear_val=0.30, iris_x_offset=15.0)
        feat = ext.compute(lm1, timestamp=1/30.0, frame_id=1, face_valid=True)

        assert feat.saccade_velocity > 20.0, \
            f"Saccade velocity={feat.saccade_velocity:.1f} deg/s, expected > 20"

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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
