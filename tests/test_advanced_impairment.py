"""Unit tests for advanced oculomotor impairment signals.

Tests verify:
  1. Binocular Vergence & Lack of Convergence (LOC - DRE sign)
  2. Vestibulo-Ocular Reflex (VOR) Micro-Compensation Gain (PMC8997842)
  3. Gaze-Evoked Nystagmus (GEN - Romano et al. 2017)
  4. Smooth Pursuit Fragmentation Ratio
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest
from pipeline.feature_extractor import FeatureExtractor, FrameFeatures
from tests.test_alcohol_signals import make_full_landmarks


class TestAdvancedOculomotorSignals:
    """Test suite for high-specificity neurological alcohol signals."""

    def test_binocular_vergence_and_loc(self):
        """Vergence should detect parallel gaze, convergence, and divergent strabismus (LOC)."""
        ext = FeatureExtractor(fps=30.0, loc_vergence_threshold_deg=-3.5)

        # 1. Parallel forward gaze (0, 0)
        lm_parallel = make_full_landmarks(ear_val=0.30, iris_x_offset=0.0)
        feat_parallel = ext.compute(lm_parallel, timestamp=0.0, frame_id=0, face_valid=True)
        assert abs(feat_parallel.vergence_angle_deg) < 2.0
        assert feat_parallel.lack_of_convergence is False

        # 2. Exotropic divergence / Lack of Convergence
        # Artificially move left iris temporal (-x) and right iris temporal (-x)
        # Left eye: outer is 263 (temporal), inner is 362 (nasal)
        # Right eye: outer is 33 (temporal), inner is 133 (nasal)
        lm_divergent = make_full_landmarks(ear_val=0.30)
        # Shift left iris temporal (outer) and right iris temporal (outer)
        # In make_full_landmarks: right eye cx=200, outer=180, inner=220 (temporal is left / -X)
        # Left eye cx=400, outer=380, inner=420 (temporal is left / -X)
        lm_divergent[473][0] -= 10.0  # right iris temporal (abducted)
        lm_divergent[468][0] -= 10.0  # left iris temporal (abducted)

        feat_divergent = ext.compute(lm_divergent, timestamp=1/30.0, frame_id=1, face_valid=True)
        assert feat_divergent.vergence_angle_deg < -3.5
        assert feat_divergent.lack_of_convergence is True

    def test_vor_micro_gain_normal_vs_depressed(self):
        """VOR micro-compensation should compute gain ~1.0 for healthy counter-rotation."""
        ext_normal = FeatureExtractor(fps=30.0, vor_min_head_speed=2.0, vor_max_head_speed=25.0)

        # Frame 0: Head at yaw=0°, iris centered
        lm0 = make_full_landmarks(ear_val=0.30, iris_x_offset=0.0)
        ext_normal.compute(lm0, timestamp=0.0, frame_id=0, face_valid=True, head_yaw=0.0)

        # Frame 1: Head turns right by +5° in 0.1s (50 deg/s is too fast, let's do 1° in 0.1s = 10 deg/s)
        # For perfect VOR (gain=1.0), eye counter-rotates left by -1° (offset ≈ -1° / 15°/unit = -0.067 units)
        # In make_full_landmarks, eye width = 40px, 1 unit = 20px, so 1° ≈ 1.33px
        for step in range(1, 10):
            t = step * 0.05
            head_yaw = step * 0.5  # Head moves +10 deg/s
            # Eye counter-rotates opposite to head
            iris_shift = -step * 0.67
            lm = make_full_landmarks(ear_val=0.30, iris_x_offset=iris_shift)
            feat = ext_normal.compute(lm, timestamp=t, frame_id=step, face_valid=True, head_yaw=head_yaw)

        assert 0.75 <= feat.vor_gain <= 1.25, f"Expected VOR gain ~1.0, got {feat.vor_gain:.2f}"

    def test_gaze_evoked_nystagmus_detection(self):
        """GEN detector should detect repeating centripetal drift and fast outward beats."""
        ext = FeatureExtractor(
            fps=30.0,
            eccentric_gaze_threshold=0.35,
            gen_drift_min_vel=2.0,
            gen_drift_max_vel=18.0,
            gen_reset_min_vel=30.0
        )

        t = 0.0
        frame_id = 0

        # Prime at eccentric gaze (iris_x ≈ +0.50, nasal)
        # In make_full_landmarks: 1 iris unit ≈ 20px, so offset=10.0px gives iris_x ≈ +0.50
        lm = make_full_landmarks(ear_val=0.30, iris_x_offset=10.0)
        ext.compute(lm, timestamp=t, frame_id=frame_id, face_valid=True)

        # Simulate 2 complete sawtooth nystagmus cycles
        for cycle in range(2):
            # Slow centripetal drift: drifts inward from 10.0px to 8.5px over 3 frames (~7 deg/s)
            for drift_step in [9.5, 9.0, 8.5]:
                t += 0.033
                frame_id += 1
                lm = make_full_landmarks(ear_val=0.30, iris_x_offset=drift_step)
                ext.compute(lm, timestamp=t, frame_id=frame_id, face_valid=True)

            # Fast corrective beat: rapid jump back outward from 8.5px to 10.5px in 1 frame (~45 deg/s)
            t += 0.033
            frame_id += 1
            lm = make_full_landmarks(ear_val=0.30, iris_x_offset=10.5)
            feat = ext.compute(lm, timestamp=t, frame_id=frame_id, face_valid=True)

        assert feat.gen_detected is True, "GEN should be detected after 2 drift-then-beat cycles"
        assert feat.gen_beat_count >= 2

    def test_smooth_pursuit_fragmentation_ratio(self):
        """Pursuit fragmentation should be high when tracking is disrupted by frequent saccades."""
        ext_smooth = FeatureExtractor(fps=30.0, pursuit_window_sec=2.0)
        ext_jerky = FeatureExtractor(fps=30.0, pursuit_window_sec=2.0)

        # 1. Smooth pursuit: small continuous steps (2 deg/s speed, no saccades)
        for i in range(30):
            t = i / 30.0
            lm = make_full_landmarks(ear_val=0.30, iris_x_offset=i * 0.1)
            feat_smooth = ext_smooth.compute(lm, timestamp=t, frame_id=i, face_valid=True)

        assert feat_smooth.pursuit_fragmentation_ratio < 0.10

        # 2. Jerky pursuit with frequent catch-up saccades
        for i in range(30):
            t = i / 30.0
            # Alternate between steady small drift and sudden rapid jump (+12px shift)
            offset = (i * 0.1) + (12.0 if (i % 5 == 0) else 0.0)
            lm = make_full_landmarks(ear_val=0.30, iris_x_offset=offset)
            feat_jerky = ext_jerky.compute(lm, timestamp=t, frame_id=i, face_valid=True)

        assert feat_jerky.pursuit_fragmentation_ratio > feat_smooth.pursuit_fragmentation_ratio
        assert feat_jerky.pursuit_fragmentation_ratio > 0.15


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
