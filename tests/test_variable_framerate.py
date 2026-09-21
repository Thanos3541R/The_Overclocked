"""Unit tests verifying dynamic dt scaling and variable frame rate robustness.

Verifies:
  1. IMU Savitzky-Golay quadratic fit correctly estimates analytical acceleration
     under irregular time intervals (non-uniform dt).
  2. IMU gate suppresses sub-pixel noise under variable dt and cleanly catches real shocks.
  3. Saccadic velocity scales inversely with dt (v = Δθ / Δt, no hardcoded 33.3ms).
  4. Blink downstroke/upstroke velocity tracks dynamic dt across dropped frames.
  5. Baseline asymmetry subtraction in ImpairmentClassifier prevents false positive
     alerts for naturally asymmetric drivers while catching pathological excess asymmetry.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest
from pipeline.imu_gate import SoftwareIMUGate
from pipeline.feature_extractor import FeatureExtractor
from pipeline.classifier import ImpairmentClassifier
from tests.test_alcohol_signals import make_full_landmarks


class TestVariableFramerateIMU:
    """Tests for SoftwareIMUGate under non-uniform and irregular dt."""

    def test_imu_polynomial_fit_analytical_parabola(self):
        """A pure parabolic trajectory y(t) = 0.5 * a * t^2 with non-uniform dt
        should yield exact acceleration a = 3.2 m/s²."""
        gate = SoftwareIMUGate(accel_threshold=2.5, alpha=1.0, window_size=5)
        # Non-uniform timestamps spanning 120ms (dt varies between 15ms and 35ms)
        timestamps = [0.000, 0.020, 0.055, 0.085, 0.120]
        a_true_m_s2 = 3.2
        a_true_mm_s2 = a_true_m_s2 * 1000.0  # 3200 mm/s²

        for t in timestamps:
            y = 0.5 * a_true_mm_s2 * (t ** 2)
            pos = np.array([0.0, y, 600.0])
            gate.update(timestamp=t, tvec=pos, euler_angles=(0.0, 0.0, 0.0))

        estimated_accel = gate._estimate_acceleration()
        # Non-uniform quadratic Vandermonde fit should recover a_true within numerical precision
        assert pytest.approx(estimated_accel, rel=1e-3) == a_true_m_s2

    def test_imu_irregular_intervals_noise_suppression(self):
        """Sub-pixel jitter under irregular frame intervals (15ms to 50ms)
        must not trigger the shock freeze threshold (2.5 m/s²)."""
        gate = SoftwareIMUGate(accel_threshold=2.5, freeze_duration_ms=150.0, alpha=0.3, window_size=5)
        base_tvec = np.array([0.0, 0.0, 600.0])

        t = 0.0
        # 15 frames with randomly varying frame intervals
        np.random.seed(42)
        for _ in range(15):
            dt = np.random.uniform(0.018, 0.048)  # 20 to 55 fps variation
            t += dt
            noise = np.random.uniform(-0.4, 0.4, size=3)
            frozen = gate.update(timestamp=t, tvec=base_tvec + noise, euler_angles=(0.0, 0.0, 0.0))
            assert frozen is False, f"Irregular frame jitter falsely triggered shock at t={t:.3f}s"

    def test_imu_shock_detected_with_dropped_frame(self):
        """A pothole shock occurring across a dropped/delayed frame (dt = 75ms)
        is still properly detected without numerical blowout."""
        gate = SoftwareIMUGate(accel_threshold=2.5, freeze_duration_ms=150.0, alpha=0.5, window_size=5)
        base_tvec = np.array([0.0, 0.0, 600.0])

        gate.update(0.000, base_tvec, (0.0, 0.0, 0.0))
        gate.update(0.033, base_tvec, (0.0, 0.0, 0.0))
        # Frame dropped: interval is 75ms instead of 33ms, accompanied by vertical shock
        shock_tvec = base_tvec + np.array([0.0, 90.0, 0.0])
        frozen = gate.update(0.108, shock_tvec, (0.0, 0.0, 0.0))

        assert frozen is True, "Pothole shock across delayed frame must trigger freeze"


class TestVariableFramerateOculomotor:
    """Tests verifying dynamic dt scaling in saccade and blink feature extraction."""

    def test_saccade_velocity_scales_inversely_with_dt(self):
        """Angular velocity v = Δθ / dt must scale inversely when dt doubles."""
        landmarks_center = make_full_landmarks(ear_val=0.30, iris_x_offset=0.0)
        landmarks_shifted = make_full_landmarks(ear_val=0.30, iris_x_offset=6.0)

        # Test at dt = 0.030s
        extractor1 = FeatureExtractor()
        extractor1.compute(landmarks_center, timestamp=0.000, frame_id=0, face_valid=True)
        feat1 = extractor1.compute(landmarks_shifted, timestamp=0.030, frame_id=1, face_valid=True)

        # Test at dt = 0.060s (double the time for the same displacement -> half the velocity)
        extractor2 = FeatureExtractor()
        extractor2.compute(landmarks_center, timestamp=0.000, frame_id=0, face_valid=True)
        feat2 = extractor2.compute(landmarks_shifted, timestamp=0.060, frame_id=1, face_valid=True)

        assert feat1.saccade_velocity > 0
        assert feat2.saccade_velocity > 0
        ratio = feat1.saccade_velocity / feat2.saccade_velocity
        assert pytest.approx(ratio, rel=1e-2) == 2.0, (
            f"Expected 2.0x velocity ratio when dt is halved, got {ratio:.3f}"
        )

    def test_blink_velocity_dynamic_dt_dropped_frame(self):
        """Blink closing velocity |dEAR/dt| must use dynamic timestamp delta dt."""
        extractor = FeatureExtractor(blink_threshold=0.20)

        # Phase 1: Steady open eyes (irregular intervals)
        for i, t in enumerate([0.000, 0.028, 0.062, 0.095]):
            lm = make_full_landmarks(ear_val=0.15)  # EAR ≈ 0.30
            extractor.compute(lm, timestamp=t, frame_id=i, face_valid=True)

        # Phase 2: Rapid closing downstroke with dropped frame (gap of 60ms)
        lm_closing1 = make_full_landmarks(ear_val=0.12)  # EAR ≈ 0.24
        extractor.compute(lm_closing1, timestamp=0.125, frame_id=4, face_valid=True)
        # Dropped frame mid-closing: jump from 0.125 to 0.185
        lm_closed = make_full_landmarks(ear_val=0.06)   # EAR ≈ 0.12 < 0.20
        extractor.compute(lm_closed, timestamp=0.185, frame_id=5, face_valid=True)

        # Phase 3: Bottom of blink
        lm_bottom = make_full_landmarks(ear_val=0.05)   # EAR ≈ 0.10
        extractor.compute(lm_bottom, timestamp=0.220, frame_id=6, face_valid=True)

        # Phase 4: Reopening upstroke (dt = 45ms)
        lm_reopen = make_full_landmarks(ear_val=0.11)   # EAR ≈ 0.22
        extractor.compute(lm_reopen, timestamp=0.265, frame_id=7, face_valid=True)

        # Phase 5: Reopened & settled (dt = 50ms)
        lm_open1 = make_full_landmarks(ear_val=0.15)    # EAR ≈ 0.30
        extractor.compute(lm_open1, timestamp=0.315, frame_id=8, face_valid=True)
        feat_final = extractor.compute(lm_open1, timestamp=0.365, frame_id=9, face_valid=True)

        # Blink completed: both closing and opening velocities must be measured and positive
        assert feat_final.blink_closing_velocity > 0.0
        assert feat_final.blink_opening_velocity > 0.0


class TestBaselineAsymmetry:
    """Tests for natural facial asymmetry baseline subtraction in ImpairmentClassifier."""

    def test_natural_asymmetry_subtracted_avoids_false_positive(self):
        """Driver with natural resting asymmetry (|ΔEAR| = 0.045) should NOT be flagged
        when baseline_asymmetry = 0.045, but WOULD be flagged without baseline."""
        dummy_frame = np.full((480, 640, 3), 180, dtype=np.uint8)
        landmarks = make_full_landmarks(ear_val=0.30)

        # Default classifier with 0.0 baseline flags 0.045 as moderate asymmetry
        default_classifier = ImpairmentClassifier(baseline_asymmetry=0.0)
        assess_default = default_classifier.evaluate_static_image(
            frame=dummy_frame,
            landmarks=landmarks,
            ear_left=0.32,
            ear_right=0.275,  # |ΔEAR| = 0.045 > 0.035
            mar=0.10,
            head_pitch=0.0,
            head_roll=0.0
        )
        assert assess_default.marker_scores["ocular_asymmetry"] == 40.0
        assert any("asymmetry" in ind.lower() for ind in assess_default.primary_indicators)

        # Calibrated classifier with driver's resting baseline (0.045) ignores natural asymmetry
        calibrated_classifier = ImpairmentClassifier(baseline_asymmetry=0.045)
        assess_calibrated = calibrated_classifier.evaluate_static_image(
            frame=dummy_frame,
            landmarks=landmarks,
            ear_left=0.32,
            ear_right=0.275,
            mar=0.10,
            head_pitch=0.0,
            head_roll=0.0
        )
        assert assess_calibrated.marker_scores["ocular_asymmetry"] == 0.0
        assert not any("asymmetry" in ind.lower() for ind in assess_calibrated.primary_indicators)

    def test_excess_asymmetry_above_baseline_is_detected(self):
        """If a calibrated driver experiences severe unilateral ptosis (e.g. |ΔEAR| = 0.12),
        excess asymmetry (0.12 - 0.045 = 0.075 > 0.06) triggers severe alert."""
        dummy_frame = np.full((480, 640, 3), 180, dtype=np.uint8)
        landmarks = make_full_landmarks(ear_val=0.20)

        calibrated_classifier = ImpairmentClassifier(baseline_asymmetry=0.045)
        assessment = calibrated_classifier.evaluate_static_image(
            frame=dummy_frame,
            landmarks=landmarks,
            ear_left=0.28,
            ear_right=0.16,  # |ΔEAR| = 0.12, excess = 0.075
            mar=0.10,
            head_pitch=0.0,
            head_roll=0.0
        )
        assert assessment.marker_scores["ocular_asymmetry"] == 75.0
        assert any("severe eyelid asymmetry" in ind.lower() for ind in assessment.primary_indicators)
