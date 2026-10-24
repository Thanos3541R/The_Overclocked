"""Unit tests for pose-compensated EAR/PERCLOS, pitch-invariant brow ratio, and rPPG IBI filtering."""
import numpy as np
import pytest

from pipeline.feature_extractor import FeatureExtractor, FrameFeatures
from tests.test_alcohol_signals import make_full_landmarks


class TestPoseCompensatedEARAndBrowRatio:
    """Verify pose compensation for yaw, pitch, and rPPG HRV outlier filtering."""

    def test_synthetic_landmarks_ear_unaltered(self):
        """When image_landmarks is None (synthetic unit tests), ear_avg must remain exact (ear_left + ear_right) / 2.0."""
        ext = FeatureExtractor(fps=30.0)
        lm = make_full_landmarks(ear_val=0.25)
        # Call compute without image_landmarks
        feat = ext.compute(lm, timestamp=0.0, frame_id=0, face_valid=True, head_yaw=25.0, head_pitch=12.0)
        assert feat.ear_avg == pytest.approx((feat.ear_left + feat.ear_right) / 2.0, rel=1e-5)

    def test_image_landmarks_yaw_and_pitch_compensation(self):
        """When image_landmarks is provided:
        - High yaw (|yaw| > 12) weights the camera-facing un-foreshortened eye more heavily.
        - Downward pitch (pitch > 6) applies pitch perspective correction.
        """
        ext = FeatureExtractor(fps=30.0)
        lm = make_full_landmarks(ear_val=0.25)
        raw_feat = ext.compute(lm, timestamp=0.0, frame_id=0, face_valid=True)
        raw_ear_avg = raw_feat.ear_avg

        # Now pass image_landmarks with head_yaw = 24.0 and downward head_pitch = 14.0
        ext_comp = FeatureExtractor(fps=30.0)
        feat_comp = ext_comp.compute(
            lm,
            timestamp=0.0,
            frame_id=0,
            face_valid=True,
            head_yaw=24.0,
            head_pitch=14.0,
            image_landmarks=lm,
        )

        # Expected pitch boost = min(0.045, (14.0 - 6.0) * 0.0025) = 0.020
        assert feat_comp.ear_avg > raw_ear_avg
        expected_boost = min(0.045, (14.0 - 6.0) * 0.0025)
        assert feat_comp.ear_avg == pytest.approx(raw_ear_avg + expected_boost, rel=1e-3)
        # Raw ear_left and ear_right contracts must remain untouched
        assert feat_comp.ear_left == pytest.approx(raw_feat.ear_left, rel=1e-5)
        assert feat_comp.ear_right == pytest.approx(raw_feat.ear_right, rel=1e-5)

    def test_voluntary_eye_widening_gated_by_high_yaw(self):
        """Voluntary eye widening must be suppressed when |head_yaw| >= 20.0."""
        ext = FeatureExtractor(fps=30.0)
        # Driver forces eyes wide (EAR = 0.40) but head turned (yaw = 25.0)
        for i in range(10):
            t = i / 30.0
            lm = make_full_landmarks(ear_val=0.20)
            ext.compute(lm, timestamp=t, frame_id=i, face_valid=True)

        # Eye surge at high yaw
        lm_wide = make_full_landmarks(ear_val=0.40)
        feat_yaw = ext.compute(lm_wide, timestamp=0.35, frame_id=11, face_valid=True, head_yaw=25.0)
        assert feat_yaw.voluntary_eye_widening is False

        # Eye surge at frontal pose
        ext_frontal = FeatureExtractor(fps=30.0)
        for i in range(10):
            t = i / 30.0
            lm = make_full_landmarks(ear_val=0.20)
            ext_frontal.compute(lm, timestamp=t, frame_id=i, face_valid=True)

        feat_frontal = ext_frontal.compute(lm_wide, timestamp=0.35, frame_id=11, face_valid=True, head_yaw=0.0)
        assert feat_frontal.voluntary_eye_widening is True

    def test_brow_pitch_compensation(self):
        """Tilting head down (head_pitch > 0) compensates raw_ratio by (1 + 0.006 * pitch)."""
        from utils.landmarks import RIGHT_EYEBROW_ARCH, LEFT_EYEBROW_ARCH
        ext = FeatureExtractor(fps=30.0)
        lm = make_full_landmarks(ear_val=0.28)
        for idx in RIGHT_EYEBROW_ARCH:
            lm[idx] = [200.0, 175.0, 0.0]
        for idx in LEFT_EYEBROW_ARCH:
            lm[idx] = [400.0, 175.0, 0.0]

        ratio_0, _, _ = ext._compute_brow_micromotion(lm, ear_avg=0.28, head_pitch=0.0)
        ext.reset()
        ratio_10, _, _ = ext._compute_brow_micromotion(lm, ear_avg=0.28, head_pitch=10.0)

        # ratio_10 should be ratio_0 * (1.0 + 0.006 * 10.0) = ratio_0 * 1.06
        assert ratio_10 == pytest.approx(ratio_0 * 1.06, rel=1e-3)

    def test_rppg_hrv_outlier_filtering_and_clamping(self):
        """Motion-artifact IBI jumps must be rejected and rmssd_ms clamped in [10.0, 135.0]."""
        ext = FeatureExtractor(fps=60.0)
        lm = make_full_landmarks(ear_val=0.30)
        from utils.landmarks import VASODILATION_LANDMARKS
        for idx in VASODILATION_LANDMARKS:
            lm[idx] = [320.0, 200.0, 0.0]

        # 60 FPS stream with 1.2 Hz pulse
        for i in range(270):
            t = i / 60.0
            frame = np.full((480, 640, 3), 120, dtype=np.uint8)
            # Add occasional noise spike to green channel to simulate sunlight / camera shake
            noise = 25.0 if (i % 60 == 0) else 0.0
            g_val = int(np.clip(120.0 + 15.0 * np.sin(2 * np.pi * 1.2 * t) + noise, 0, 255))
            frame[:, :, 1] = g_val
            frame[:, :, 0] = 100
            frame[:, :, 2] = 140
            feat = ext.compute(lm, timestamp=t, frame_id=i+1, face_valid=True, frame=frame)

        assert feat.rppg_valid is True
        # RMSSD must stay physiologically plausible (<= 135.0 ms)
        assert 10.0 <= feat.rppg_hrv_rmssd_ms <= 135.0
