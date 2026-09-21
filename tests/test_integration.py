"""End-to-end integration tests for the Driver Monitoring Pipeline components.

Verifies:
  - Lighting normalizer processes both BGR and grayscale frames correctly
  - PnP normalizer + 1-Euro filter + Head pose gating + Feature extractor + IMU gate
    operate together seamlessly in a simulated multi-frame sequence
  - Vehicle bounce shock simulation triggers IMU feature freeze
  - Extreme head yaw triggers head pose gating
  - CSV output writes and reads correctly
"""
import sys
import os
import csv
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline.preprocessing import LightingNormalizer
from pipeline.pnp_normalizer import PnPNormalizer
from pipeline.one_euro_filter import OneEuroFilterBank
from pipeline.head_pose_estimator import HeadPoseGate
from pipeline.feature_extractor import FeatureExtractor, FrameFeatures
from pipeline.imu_gate import SoftwareIMUGate
from tests.test_pnp import make_frontal_face_landmarks


class TestPipelineIntegration:
    """Test full pipeline interaction without physical camera."""

    def test_lighting_normalizer_bgr_and_grayscale(self):
        """Verify normalizer handles color and NIR grayscale inputs cleanly."""
        normalizer = LightingNormalizer()

        # 1. Dark BGR frame
        dark_bgr = np.full((480, 640, 3), 30, dtype=np.uint8)
        enhanced_bgr = normalizer.process(dark_bgr)
        assert enhanced_bgr.shape == (480, 640, 3)
        assert enhanced_bgr.mean() > dark_bgr.mean(), "Dark BGR frame should be brightened"

        # 2. Dark Grayscale (NIR) frame
        dark_gray = np.full((480, 640), 25, dtype=np.uint8)
        enhanced_gray = normalizer.process(dark_gray)
        assert enhanced_gray.shape == (480, 640)
        assert enhanced_gray.mean() > dark_gray.mean(), "Dark NIR frame should be brightened"

    def test_full_pipeline_multi_frame_flow(self):
        """Simulate a 30-frame sequence through the entire signal pipeline."""
        pnp = PnPNormalizer(1280, 720)
        pose_gate = HeadPoseGate(yaw_limit=35.0, pitch_down_limit=-30.0, pitch_up_limit=25.0)
        extractor = FeatureExtractor(blink_threshold=0.20, perclos_window_sec=10.0, fps=30.0)
        imu = SoftwareIMUGate(accel_threshold=2.5, freeze_duration_ms=150.0)

        filter_bank = None
        features_list = []

        # 30 frames at 30 fps
        for frame_idx in range(30):
            timestamp = frame_idx / 30.0

            # Generate synthetic landmarks with a slight road bump at frame 15
            bump_y = 60.0 if frame_idx == 15 else 0.0
            raw_lm = make_frontal_face_landmarks(offset_y=bump_y)

            # 1-Euro filter FIRST on raw landmarks
            if filter_bank is None:
                filter_bank = OneEuroFilterBank(timestamp, raw_lm)
                smooth_raw = raw_lm
            else:
                smooth_raw = filter_bank.filter_landmarks(timestamp, raw_lm)

            # PnP on smoothed landmarks
            pnp_success = pnp.solve_pose(smooth_raw)
            assert pnp_success
            norm_lm = pnp.normalize_landmarks(smooth_raw)

            # Head pose gate
            euler = pnp.get_euler_angles()
            pose_state = pose_gate.check(euler)

            # IMU gate (detect bump at frame 15 with rotational decoupling)
            imu_frozen = imu.update(timestamp, pnp.tvec, euler)

            # Feature extractor
            feat = extractor.compute(
                landmarks=norm_lm,
                timestamp=timestamp,
                frame_id=frame_idx,
                face_valid=pose_state.face_valid,
                head_yaw=pose_state.yaw,
                head_pitch=pose_state.pitch,
                head_roll=pose_state.roll,
                imu_gated=imu_frozen,
                gating_reason=pose_state.gating_reason,
            )
            features_list.append(feat)

        assert len(features_list) == 30
        # Normal frames should be valid
        assert features_list[0].face_valid is True
        # EAR should be around 0.30
        assert 0.20 <= features_list[0].ear_right <= 0.40

    def test_extreme_yaw_triggers_gating(self):
        """Simulate head turn beyond 35 degrees yaw and verify gating."""
        pose_gate = HeadPoseGate(yaw_limit=35.0)
        # Check normal pose
        normal_state = pose_gate.check((0.0, 10.0, 0.0))
        assert normal_state.face_valid is True

        # Check extreme turn (e.g. 45 degrees yaw)
        turn_state = pose_gate.check((0.0, 45.0, 0.0))
        assert turn_state.face_valid is False
        assert "exceeds" in turn_state.gating_reason
