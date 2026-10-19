"""Unit tests for SoftwareIMUGate: road shock detection and rotational decoupling.

Tests verify:
  1. Low-pass filter stabilizes tvec and ignores sub-pixel jitter.
  2. Pure vertical linear shock (pothole) with low rotational velocity triggers 150ms freeze.
  3. High linear translation accompanied by intentional head turn (> 20 deg/s) does NOT trigger freeze.
  4. Freeze duration holds across consecutive frames.
  5. Reset clears state.
"""
import numpy as np
import pytest
from pipeline.imu_gate import SoftwareIMUGate


class TestSoftwareIMUGate:

    def test_subpixel_jitter_does_not_trigger_freeze(self):
        """Micro-jitter (< 1mm noise) should be damped by the low-pass filter."""
        gate = SoftwareIMUGate(accel_threshold=2.5, freeze_duration_ms=150.0, alpha=0.3)
        base_tvec = np.array([0.0, 0.0, 600.0])  # mm

        # 10 frames with +/- 0.5 mm noise at 30 fps (dt = 0.033s)
        for i in range(10):
            t = i * 0.033
            noise = np.random.uniform(-0.5, 0.5, size=3)
            frozen = gate.update(timestamp=t, tvec=base_tvec + noise, euler_angles=(0.0, 0.0, 0.0))
            assert frozen is False, f"Sub-pixel noise should not trigger shock at frame {i}"

    def test_pothole_vertical_shock_triggers_freeze(self):
        """Vertical chassis shock with zero head rotation triggers feature freeze."""
        gate = SoftwareIMUGate(accel_threshold=2.5, freeze_duration_ms=150.0, alpha=0.5)
        base_tvec = np.array([0.0, 0.0, 600.0])

        # Frame 0, 1: steady
        gate.update(0.000, base_tvec, (0.0, 0.0, 0.0))
        gate.update(0.033, base_tvec, (0.0, 0.0, 0.0))

        # Frame 2: sudden pothole vertical displacement of 80mm in 33ms (massive linear accel)
        shock_tvec = base_tvec + np.array([0.0, 80.0, 0.0])
        frozen = gate.update(0.066, shock_tvec, (0.0, 0.0, 0.0))

        assert frozen is True, "Pothole shock without head rotation must trigger freeze"

    def test_head_turn_does_not_trigger_freeze(self):
        """Rapid head turn (e.g. 45 deg/s) with translation must NOT trigger shock freeze."""
        gate = SoftwareIMUGate(accel_threshold=2.5, max_rot_vel_for_shock=20.0, alpha=0.5)
        base_tvec = np.array([0.0, 0.0, 600.0])

        gate.update(0.000, base_tvec, (0.0, 0.0, 0.0))
        gate.update(0.033, base_tvec, (0.0, 5.0, 0.0))

        # Driver turns head rapidly to check mirror: 80mm displacement AND 25 degrees rotation in 33ms (~750 deg/s)
        turn_tvec = base_tvec + np.array([50.0, 0.0, 0.0])
        turn_euler = (0.0, 30.0, 0.0)  # yaw turned rapidly

        frozen = gate.update(0.066, turn_tvec, turn_euler)
        assert frozen is False, "Rapid head turn with high rotational velocity must NOT be mistaken for road shock"

    def test_freeze_persists_for_duration(self):
        """Feature freeze must persist across the full 150ms window."""
        gate = SoftwareIMUGate(accel_threshold=2.5, freeze_duration_ms=150.0, alpha=0.5)
        base_tvec = np.array([0.0, 0.0, 600.0])

        gate.update(0.000, base_tvec, (0.0, 0.0, 0.0))
        gate.update(0.033, base_tvec, (0.0, 0.0, 0.0))
        # Trigger shock at t = 0.066s -> freeze until t = 0.216s
        shock_tvec = base_tvec + np.array([0.0, 80.0, 0.0])
        gate.update(0.066, shock_tvec, (0.0, 0.0, 0.0))

        # At t = 0.100s and 0.150s, car remains at new elevation, still in 150ms freeze window
        assert gate.update(0.100, shock_tvec, (0.0, 0.0, 0.0)) is True
        assert gate.update(0.150, shock_tvec, (0.0, 0.0, 0.0)) is True

        # At t = 0.250s (> 0.216s), freeze window has expired
        assert gate.update(0.250, shock_tvec, (0.0, 0.0, 0.0)) is False

    def test_speed_bump_with_default_alpha_triggers_freeze(self):
        """Default alpha=0.65 preserves vertical speed bump acceleration impulse."""
        gate = SoftwareIMUGate()  # default alpha=0.65
        base_tvec = np.array([0.0, 0.0, 600.0])

        gate.update(0.000, base_tvec, (0.0, 0.0, 0.0))
        gate.update(0.033, base_tvec, (0.0, 0.0, 0.0))

        # Speed bump: +60mm in 33ms without head rotation
        bump_tvec = base_tvec + np.array([0.0, 60.0, 0.0])
        frozen = gate.update(0.066, bump_tvec, (0.0, 0.0, 0.0))
        assert frozen is True, "Speed bump impulse with default alpha=0.65 must trigger freeze"

    def test_single_pothole_shock_does_not_activate_continuous_vibration(self):
        """A single transient pothole (> 4.0 m/s²) triggers 150ms freeze but not continuous vibration."""
        gate = SoftwareIMUGate(accel_threshold=4.0, freeze_duration_ms=150.0)
        base_tvec = np.array([0.0, 0.0, 600.0])

        gate.update(0.000, base_tvec, (0.0, 0.0, 0.0))
        gate.update(0.033, base_tvec, (0.0, 0.0, 0.0))

        # Pothole vertical displacement: 80mm in 33ms
        shock_tvec = base_tvec + np.array([0.0, 80.0, 0.0])
        frozen = gate.update(0.066, shock_tvec, (0.0, 0.0, 0.0))

        assert frozen is True
        assert gate.is_frozen is True
        assert gate.continuous_vibration_active is False

        # Frames within the 150ms freeze window (t = 0.100s, 0.150s)
        assert gate.update(0.100, shock_tvec, (0.0, 0.0, 0.0)) is True
        assert gate.update(0.150, shock_tvec, (0.0, 0.0, 0.0)) is True

        # After 150ms expires (t = 0.250s > 0.216s)
        unfrozen = gate.update(0.250, shock_tvec, (0.0, 0.0, 0.0))
        assert unfrozen is False
        assert gate.is_frozen is False
        assert gate.continuous_vibration_active is False

    def test_continuous_vibration_mode_activates_without_indefinite_freeze(self):
        """Sustained oscillatory acceleration (1.5 - 4.0 m/s²) for > 500ms activates
        continuous_vibration_active while keeping feature extraction live (never freezing)."""
        gate = SoftwareIMUGate(accel_threshold=4.0, alpha=1.0)
        base_tvec = np.array([0.0, 0.0, 600.0])

        # 25 frames at 30 fps (800ms) with pure linear acceleration a = 2.5 m/s² (2500 mm/s²)
        # y(t) = 0.5 * a * t^2
        frozen_history = []
        vib_active_history = []

        for i in range(25):
            t = i * 0.033
            y = 0.5 * 2500.0 * (t ** 2)
            tvec = base_tvec + np.array([0.0, y, 0.0])
            frozen = gate.update(t, tvec, (0.0, 0.0, 0.0))
            frozen_history.append(frozen)
            vib_active_history.append(gate.continuous_vibration_active)

        # In EVERY frame, the pipeline must NOT freeze (acceleration < 4.0 m/s²)
        assert not any(frozen_history), "Continuous vibration (1.5-4.0 m/s²) must NOT freeze the pipeline"

        # Continuous vibration should become True after 500ms (around frame 16)
        assert vib_active_history[10] is False, "Continuous vibration should not be active before 500ms"
        assert vib_active_history[-1] is True, "Continuous vibration MUST be active after 500ms"

    def test_max_freeze_duty_cycle_cap_over_rolling_window(self):
        """Repeated severe shocks (> 4.0 m/s²) must be capped at 25% duty cycle over 2.0s rolling window,
        switching to continuous_vibration_active = True and preventing permanent freeze."""
        gate = SoftwareIMUGate(
            accel_threshold=4.0,
            freeze_duration_ms=150.0,
            max_freeze_duty_cycle=0.25,
            duty_cycle_window_sec=2.0,
        )
        base_tvec = np.array([0.0, 0.0, 600.0])

        # 60 frames over 2.0 seconds at 30 fps (dt = 0.033s)
        # Repeated vertical shocks every 5 frames (~165ms)
        frozen_count = 0
        for i in range(60):
            t = i * 0.033
            if i % 5 == 2:
                tvec = base_tvec + np.array([0.0, 80.0, 0.0])
            elif i % 5 == 3:
                tvec = base_tvec + np.array([0.0, -80.0, 0.0])
            else:
                tvec = base_tvec.copy()

            frozen = gate.update(t, tvec, (0.0, 0.0, 0.0))
            if frozen:
                frozen_count += 1

        duty_cycle = (frozen_count * 0.033) / 2.0
        # Duty cycle must be capped near 25% (at most ~18 frames out of 60)
        assert frozen_count <= 18, f"Freeze duty cycle exceeded cap: {frozen_count}/60 frames ({duty_cycle:.1%})"
        assert gate.continuous_vibration_active is True, "Continuous vibration must be active when duty cycle cap reached"

    def test_continuous_vibration_resets_on_smooth_road(self):
        """Continuous vibration mode should reset to False once vehicle drives on smooth road for > 500ms."""
        gate = SoftwareIMUGate(accel_threshold=4.0, alpha=1.0)
        base_tvec = np.array([0.0, 0.0, 600.0])

        # 1. Drive on rough road for 700ms (22 frames)
        for i in range(22):
            t = i * 0.033
            y = 0.5 * 2500.0 * (t ** 2)
            gate.update(t, base_tvec + np.array([0.0, y, 0.0]), (0.0, 0.0, 0.0))

        assert gate.continuous_vibration_active is True

        # 2. Drive on smooth road for 700ms (accel ~ 0)
        t_start = 22 * 0.033
        for i in range(22):
            t = t_start + i * 0.033
            gate.update(t, base_tvec, (0.0, 0.0, 0.0))

        assert gate.continuous_vibration_active is False, "Continuous vibration should deactivate after smooth road"

    def test_gate_reset_clears_continuous_vibration_state(self):
        """Calling reset() must clear continuous vibration and duty cycle states."""
        gate = SoftwareIMUGate(accel_threshold=4.0, alpha=1.0)
        base_tvec = np.array([0.0, 0.0, 600.0])

        for i in range(25):
            t = i * 0.033
            y = 0.5 * 2500.0 * (t ** 2)
            gate.update(t, base_tvec + np.array([0.0, y, 0.0]), (0.0, 0.0, 0.0))

        assert gate.continuous_vibration_active is True

        gate.reset()
        assert gate.continuous_vibration_active is False
        assert gate.is_frozen is False
        assert gate._duty_cycle_exceeded is False


class TestFrameCaptureHardwareShutterControl:
    """Verify hardware shutter and buffer control flags in FrameCapture."""

    @pytest.fixture(autouse=True)
    def patch_cv2(self, monkeypatch):
        from unittest.mock import MagicMock, patch
        self.mock_cap = MagicMock()
        self.mock_cap.isOpened.return_value = True
        self.mock_cap.read.return_value = (True, np.zeros((720, 1280, 3), dtype=np.uint8))
        self.mock_cap.get.return_value = 60.0

    def test_live_camera_sets_fps_and_buffersize(self):
        """Live camera capture must request target_fps and buffersize=1 to eliminate frame lag."""
        from unittest.mock import patch
        import cv2
        from pipeline.frame_capture import FrameCapture

        with patch("cv2.VideoCapture", return_value=self.mock_cap):
            with FrameCapture(source=0, width=1280, height=720, target_fps=60.0, anti_motion_blur=True) as cap:
                assert cap.target_fps == 60.0
                assert cap.anti_motion_blur is True
                self.mock_cap.set.assert_any_call(cv2.CAP_PROP_FPS, 60.0)
                self.mock_cap.set.assert_any_call(cv2.CAP_PROP_BUFFERSIZE, 1)

    def test_video_file_does_not_set_buffersize(self):
        """Video file playback must NOT alter buffersize or override frame rate."""
        from unittest.mock import patch
        import cv2
        from pipeline.frame_capture import FrameCapture

        with patch("cv2.VideoCapture", return_value=self.mock_cap):
            with FrameCapture(source="tests/sample_driver_video.mp4") as cap:
                calls = [c.args for c in self.mock_cap.set.mock_calls]
                assert (cv2.CAP_PROP_BUFFERSIZE, 1) not in calls
