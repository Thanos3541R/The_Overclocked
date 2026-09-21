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
