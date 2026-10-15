"""Unit tests for KinematicStabilizer: zero-blur road shock & vibration estimation."""
import cv2
import numpy as np
import pytest
from pipeline.stabilizer import KinematicStabilizer


class TestKinematicStabilizer:

    def test_initialization_and_reset(self):
        stabilizer = KinematicStabilizer(history_len=10)
        assert stabilizer.last_dx == 0.0
        assert stabilizer.last_dy == 0.0
        assert stabilizer.vibration_score == 0.0
        assert stabilizer.is_bumpy is False

        # Simulate state
        stabilizer.last_dx = 5.0
        stabilizer.last_dy = 15.0
        stabilizer.is_bumpy = True
        stabilizer.vibration_score = 0.8
        stabilizer._dx_history.append(5.0)
        stabilizer._dy_history.append(15.0)

        stabilizer.reset()
        assert stabilizer.last_dx == 0.0
        assert stabilizer.last_dy == 0.0
        assert stabilizer.vibration_score == 0.0
        assert stabilizer.is_bumpy is False
        assert len(stabilizer._dx_history) == 0
        assert len(stabilizer._dy_history) == 0

    def test_degenerate_inputs_handled_safely(self):
        stabilizer = KinematicStabilizer()
        assert stabilizer.estimate_frame_shift(None) == (0.0, 0.0, False)
        assert stabilizer.estimate_frame_shift(np.zeros((0, 0), dtype=np.uint8)) == (0.0, 0.0, False)
        assert stabilizer.estimate_frame_shift(np.zeros((10, 10), dtype=np.uint8)) == (0.0, 0.0, False)

    def test_speed_bump_vertical_jolt_detected(self):
        stabilizer = KinematicStabilizer()
        h, w = 240, 320
        # Frame 1: base frame with a high-contrast pattern
        f1 = np.zeros((h, w, 3), dtype=np.uint8)
        cv2.circle(f1, (160, 120), 40, (255, 255, 255), -1)
        cv2.rectangle(f1, (50, 50), (100, 100), (200, 200, 200), -1)

        dx1, dy1, bumpy1 = stabilizer.estimate_frame_shift(f1)
        assert (dx1, dy1, bumpy1) == (0.0, 0.0, False)

        # Frame 2: sudden vertical jump by +16px (speed bump)
        f2 = np.zeros((h, w, 3), dtype=np.uint8)
        cv2.circle(f2, (160, 136), 40, (255, 255, 255), -1)
        cv2.rectangle(f2, (50, 66), (100, 116), (200, 200, 200), -1)

        dx2, dy2, bumpy2 = stabilizer.estimate_frame_shift(f2)
        assert abs(dx2) < 3.0
        assert abs(dy2 - 16.0) < 4.0
        assert bumpy2 is True, "Vertical shift > 12px must flag is_bumpy=True"

    def test_high_frequency_vibration_energy(self):
        stabilizer = KinematicStabilizer(history_len=15)
        # Feed alternating positive and negative vertical shifts simulating 8-12 Hz vibration
        vibrations = [2.5, -2.8, 3.0, -2.7, 3.2, -3.1, 2.9, -2.6]
        for dy in vibrations:
            stabilizer._dy_history.append(dy)

        # Force vibration score calculation
        dys = np.array(stabilizer._dy_history, dtype=np.float64)
        signs = np.sign(dys[np.abs(dys) > 0.6])
        reversals = float(np.sum(signs[1:] * signs[:-1] < 0)) / float(len(signs) - 1)
        rms_dy = float(np.sqrt(np.mean(dys ** 2)))
        vibration_score = float(np.clip(reversals * (rms_dy / 3.0), 0.0, 1.0))

        assert reversals > 0.8, "Alternating signs must yield high reversal rate"
        assert vibration_score > 0.5, "High frequency sign reversals must yield vibration_score > 0.5"
