"""Unit tests for Eye Aspect Ratio (EAR) computation.

Tests the EAR formula with synthetic landmarks to verify:
  - Open eyes produce EAR ≈ 0.30
  - Closed eyes produce EAR ≈ 0.05
  - Half-closed eyes produce intermediate EAR
  - Zero-width eyes don't crash (division by zero guard)
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest
from pipeline.feature_extractor import FeatureExtractor
from utils.landmarks import RIGHT_EYE_EAR_FLAT, LEFT_EYE_EAR_FLAT


def make_eye_landmarks(eye_indices, horizontal_width=40.0,
                       vertical_opening=12.0, center=(200.0, 200.0)):
    """Create synthetic landmarks for a single eye.

    Builds a (478, 3) array with the 6 EAR points positioned to simulate
    an eye with given dimensions.

    P1 (outer) ──────── P4 (inner)
         P2    P3     (upper lid)
         P6    P5     (lower lid)
    """
    landmarks = np.zeros((478, 3), dtype=np.float64)
    cx, cy = center

    half_w = horizontal_width / 2.0
    half_v = vertical_opening / 2.0

    # P1: outer corner (left side)
    landmarks[eye_indices[0]] = [cx - half_w, cy, 0]
    # P2: upper lid pair 1
    landmarks[eye_indices[1]] = [cx - half_w * 0.3, cy - half_v, 0]
    # P3: upper lid pair 2
    landmarks[eye_indices[2]] = [cx + half_w * 0.3, cy - half_v, 0]
    # P4: inner corner (right side)
    landmarks[eye_indices[3]] = [cx + half_w, cy, 0]
    # P5: lower lid pair 2
    landmarks[eye_indices[4]] = [cx + half_w * 0.3, cy + half_v, 0]
    # P6: lower lid pair 1
    landmarks[eye_indices[5]] = [cx - half_w * 0.3, cy + half_v, 0]

    return landmarks


class TestEAR:
    """Test suite for Eye Aspect Ratio computation."""

    def test_open_eye_ear(self):
        """Open eye should produce EAR ≈ 0.30."""
        landmarks = make_eye_landmarks(
            RIGHT_EYE_EAR_FLAT,
            horizontal_width=40.0,
            vertical_opening=12.0,  # ~30% of width
        )
        ear = FeatureExtractor.compute_ear(landmarks, RIGHT_EYE_EAR_FLAT)
        assert 0.25 <= ear <= 0.35, f"Open eye EAR={ear:.4f}, expected 0.25-0.35"

    def test_closed_eye_ear(self):
        """Closed eye should produce EAR ≈ 0.05."""
        landmarks = make_eye_landmarks(
            RIGHT_EYE_EAR_FLAT,
            horizontal_width=40.0,
            vertical_opening=1.0,  # nearly closed
        )
        ear = FeatureExtractor.compute_ear(landmarks, RIGHT_EYE_EAR_FLAT)
        assert ear < 0.10, f"Closed eye EAR={ear:.4f}, expected < 0.10"

    def test_half_closed_ear(self):
        """Half-closed eye should produce intermediate EAR."""
        landmarks = make_eye_landmarks(
            RIGHT_EYE_EAR_FLAT,
            horizontal_width=40.0,
            vertical_opening=6.0,
        )
        ear = FeatureExtractor.compute_ear(landmarks, RIGHT_EYE_EAR_FLAT)
        assert 0.10 <= ear <= 0.25, f"Half-closed EAR={ear:.4f}, expected 0.10-0.25"

    def test_left_eye_ear(self):
        """Left eye indices should also produce valid EAR."""
        landmarks = make_eye_landmarks(
            LEFT_EYE_EAR_FLAT,
            horizontal_width=40.0,
            vertical_opening=12.0,
            center=(400.0, 200.0),
        )
        ear = FeatureExtractor.compute_ear(landmarks, LEFT_EYE_EAR_FLAT)
        assert 0.25 <= ear <= 0.35, f"Left eye EAR={ear:.4f}, expected 0.25-0.35"

    def test_zero_width_eye(self):
        """Zero-width eye should return 0.0 (not crash)."""
        landmarks = np.zeros((478, 3), dtype=np.float64)
        # All points at same location = zero width
        for idx in RIGHT_EYE_EAR_FLAT:
            landmarks[idx] = [100, 100, 0]
        ear = FeatureExtractor.compute_ear(landmarks, RIGHT_EYE_EAR_FLAT)
        assert ear == 0.0, f"Zero-width EAR={ear:.4f}, expected 0.0"

    def test_symmetric_eyes_equal(self):
        """Symmetric left and right eyes should have equal EAR."""
        r_landmarks = make_eye_landmarks(
            RIGHT_EYE_EAR_FLAT, horizontal_width=38.0,
            vertical_opening=11.0, center=(150, 200))
        l_landmarks = make_eye_landmarks(
            LEFT_EYE_EAR_FLAT, horizontal_width=38.0,
            vertical_opening=11.0, center=(350, 200))

        # Merge into one array
        combined = np.zeros((478, 3), dtype=np.float64)
        for idx in RIGHT_EYE_EAR_FLAT:
            combined[idx] = r_landmarks[idx]
        for idx in LEFT_EYE_EAR_FLAT:
            combined[idx] = l_landmarks[idx]

        ear_r = FeatureExtractor.compute_ear(combined, RIGHT_EYE_EAR_FLAT)
        ear_l = FeatureExtractor.compute_ear(combined, LEFT_EYE_EAR_FLAT)
        assert abs(ear_r - ear_l) < 0.01, \
            f"Symmetric eyes should match: R={ear_r:.4f}, L={ear_l:.4f}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
