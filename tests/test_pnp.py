"""Unit tests for PnP face normalization.

Tests verify:
  - PnP solver finds a valid pose from synthetic frontal landmarks
  - Euler angles are near-zero for a frontal face
  - Normalized internal distances (EAR-relevant) remain constant
    when the entire face is translated (simulating vehicle bounce)
  - Normalization handles missing pose gracefully
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest
from pipeline.pnp_normalizer import PnPNormalizer
from utils.landmarks import (
    PNP_LANDMARK_INDICES, CANONICAL_FACE_3D,
    RIGHT_EYE_EAR_FLAT,
)


def make_frontal_face_landmarks(frame_width=1280, frame_height=720,
                                 offset_x=0.0, offset_y=0.0):
    """Create synthetic landmarks for a roughly frontal face.

    Places the 6 PnP anchor points at approximate pixel positions
    for a centered frontal face, with optional translation offset
    (simulating vehicle bounce).
    """
    landmarks = np.zeros((478, 3), dtype=np.float64)

    cx = frame_width / 2.0 + offset_x
    cy = frame_height / 2.0 + offset_y

    # Approximate pixel positions for a frontal face at ~60cm distance
    # Matching PNP_LANDMARK_INDICES order:
    # [NOSE_TIP=4, CHIN=152, RIGHT_EYE_OUTER=33,
    #  LEFT_EYE_OUTER=263, RIGHT_MOUTH=61, LEFT_MOUTH=291]
    positions = {
        4:   [cx, cy - 30, 0],           # Nose tip
        152: [cx, cy + 120, 0],          # Chin
        33:  [cx - 80, cy - 70, 0],      # Right eye outer
        263: [cx + 80, cy - 70, 0],      # Left eye outer
        61:  [cx - 50, cy + 50, 0],      # Right mouth
        291: [cx + 50, cy + 50, 0],      # Left mouth
    }

    for idx, pos in positions.items():
        landmarks[idx] = pos

    # Add some eye landmarks for EAR testing
    # Right eye (around idx 33 at cx-80, cy-70)
    eye_cx = cx - 80
    eye_cy = cy - 70
    eye_w = 20.0
    eye_h = 6.0
    landmarks[33]  = [eye_cx - eye_w, eye_cy, 0]       # P1 outer
    landmarks[160] = [eye_cx - eye_w*0.3, eye_cy - eye_h, 0]  # P2 upper1
    landmarks[158] = [eye_cx + eye_w*0.3, eye_cy - eye_h, 0]  # P3 upper2
    landmarks[133] = [eye_cx + eye_w, eye_cy, 0]       # P4 inner
    landmarks[153] = [eye_cx + eye_w*0.3, eye_cy + eye_h, 0]  # P5 lower2
    landmarks[144] = [eye_cx - eye_w*0.3, eye_cy + eye_h, 0]  # P6 lower1

    return landmarks


def compute_internal_distances(landmarks: np.ndarray) -> np.ndarray:
    """Compute distances between EAR landmarks (internal to the eye)."""
    indices = RIGHT_EYE_EAR_FLAT
    pts = landmarks[indices]
    # P2-P6 distance and P3-P5 distance
    d1 = np.linalg.norm(pts[1] - pts[5])  # P2-P6
    d2 = np.linalg.norm(pts[2] - pts[4])  # P3-P5
    d3 = np.linalg.norm(pts[0] - pts[3])  # P1-P4
    return np.array([d1, d2, d3])


class TestPnPNormalizer:
    """Test suite for PnP face normalization."""

    def test_pnp_solves_frontal_face(self):
        """PnP should find a valid solution for a frontal face."""
        normalizer = PnPNormalizer(frame_width=1280, frame_height=720)
        landmarks = make_frontal_face_landmarks()

        success = normalizer.solve_pose(landmarks)
        assert success, "PnP failed on frontal face landmarks"
        assert normalizer.rvec is not None
        assert normalizer.tvec is not None
        assert normalizer.rotation_matrix is not None

    def test_euler_angles_near_zero_for_frontal(self):
        """A frontal face should produce near-zero Euler angles."""
        normalizer = PnPNormalizer(frame_width=1280, frame_height=720)
        landmarks = make_frontal_face_landmarks()
        normalizer.solve_pose(landmarks)

        euler = normalizer.get_euler_angles()
        assert euler is not None
        pitch, yaw, roll = euler

        # Allow generous ±20° tolerance since synthetic points aren't perfect
        assert abs(yaw) < 20.0, f"Frontal face yaw={yaw:.1f}°, expected near 0"
        assert abs(pitch) < 30.0, f"Frontal face pitch={pitch:.1f}°, expected near 0"
        assert abs(roll) < 20.0, f"Frontal face roll={roll:.1f}°, expected near 0"

    def test_translation_does_not_change_internal_distances(self):
        """Vehicle bounce (translation) should not change internal eye distances.

        This is THE core test: if we translate the entire face by (dx, dy),
        the normalized EAR-relevant distances should remain constant.
        """
        normalizer = PnPNormalizer(frame_width=1280, frame_height=720)

        # Baseline: centered face
        landmarks_center = make_frontal_face_landmarks(offset_x=0, offset_y=0)
        normalizer.solve_pose(landmarks_center)
        norm_center = normalizer.normalize_landmarks(landmarks_center)
        dists_center = compute_internal_distances(norm_center)

        # Bounced face: shifted 50px down (simulating pothole)
        landmarks_bounced = make_frontal_face_landmarks(offset_x=0, offset_y=50)
        normalizer.solve_pose(landmarks_bounced)
        norm_bounced = normalizer.normalize_landmarks(landmarks_bounced)
        dists_bounced = compute_internal_distances(norm_bounced)

        # Internal distances should be nearly identical
        np.testing.assert_allclose(
            dists_center, dists_bounced, rtol=0.15,
            err_msg="Vehicle bounce changed internal eye distances! "
                    f"Center={dists_center}, Bounced={dists_bounced}")

    def test_normalization_without_pose_returns_copy(self):
        """If no pose is solved, normalize should return a copy of input."""
        normalizer = PnPNormalizer()
        landmarks = np.random.randn(478, 3)

        result = normalizer.normalize_landmarks(landmarks)
        np.testing.assert_array_equal(result, landmarks)
        # Verify it's a copy, not the same object
        assert result is not landmarks

    def test_euler_angles_none_without_pose(self):
        """get_euler_angles should return None before solve_pose is called."""
        normalizer = PnPNormalizer()
        assert normalizer.get_euler_angles() is None

    def test_frame_size_update(self):
        """Camera matrix should update when frame size changes."""
        normalizer = PnPNormalizer(frame_width=640, frame_height=480)
        assert normalizer.camera_matrix[0, 2] == 320.0  # cx

        normalizer.update_frame_size(1280, 720)
        assert normalizer.camera_matrix[0, 2] == 640.0  # cx updated


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
