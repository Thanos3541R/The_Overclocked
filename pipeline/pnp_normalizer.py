"""PnP-based rigid face normalization for vehicle bounce cancellation.

The core trick: establishes a canonical face coordinate system using
skull-bound anchor landmarks and cv2.solvePnPRansac. When the vehicle
hits a bump, the entire coordinate box shifts together, but internal
distances (e.g., eyelid separation for EAR) remain constant.

Pipeline:
  1. Extract 2D pixel coords for 6 rigid anchor landmarks
  2. Solve PnP against a canonical 3D anthropometric face model
  3. Obtain rotation (rvec) and translation (tvec)
  4. Transform all 478 landmarks from camera-space to face-centered space
  5. Vehicle bounce cancels out; only facial muscle movements remain
"""
import cv2
import numpy as np
from typing import Optional, Tuple
from utils.landmarks import (PNP_LANDMARK_INDICES, CANONICAL_FACE_3D, NOSE_TIP)


class PnPNormalizer:
    """Rigid face normalization via Perspective-n-Point."""

    def __init__(self, frame_width: int = 1280, frame_height: int = 720,
                 focal_length: Optional[float] = None):
        """
        Args:
            frame_width: Camera frame width in pixels.
            frame_height: Camera frame height in pixels.
            focal_length: Camera focal length in pixels.
                          If None, approximated as frame_width.
        """
        self.frame_width = frame_width
        self.frame_height = frame_height

        f = focal_length if focal_length is not None else float(frame_width)
        cx = frame_width / 2.0
        cy = frame_height / 2.0

        self.camera_matrix = np.array([
            [f,   0.0, cx],
            [0.0, f,   cy],
            [0.0, 0.0, 1.0],
        ], dtype=np.float64)

        self.dist_coeffs = np.zeros((4, 1), dtype=np.float64)

        # Store last valid pose for gating
        self.rvec: Optional[np.ndarray] = None
        self.tvec: Optional[np.ndarray] = None
        self.rotation_matrix: Optional[np.ndarray] = None

    def solve_pose(self, landmarks: np.ndarray) -> bool:
        """Solve head pose from landmarks.

        Args:
            landmarks: Full landmark array in pixel coords (e.g. 478×3 or 478×2).

        Returns:
            True if PnP solution found, False otherwise.
        """
        if landmarks is None:
            return False

        try:
            lms = np.asarray(landmarks, dtype=np.float64)
        except Exception:
            return False

        if (lms.ndim != 2 or
                lms.shape[0] <= max(PNP_LANDMARK_INDICES) or
                lms.shape[1] < 2 or
                np.isnan(lms).any() or
                np.isinf(lms).any()):
            return False

        image_points = lms[PNP_LANDMARK_INDICES, :2]

        try:
            success, rvec, tvec, inliers = cv2.solvePnPRansac(
                CANONICAL_FACE_3D,
                image_points,
                self.camera_matrix,
                self.dist_coeffs,
                flags=cv2.SOLVEPNP_ITERATIVE,
            )
        except cv2.error:
            return False

        if not success or inliers is None or len(inliers) < 4:
            return False

        self.rvec = rvec
        self.tvec = tvec
        self.rotation_matrix, _ = cv2.Rodrigues(rvec)
        return True

    def normalize_landmarks(self, landmarks: np.ndarray) -> np.ndarray:
        """Transform landmarks to face-aligned space while preserving pixel coordinates for visual sampling.

        Removes vehicle bounce: translates coordinates relative to the face origin (nose tip) in pixel space,
        applies canonical rotation alignment, and anchors back to pixel coordinates so downstream visual sampling
        (e.g., cheek flushing patches and eye crops) remains perfectly registered on the image frame.

        Args:
            landmarks: Full landmark array in pixel coords (e.g. 478×3 or 478×2).

        Returns:
            Normalized landmarks in stabilized pixel coordinates.
            If no valid pose, returns landmarks unchanged.
        """
        if landmarks is None:
            return None

        try:
            lms = np.asarray(landmarks)
        except Exception:
            return landmarks.copy() if hasattr(landmarks, "copy") else landmarks

        if self.rotation_matrix is None or lms.ndim != 2 or lms.shape[0] == 0 or lms.shape[1] < 2:
            return lms.copy() if isinstance(landmarks, np.ndarray) else (landmarks.copy() if hasattr(landmarks, "copy") else landmarks)

        # Anchor origin in pixel space (nose tip if present, otherwise centroid)
        if lms.shape[0] > NOSE_TIP:
            origin = lms[NOSE_TIP].copy()
        else:
            origin = np.mean(lms, axis=0)

        n_dim = lms.shape[1]
        normalized = lms.copy()

        if n_dim >= 3:
            centered = lms[:, :3] - origin[:3]
            rotated = (self.rotation_matrix.T @ centered.T).T
            normalized[:, :3] = rotated + origin[:3]
        elif n_dim == 2:
            centered_3d = np.column_stack([lms[:, :2] - origin[:2], np.zeros(len(lms))])
            rotated_3d = (self.rotation_matrix.T @ centered_3d.T).T
            normalized[:, :2] = rotated_3d[:, :2] + origin[:2]

        return normalized

    def get_euler_angles(self) -> Optional[Tuple[float, float, float]]:
        """Extract pitch, yaw, roll from the current rotation matrix.

        Returns:
            (pitch, yaw, roll) in degrees, or None if no valid pose.
        """
        if self.rotation_matrix is None:
            return None

        angles, _, _, _, _, _ = cv2.RQDecomp3x3(self.rotation_matrix)
        pitch, yaw, roll = angles[0], angles[1], angles[2]
        return pitch, yaw, roll

    def update_frame_size(self, width: int, height: int):
        """Update camera parameters if frame size changes."""
        if width != self.frame_width or height != self.frame_height:
            self.frame_width = width
            self.frame_height = height
            f = float(width)
            self.camera_matrix[0, 0] = f
            self.camera_matrix[1, 1] = f
            self.camera_matrix[0, 2] = width / 2.0
            self.camera_matrix[1, 2] = height / 2.0
