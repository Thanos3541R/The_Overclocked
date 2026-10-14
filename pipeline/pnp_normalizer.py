"""PnP-based rigid face normalization for vehicle bounce cancellation.

Establishes a canonical face coordinate system using skull-bound anchor landmarks
and cv2.solvePnP. When the vehicle hits a bump, the entire coordinate box shifts
together, but internal distances (e.g., eyelid separation for EAR) remain constant.

Pipeline:
  1. Extract 2D pixel coords for rigid anchor landmarks (8-point expanded or 6-point canonical)
  2. Solve PnP using cv2.solvePnP with SOLVEPNP_ITERATIVE and useExtrinsicGuess=True
  3. Fall back to deterministic direct solvers (SOLVEPNP_SQPNP, SOLVEPNP_EPNP) on acquisition or failure
  4. Invalidate stale pose (rvec, tvec, rotation_matrix = None) on any solve failure
  5. Single-frame Euler jump rejection (>25 deg) with persistence escape (>= 2 frames)
  6. Transform landmarks from camera-space to face-centered space while preserving pixel coordinate range
"""
from typing import Optional, Tuple
import cv2
import numpy as np

from utils.landmarks import (
    CANONICAL_FACE_3D,
    CANONICAL_FACE_3D_8,
    MID_EYEBROW,
    NOSE_BRIDGE,
    NOSE_TIP,
    PNP_LANDMARK_INDICES,
    PNP_LANDMARK_INDICES_8,
)


class PnPNormalizer:
    """Rigid face normalization via Perspective-n-Point with pose jump rejection."""

    def __init__(self, frame_width: int = 1280, frame_height: int = 720,
                 focal_length: Optional[float] = None,
                 max_euler_jump_deg: float = 25.0,
                 max_jump_rejections: int = 2):
        """
        Args:
            frame_width: Camera frame width in pixels.
            frame_height: Camera frame height in pixels.
            focal_length: Camera focal length in pixels.
                          If None, approximated as frame_width.
            max_euler_jump_deg: Max allowable Euler angle delta between consecutive frames (degrees)
                                to reject single-frame PnP flips (e.g. Frame 113 artifact).
            max_jump_rejections: Number of consecutive rejected frames before accepting the new pose
                                 as a genuine fast head turn, preventing tracker deadlock.
        """
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.max_euler_jump_deg = float(max_euler_jump_deg)
        self.max_jump_rejections = int(max_jump_rejections)

        f = focal_length if focal_length is not None else float(frame_width)
        cx = frame_width / 2.0
        cy = frame_height / 2.0

        self.camera_matrix = np.array([
            [f,   0.0, cx],
            [0.0, f,   cy],
            [0.0, 0.0, 1.0],
        ], dtype=np.float64)

        self.dist_coeffs = np.zeros((4, 1), dtype=np.float64)

        # Stored pose state for current frame (strictly None when pose fails)
        self.rvec: Optional[np.ndarray] = None
        self.tvec: Optional[np.ndarray] = None
        self.rotation_matrix: Optional[np.ndarray] = None
        self._last_valid_euler: Optional[Tuple[float, float, float]] = None
        self._jump_reject_count = 0

    def reset(self) -> None:
        """Explicitly clear stored pose state."""
        self.rvec = None
        self.tvec = None
        self.rotation_matrix = None
        self._last_valid_euler = None
        self._jump_reject_count = 0

    @staticmethod
    def _extract_euler(rotation_matrix: np.ndarray) -> Tuple[float, float, float]:
        """Extract pitch, yaw, roll from a 3x3 rotation matrix via RQ decomposition."""
        angles, _, _, _, _, _ = cv2.RQDecomp3x3(rotation_matrix)
        return float(angles[0]), float(angles[1]), float(angles[2])

    def solve_pose(self, landmarks: np.ndarray) -> bool:
        """Solve head pose from landmarks.

        Uses robust cv2.solvePnP with SOLVEPNP_ITERATIVE + extrinsic guess when a
        previous valid pose exists, falling back to SOLVEPNP_SQPNP / SOLVEPNP_EPNP
        on initial acquisition or tracking loss.

        Stale pose vectors are immediately invalidated (set to None) on failure so
        old vectors never freeze or persist across failed frames.

        Args:
            landmarks: Full landmark array in pixel coords (e.g. 478×3 or 478×2).

        Returns:
            True if PnP solution found for this frame, False otherwise.
        """
        # Save previous valid pose for extrinsic guess and jump rejection
        prev_rvec = self.rvec.copy() if self.rvec is not None else None
        prev_tvec = self.tvec.copy() if self.tvec is not None else None
        prev_euler = self._last_valid_euler

        # Invalidate stale pose immediately on every frame
        self.rvec = None
        self.tvec = None
        self.rotation_matrix = None

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

        # Determine if expanded 8-point rigid anchor set is available and non-zero
        use_8 = False
        if lms.shape[0] > max(NOSE_BRIDGE, MID_EYEBROW):
            pt_bridge = lms[NOSE_BRIDGE, :2]
            pt_sellion = lms[MID_EYEBROW, :2]
            if (np.linalg.norm(pt_bridge) > 1.0 and np.linalg.norm(pt_sellion) > 1.0 and
                    not np.allclose(pt_bridge, pt_sellion)):
                use_8 = True

        if use_8:
            object_points = CANONICAL_FACE_3D_8
            image_points = lms[PNP_LANDMARK_INDICES_8, :2].astype(np.float64)
        else:
            object_points = CANONICAL_FACE_3D
            image_points = lms[PNP_LANDMARK_INDICES, :2].astype(np.float64)

        # Reject degenerate/collapsed landmarks (e.g. all zeros, collapsed, or <10px bounding box)
        pt_span_x = float(np.ptp(image_points[:, 0]))
        pt_span_y = float(np.ptp(image_points[:, 1]))
        if pt_span_x < 10.0 or pt_span_y < 10.0:
            return False

        success = False
        rvec = None
        tvec = None

        # Step A: Temporal tracking with SOLVEPNP_ITERATIVE using previous pose as guess
        if prev_rvec is not None and prev_tvec is not None:
            try:
                r_guess = prev_rvec.copy()
                t_guess = prev_tvec.copy()
                ret, r_sol, t_sol = cv2.solvePnP(
                    object_points,
                    image_points,
                    self.camera_matrix,
                    self.dist_coeffs,
                    rvec=r_guess,
                    tvec=t_guess,
                    useExtrinsicGuess=True,
                    flags=cv2.SOLVEPNP_ITERATIVE,
                )
                if ret and np.all(np.isfinite(r_sol)) and np.all(np.isfinite(t_sol)):
                    z_val = float(t_sol[2, 0] if t_sol.ndim == 2 else t_sol[2])
                    if z_val > 0.0:
                        success = True
                        rvec = r_sol
                        tvec = t_sol
            except cv2.error:
                success = False

        # Step B: Direct solver fallback (SOLVEPNP_SQPNP or SOLVEPNP_EPNP)
        if not success:
            for flag in (cv2.SOLVEPNP_SQPNP, cv2.SOLVEPNP_EPNP):
                try:
                    ret, r_sol, t_sol = cv2.solvePnP(
                        object_points,
                        image_points,
                        self.camera_matrix,
                        self.dist_coeffs,
                        flags=flag,
                    )
                    if ret and np.all(np.isfinite(r_sol)) and np.all(np.isfinite(t_sol)):
                        z_val = float(t_sol[2, 0] if t_sol.ndim == 2 else t_sol[2])
                        if z_val > 0.0:
                            # Optional refinement with iterative
                            try:
                                ret_ref, r_ref, t_ref = cv2.solvePnP(
                                    object_points,
                                    image_points,
                                    self.camera_matrix,
                                    self.dist_coeffs,
                                    rvec=r_sol.copy(),
                                    tvec=t_sol.copy(),
                                    useExtrinsicGuess=True,
                                    flags=cv2.SOLVEPNP_ITERATIVE,
                                )
                                if ret_ref and np.all(np.isfinite(r_ref)) and np.all(np.isfinite(t_ref)):
                                    r_sol, t_sol = r_ref, t_ref
                            except Exception:
                                pass
                            success = True
                            rvec = r_sol
                            tvec = t_sol
                            break
                except cv2.error:
                    continue

        if not success or rvec is None or tvec is None:
            self.rvec = None
            self.tvec = None
            self.rotation_matrix = None
            return False

        # Extract candidate rotation matrix and Euler angles
        cand_rot_matrix, _ = cv2.Rodrigues(rvec)
        cand_pitch, cand_yaw, cand_roll = self._extract_euler(cand_rot_matrix)

        # Step C: Single-frame jump rejection defense with persistence escape
        if prev_euler is not None:
            p_prev, y_prev, r_prev = prev_euler
            dp = abs(cand_pitch - p_prev)
            dy = abs(cand_yaw - y_prev)
            dr = abs(cand_roll - r_prev)
            # Handle angle wraparound at +-180 deg
            if dp > 180.0: dp = 360.0 - dp
            if dy > 180.0: dy = 360.0 - dy
            if dr > 180.0: dr = 360.0 - dr

            if max(dp, dy, dr) > self.max_euler_jump_deg:
                self._jump_reject_count += 1
                if self._jump_reject_count < self.max_jump_rejections:
                    # Single-frame transient glitch -> reject
                    return False
                # Jump persisted for >= max_jump_rejections frames -> accept real fast head turn
                self._jump_reject_count = 0
            else:
                self._jump_reject_count = 0
        else:
            self._jump_reject_count = 0

        self.rvec = rvec
        self.tvec = tvec
        self.rotation_matrix = cand_rot_matrix
        self._last_valid_euler = (cand_pitch, cand_yaw, cand_roll)
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
        if self.rotation_matrix is None or self._last_valid_euler is None:
            return None

        return self._last_valid_euler

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
