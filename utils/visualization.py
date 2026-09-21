"""Debug visualization overlays for the driver monitoring pipeline.

Draws real-time annotations on the camera feed:
  - Face mesh wireframe
  - Eye contour highlights with EAR values
  - Iris crosshairs for gaze visualization
  - Head pose axis arrows (RGB = XYZ)
  - MAR, PERCLOS, FPS text overlays
  - Gating status border (green = valid, red = gated)
"""
import cv2
import numpy as np
from typing import Optional
import time

from utils.landmarks import (
    RIGHT_EYE_CONTOUR, LEFT_EYE_CONTOUR,
    RIGHT_IRIS_CENTER, LEFT_IRIS_CENTER,
    RIGHT_IRIS_RING, LEFT_IRIS_RING,
    INNER_LIP_UPPER, INNER_LIP_LOWER,
    PNP_LANDMARK_INDICES,
)


class Visualizer:
    """Debug overlay renderer for the DMS pipeline."""

    def __init__(self, show_mesh: bool = True,
                 show_iris: bool = True,
                 show_pose_axes: bool = True):
        self.show_mesh = show_mesh
        self.show_iris = show_iris
        self.show_pose_axes = show_pose_axes
        self._prev_time = time.time()
        self._fps = 0.0
        self._fps_alpha = 0.1  # EMA smoothing for FPS display

    def draw(self, frame: np.ndarray,
             landmarks: Optional[np.ndarray],
             features=None,
             pnp_normalizer=None) -> np.ndarray:
        """Draw all debug overlays on the frame.

        Args:
            frame: BGR frame to draw on (modified in-place).
            landmarks: Raw pixel-space landmarks (478, 3), or None.
            features: FrameFeatures dataclass, or None.
            pnp_normalizer: PnPNormalizer instance for pose axes.

        Returns:
            Annotated frame.
        """
        # Update FPS
        now = time.time()
        dt = now - self._prev_time
        if dt > 0:
            instant_fps = 1.0 / dt
            self._fps = self._fps * (1 - self._fps_alpha) + instant_fps * self._fps_alpha
        self._prev_time = now

        if landmarks is not None:
            pts = landmarks[:, :2].astype(np.int32)

            # ── Eye contours (cyan) ──
            self._draw_contour(frame, pts, RIGHT_EYE_CONTOUR, (255, 255, 0))
            self._draw_contour(frame, pts, LEFT_EYE_CONTOUR, (255, 255, 0))

            # ── Iris crosshairs (yellow) ──
            if self.show_iris:
                for iris_idx in [LEFT_IRIS_CENTER, RIGHT_IRIS_CENTER]:
                    cx, cy = int(pts[iris_idx][0]), int(pts[iris_idx][1])
                    cv2.drawMarker(frame, (cx, cy), (0, 255, 255),
                                  cv2.MARKER_CROSS, 8, 1)

                for ring in [LEFT_IRIS_RING, RIGHT_IRIS_RING]:
                    ring_pts = pts[ring].reshape(-1, 1, 2)
                    cv2.polylines(frame, [ring_pts], True, (0, 255, 255), 1)

            # ── Mouth contour (magenta) ──
            self._draw_contour(frame, pts, INNER_LIP_UPPER, (255, 0, 255))
            self._draw_contour(frame, pts, INNER_LIP_LOWER, (255, 0, 255))

            # ── Head pose axes ──
            if self.show_pose_axes and pnp_normalizer is not None:
                self._draw_pose_axes(frame, pnp_normalizer)

        # ── Gating border ──
        if features is not None:
            color = (0, 255, 0) if features.face_valid else (0, 0, 255)
            h, w = frame.shape[:2]
            cv2.rectangle(frame, (0, 0), (w - 1, h - 1), color, 3)

        # ── Text overlay ──
        self._draw_text_panel(frame, features)

        return frame

    def _draw_contour(self, frame: np.ndarray, pts: np.ndarray,
                      indices: list, color: tuple):
        """Draw a polyline contour."""
        contour_pts = pts[indices].reshape(-1, 1, 2)
        cv2.polylines(frame, [contour_pts], True, color, 1, cv2.LINE_AA)

    def _draw_pose_axes(self, frame: np.ndarray, normalizer):
        """Draw RGB head pose axes at the nose tip."""
        if normalizer.rvec is None or normalizer.tvec is None:
            return

        axis_length = 100.0
        axes_3d = np.float64([
            [axis_length, 0, 0],   # X (Red)
            [0, axis_length, 0],   # Y (Green)
            [0, 0, axis_length],   # Z (Blue)
            [0, 0, 0],             # Origin
        ])

        img_pts, _ = cv2.projectPoints(
            axes_3d, normalizer.rvec, normalizer.tvec,
            normalizer.camera_matrix, normalizer.dist_coeffs)

        origin = tuple(img_pts[3].ravel().astype(int))
        colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0)]  # BGR for XYZ
        for i, color in enumerate(colors):
            end = tuple(img_pts[i].ravel().astype(int))
            cv2.arrowedLine(frame, origin, end, color, 2, cv2.LINE_AA)

    def _draw_text_panel(self, frame: np.ndarray, features):
        """Draw feature values as text overlay."""
        h, w = frame.shape[:2]
        y_offset = 25
        x = 10
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.55
        color = (255, 255, 255)
        bg_color = (0, 0, 0)
        thickness = 1

        lines = [f"FPS: {self._fps:.1f}"]

        if features is not None:
            lines.extend([
                f"EAR L:{features.ear_left:.3f} R:{features.ear_right:.3f} Avg:{features.ear_avg:.3f}",
                f"MAR: {features.mar:.3f}",
                f"PERCLOS: {features.perclos:.3f}",
                f"Yaw:{features.head_yaw:.1f} Pitch:{features.head_pitch:.1f} Roll:{features.head_roll:.1f}",
                f"Saccade: {features.saccade_velocity:.1f} deg/s (lat: {features.saccade_latency_ms:.0f}ms)",
                f"Blink Vel: dn {features.blink_closing_velocity:.2f} | up {features.blink_opening_velocity:.2f}",
                f"Gaze Yaw Disp (sigma): {features.gaze_yaw_dispersion:.3f}",
                f"Valid: {features.face_valid}" + (f" ({features.gating_reason})" if features.gating_reason else ""),
            ])

            if features.saccade_detected:
                lines.append(">>> SACCADE <<<")
            if features.blink_detected:
                lines.append(">>> BLINK <<<")
            if features.microsleep_detected:
                lines.append("!!! MICRO-SLEEP !!!")
            if features.yawn_detected:
                lines.append(">>> YAWN <<<")

        for i, text in enumerate(lines):
            y = y_offset + i * 22
            # Background rectangle for readability
            (tw, th), _ = cv2.getTextSize(text, font, scale, thickness)
            cv2.rectangle(frame, (x - 2, y - th - 4),
                          (x + tw + 2, y + 4), bg_color, -1)
            cv2.putText(frame, text, (x, y), font, scale, color,
                        thickness, cv2.LINE_AA)
