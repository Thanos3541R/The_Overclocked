"""Debug visualization overlays for the driver monitoring pipeline.

Draws real-time annotations on the camera feed:
  - Color-coded facial landmark overlay (Key Regions or Full Mesh)
  - Iris gaze crosshairs
  - Head pose axis arrows (RGB = XYZ)
  - Live HUD text panel with FPS, overlay latency, and driver state metrics
  - Small region color legend
  - Gating status border (green = valid, red = gated/frozen)
"""
import time
from typing import Optional, Tuple
import cv2
import numpy as np

from utils.landmarks import (
    LEFT_IRIS_CENTER,
    RIGHT_IRIS_CENTER,
)
from utils.overlay_regions import (
    ALL_REGIONS,
    FULL_MESH_REGION,
    KEY_REGIONS,
    LEFT_IRIS_CORNERS_REGION,
    RIGHT_IRIS_CORNERS_REGION,
)


class Visualizer:
    """Debug overlay renderer for the DMS pipeline with vectorized batch drawing."""

    OVERLAY_MODES = ("off", "regions", "full")

    def __init__(
        self,
        overlay_mode: str = "regions",
        show_iris: bool = True,
        show_pose_axes: bool = True,
        show_mesh: bool = True,
        mesh_style: Optional[str] = None,
    ):
        """Initialize HUD visualizer.

        Args:
            overlay_mode: Initial overlay mode: 'off', 'regions' (default), or 'full'.
            show_iris: If True, draws iris center crosshairs.
            show_pose_axes: If True, draws 3D head pose coordinate axes.
            show_mesh: Legacy parameter; if False, forces overlay_mode='off'.
            mesh_style: Legacy parameter; mapped to overlay_mode if provided.
        """
        if mesh_style is not None:
            if mesh_style in ("off", "none"):
                overlay_mode = "off"
            elif mesh_style in ("full", "tesselation"):
                overlay_mode = "full"
            else:
                overlay_mode = "regions"
        elif not show_mesh:
            overlay_mode = "off"

        if overlay_mode not in self.OVERLAY_MODES:
            overlay_mode = "regions"

        self.overlay_mode = overlay_mode
        self.show_iris = show_iris
        self.show_pose_axes = show_pose_axes

        self._prev_time = time.time()
        self._fps = 0.0
        self._fps_alpha = 0.1  # EMA smoothing for FPS display

        self._last_overlay_ms = 0.0
        self._overlay_ms_ema = 0.0
        self._overlay_alpha = 0.15

    @property
    def show_mesh(self) -> bool:
        """Compatibility property: True if overlay is active."""
        return self.overlay_mode != "off"

    @show_mesh.setter
    def show_mesh(self, val: bool) -> None:
        if not val:
            self.overlay_mode = "off"
        elif self.overlay_mode == "off":
            self.overlay_mode = "regions"

    @property
    def mesh_style(self) -> str:
        """Compatibility property: returns current overlay mode."""
        return self.overlay_mode

    @mesh_style.setter
    def mesh_style(self, style: str) -> None:
        if style in self.OVERLAY_MODES:
            self.overlay_mode = style
        elif style in ("contours", "dots"):
            self.overlay_mode = "regions"
        else:
            self.overlay_mode = "off"

    @property
    def last_overlay_ms(self) -> float:
        """Most recent landmark overlay rendering duration in milliseconds."""
        return self._last_overlay_ms

    @property
    def overlay_ms(self) -> float:
        """Smoothed landmark overlay rendering duration in milliseconds."""
        return self._overlay_ms_ema

    def cycle_overlay_mode(self) -> str:
        """Cycle overlay mode: off -> regions -> full -> off.

        Returns:
            New overlay mode string.
        """
        modes = list(self.OVERLAY_MODES)
        idx = (modes.index(self.overlay_mode) + 1) % len(modes) if self.overlay_mode in modes else 1
        self.overlay_mode = modes[idx]
        return self.overlay_mode

    def cycle_mesh_mode(self) -> str:
        """Alias for cycle_overlay_mode for backwards compatibility."""
        return self.cycle_overlay_mode()

    def draw(
        self,
        frame: np.ndarray,
        landmarks: Optional[np.ndarray],
        features=None,
        pnp_normalizer=None,
        pnp_success: Optional[bool] = None,
    ) -> np.ndarray:
        """Draw all debug overlays on the frame.

        Renders color-coded regions on a pre-copied display frame with zero
        per-landmark Python loops using batch cv2.polylines calls.

        Args:
            frame: Display frame to draw on (modified in-place).
            landmarks: Smoothed pixel-space landmarks (478, 3), or None.
            features: FrameFeatures dataclass, or None.
            pnp_normalizer: PnPNormalizer instance for pose axes, or None.
            pnp_success: Optional bool flag indicating whether PnP solved successfully this frame.

        Returns:
            Annotated display frame.
        """
        # Update FPS
        now = time.time()
        dt = now - self._prev_time
        if dt > 0:
            instant_fps = 1.0 / dt
            self._fps = self._fps * (1 - self._fps_alpha) + instant_fps * self._fps_alpha
        self._prev_time = now

        # Determine gating state
        is_gated = False
        if features is not None:
            is_gated = (not features.face_valid) or bool(features.imu_gated)

        # ── Color-Coded Landmark Overlay (Vectorized Batch) ──
        t0 = time.perf_counter()
        if (landmarks is not None and getattr(landmarks, "ndim", 0) >= 2 and
                landmarks.shape[1] >= 2 and len(landmarks) > 0 and self.overlay_mode != "off"):
            # Convert smoothed landmarks to int pixel coordinates ONCE per frame
            pts = np.rint(landmarks[:, :2]).astype(np.int32)
            dimmed = is_gated

            # Full mesh wireframe if active
            if self.overlay_mode == "full":
                FULL_MESH_REGION.draw(frame, pts, dimmed=dimmed)

            # Key regions (batch cv2.polylines per region)
            for region in KEY_REGIONS:
                region.draw(frame, pts, dimmed=dimmed)

            # Iris gaze crosshair markers
            if self.show_iris and len(pts) > max(LEFT_IRIS_CENTER, RIGHT_IRIS_CENTER):
                c_r = RIGHT_IRIS_CORNERS_REGION.dimmed_color if dimmed else RIGHT_IRIS_CORNERS_REGION.color
                c_l = LEFT_IRIS_CORNERS_REGION.dimmed_color if dimmed else LEFT_IRIS_CORNERS_REGION.color
                cv2.drawMarker(frame, tuple(pts[LEFT_IRIS_CENTER]), c_r,
                               cv2.MARKER_CROSS, 8, 1, line_type=cv2.LINE_AA)
                cv2.drawMarker(frame, tuple(pts[RIGHT_IRIS_CENTER]), c_l,
                               cv2.MARKER_CROSS, 8, 1, line_type=cv2.LINE_AA)

            t1 = time.perf_counter()
            self._last_overlay_ms = (t1 - t0) * 1000.0
            self._overlay_ms_ema = (
                self._overlay_ms_ema * (1 - self._overlay_alpha) + self._last_overlay_ms * self._overlay_alpha
            )
        else:
            self._last_overlay_ms = 0.0
            self._overlay_ms_ema = 0.0

        # ── Head Pose Axes (Only drawn if pose is valid and current) ──
        if self.show_pose_axes and pnp_normalizer is not None and pnp_normalizer.rvec is not None:
            pose_ok = True
            if pnp_success is not None:
                pose_ok = bool(pnp_success)
            elif features is not None and getattr(features, "gating_reason", "") == "no_pose":
                pose_ok = False

            if pose_ok:
                self._draw_pose_axes(frame, pnp_normalizer, is_gated=is_gated)

        # ── Gating Border: Red on gated frames, Green on valid ──
        if features is not None:
            border_color = (0, 0, 255) if is_gated else (0, 255, 0)
            h, w = frame.shape[:2]
            cv2.rectangle(frame, (0, 0), (w - 1, h - 1), border_color, 3)

        # ── Region Legend (Top-Right) ──
        if self.overlay_mode != "off":
            self._draw_legend(frame, is_gated=is_gated)

        # ── Text HUD Panel (Top-Left) ──
        self._draw_text_panel(frame, features)

        return frame

    def _draw_legend(self, frame: np.ndarray, is_gated: bool) -> None:
        """Render small color-coded region legend in top-right corner."""
        h, w = frame.shape[:2]
        box_w = 175
        box_x = max(10, w - box_w - 12)
        box_y = 12

        items = list(KEY_REGIONS)
        if self.overlay_mode == "full":
            items.append(FULL_MESH_REGION)

        n_items = len(items)
        box_h = 28 + n_items * 18 + 8

        # Background box
        overlay = frame.copy()
        cv2.rectangle(overlay, (box_x, box_y), (box_x + box_w, box_y + box_h), (18, 18, 18), -1)
        cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)
        cv2.rectangle(frame, (box_x, box_y), (box_x + box_w, box_y + box_h), (80, 80, 80), 1)

        # Header title
        mode_str = "REGIONS" if self.overlay_mode == "regions" else "FULL MESH"
        header = f"LEGEND [{mode_str}]"
        cv2.putText(frame, header, (box_x + 8, box_y + 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, (255, 255, 255), 1, cv2.LINE_AA)

        # Region items with color swatches
        for idx, reg in enumerate(items):
            item_y = box_y + 36 + idx * 18
            c = reg.dimmed_color if is_gated else reg.color
            # Swatch
            cv2.rectangle(frame, (box_x + 8, item_y - 10), (box_x + 20, item_y + 1), c, -1)
            cv2.rectangle(frame, (box_x + 8, item_y - 10), (box_x + 20, item_y + 1), (180, 180, 180), 1)
            # Label
            cv2.putText(frame, reg.name, (box_x + 26, item_y - 1),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (220, 220, 220), 1, cv2.LINE_AA)

    @staticmethod
    def _draw_dashed_line(
        img: np.ndarray,
        pt1: Tuple[int, int],
        pt2: Tuple[int, int],
        color: Tuple[int, int, int],
        thickness: int = 1,
        dash_len: int = 6,
    ) -> None:
        """Draw a dashed anti-aliased line between pt1 and pt2."""
        dist = np.hypot(pt2[0] - pt1[0], pt2[1] - pt1[1])
        if dist < 1.0:
            return
        n_dashes = max(1, int(dist / dash_len))
        for j in range(0, n_dashes, 2):
            s_frac = j / n_dashes
            e_frac = min(j + 1, n_dashes) / n_dashes
            start = (
                int(pt1[0] + (pt2[0] - pt1[0]) * s_frac),
                int(pt1[1] + (pt2[1] - pt1[1]) * s_frac),
            )
            end = (
                int(pt1[0] + (pt2[0] - pt1[0]) * e_frac),
                int(pt1[1] + (pt2[1] - pt1[1]) * e_frac),
            )
            cv2.line(img, start, end, color, thickness, cv2.LINE_AA)

    def _draw_pose_axes(
        self,
        frame: np.ndarray,
        normalizer,
        is_gated: bool = False,
    ) -> None:
        """Draw RGB head pose axes at the nose tip.

        Args:
            frame: Display frame.
            normalizer: PnPNormalizer with valid rvec/tvec.
            is_gated: If True, renders axes dimmed with dashed indicator.
        """
        if normalizer.rvec is None or normalizer.tvec is None:
            return

        axis_length = 100.0
        axes_3d = np.float64([
            [axis_length, 0, 0],   # X (Red)
            [0, axis_length, 0],   # Y (Green)
            [0, 0, -axis_length],  # Z (Blue)
            [0, 0, 0],             # Origin
        ])

        try:
            img_pts, _ = cv2.projectPoints(
                axes_3d, normalizer.rvec, normalizer.tvec,
                normalizer.camera_matrix, normalizer.dist_coeffs
            )
        except cv2.error:
            return

        origin = tuple(img_pts[3].ravel().astype(int))
        colors_solid = [(0, 0, 255), (0, 255, 0), (255, 0, 0)]  # BGR for XYZ
        colors_dimmed = [(0, 0, 110), (0, 110, 0), (110, 0, 0)]

        for i in range(3):
            end = tuple(img_pts[i].ravel().astype(int))
            if is_gated:
                self._draw_dashed_line(frame, origin, end, colors_dimmed[i], thickness=1, dash_len=6)
                cv2.circle(frame, end, 2, colors_dimmed[i], -1, cv2.LINE_AA)
            else:
                cv2.arrowedLine(frame, origin, end, colors_solid[i], 2, cv2.LINE_AA)

    def _draw_text_panel(self, frame: np.ndarray, features) -> None:
        """Draw feature values and overlay timing as HUD text panel."""
        h, w = frame.shape[:2]
        y_offset = 25
        x = 10
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.52
        color = (255, 255, 255)
        bg_color = (0, 0, 0)
        thickness = 1

        mode_label = self.overlay_mode.upper()
        timing_str = f"{self._last_overlay_ms:.2f}ms" if self.overlay_mode != "off" else "0.0ms"
        lines = [
            f"FPS: {self._fps:.1f} | Overlay [v]: {mode_label} ({timing_str})"
        ]

        if features is not None:
            lines.extend([
                f"EAR L:{features.ear_left:.3f} R:{features.ear_right:.3f} Avg:{features.ear_avg:.3f}",
                f"MAR: {features.mar:.3f}",
                f"PERCLOS: {features.perclos:.3f}",
                f"Yaw:{features.head_yaw:.1f} Pitch:{features.head_pitch:.1f} Roll:{features.head_roll:.1f}",
                f"Saccade: {features.saccade_velocity:.1f} deg/s [dx:{getattr(features, 'raw_iris_dx', 0.0):.4f}] (lat: {features.saccade_latency_ms:.0f}ms)",
                f"Blink Vel: dn {features.blink_closing_velocity:.2f} | up {features.blink_opening_velocity:.2f}",
                f"Gaze Yaw Disp: {features.gaze_yaw_dispersion:.3f}",
                f"Valid: {features.face_valid}" + (f" ({features.gating_reason})" if features.gating_reason else ""),
            ])

            if getattr(features, "imu_gated", False):
                lines.append(">>> IMU GATED / SHOCK <<<")
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
            (tw, th), _ = cv2.getTextSize(text, font, scale, thickness)
            cv2.rectangle(frame, (x - 2, y - th - 4), (x + tw + 2, y + 4), bg_color, -1)
            cv2.putText(frame, text, (x, y), font, scale, color, thickness, cv2.LINE_AA)
