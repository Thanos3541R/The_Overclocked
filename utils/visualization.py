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
from typing import Dict, List, Optional, Tuple
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
    SOLID_PART_KEYS,
    SOLID_REGIONS,
    draw_solid_face_regions,
)


class Visualizer:
    """Debug overlay renderer for the DMS pipeline with vectorized batch drawing."""

    OVERLAY_MODES = ("off", "regions", "full", "solid")

    def __init__(
        self,
        overlay_mode: str = "regions",
        show_iris: bool = True,
        show_pose_axes: bool = True,
        show_mesh: bool = True,
        mesh_style: Optional[str] = None,
        solid_alpha: float = 0.42,
        enabled_solid_parts: Optional[List[str]] = None,
        show_roi_box: bool = False,
    ):
        """Initialize HUD visualizer.

        Args:
            overlay_mode: Initial overlay mode: 'off', 'regions' (default), 'full', or 'solid'.
            show_iris: If True, draws iris center crosshairs.
            show_pose_axes: If True, draws 3D head pose coordinate axes.
            show_mesh: Legacy parameter; if False, forces overlay_mode='off'.
            mesh_style: Legacy parameter; mapped to overlay_mode if provided.
            solid_alpha: Transparency alpha for solid color facial regions (default: 0.42).
            enabled_solid_parts: Optional list of part names to enable initially. If None, all parts enabled.
            show_roi_box: If True, renders AdaptiveROITracker bounding box and KinematicStabilizer shift vector.
        """
        if mesh_style is not None:
            if mesh_style in ("off", "none"):
                overlay_mode = "off"
            elif mesh_style in ("full", "tesselation"):
                overlay_mode = "full"
            elif mesh_style in ("solid", "color", "mask", "facemask"):
                overlay_mode = "solid"
            else:
                overlay_mode = "regions"
        elif not show_mesh:
            overlay_mode = "off"

        if overlay_mode not in self.OVERLAY_MODES:
            overlay_mode = "regions"

        self.overlay_mode = overlay_mode
        self.show_iris = show_iris
        self.show_pose_axes = show_pose_axes
        self.solid_alpha = float(solid_alpha)
        self.show_roi_box: bool = bool(show_roi_box)

        # Dictionary of individual solid facial part toggles
        if enabled_solid_parts is not None:
            self.solid_parts: Dict[str, bool] = {r.key: False for r in SOLID_REGIONS}
            for part in enabled_solid_parts:
                canonical = self._normalize_part_name(part)
                if canonical in self.solid_parts:
                    self.solid_parts[canonical] = True
        else:
            self.solid_parts: Dict[str, bool] = {r.key: True for r in SOLID_REGIONS}

        self._legend_item_boxes: List[Tuple[int, int, int, int, str]] = []
        self._mouse_clicked: bool = False

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
        elif style in ("solid", "color", "mask", "facemask"):
            self.overlay_mode = "solid"
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

    @staticmethod
    def _normalize_part_name(name: str) -> str:
        """Map user / key input aliases to canonical solid part keys."""
        k = str(name).strip().lower().replace(" ", "_").replace("-", "_")
        aliases = {
            "1": "forehead",
            "f": "forehead",
            "head": "forehead",
            "forehead": "forehead",
            "2": "cheeks",
            "c": "cheeks",
            "cheek": "cheeks",
            "cheeks": "cheeks",
            "3": "chin",
            "chin": "chin",
            "jaw": "chin",
            "4": "lips",
            "l": "lips",
            "lip": "lips",
            "mouth": "lips",
            "lips": "lips",
            "5": "eyes",
            "e": "eyes",
            "eye": "eyes",
            "eyes": "eyes",
            "6": "nose",
            "n": "nose",
            "nose": "nose",
            "7": "eyebrows",
            "b": "eyebrows",
            "brow": "eyebrows",
            "brows": "eyebrows",
            "eyebrow": "eyebrows",
            "eyebrows": "eyebrows",
            "8": "face",
            "oval": "face",
            "face_oval": "face",
            "facemask": "face",
            "face": "face",
        }
        return aliases.get(k, k)

    def toggle_solid_part(self, part_name: str) -> bool:
        """Toggle an individual facial part solid color overlay on/off.

        Args:
            part_name: Part key, name, number, or alias (e.g. 'forehead', '1', 'lips', 'l').

        Returns:
            New boolean status of the part.
        """
        canonical = self._normalize_part_name(part_name)
        if canonical not in self.solid_parts:
            return False
        self.solid_parts[canonical] = not self.solid_parts[canonical]
        return self.solid_parts[canonical]

    def set_solid_part(self, part_name: str, enabled: bool) -> None:
        """Explicitly set an individual facial part solid color state."""
        canonical = self._normalize_part_name(part_name)
        if canonical in self.solid_parts:
            self.solid_parts[canonical] = bool(enabled)

    def is_solid_part_enabled(self, part_name: str) -> bool:
        """Check if an individual facial part solid color is enabled."""
        canonical = self._normalize_part_name(part_name)
        return self.solid_parts.get(canonical, False)

    def toggle_all_solid_parts(self) -> bool:
        """Toggle all solid facial parts simultaneously.

        If any part is currently ON, turns all OFF.
        If all parts are currently OFF, turns all ON.

        Returns:
            True if any part is now enabled, False if all disabled.
        """
        any_enabled = any(self.solid_parts.values())
        new_state = not any_enabled
        for k in self.solid_parts:
            self.solid_parts[k] = new_state
        return new_state

    def cycle_overlay_mode(self) -> str:
        """Cycle overlay mode: off -> regions -> full -> solid -> off.

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

    def handle_mouse_click(self, event: int, x: int, y: int, flags: int = 0, param=None) -> bool:
        """Handle mouse click events on the HUD (e.g. clicking legend items).

        Args:
            event: OpenCV mouse event code (e.g. cv2.EVENT_LBUTTONDOWN).
            x: X pixel coordinate of mouse event.
            y: Y pixel coordinate of mouse event.
            flags: OpenCV event flags.
            param: Optional user data.

        Returns:
            True if event was handled and changed HUD state, False otherwise.
        """
        if event != cv2.EVENT_LBUTTONDOWN or self.overlay_mode != "solid":
            return False

        for x1, y1, x2, y2, key in getattr(self, "_legend_item_boxes", []):
            if x1 <= x <= x2 and y1 <= y <= y2:
                self._mouse_clicked = True
                if key == "__all__":
                    self.toggle_all_solid_parts()
                    return True
                else:
                    self.toggle_solid_part(key)
                    return True
        return False

    def handle_key(self, key: int) -> bool:
        """Handle keyboard shortcuts for HUD overlay and solid parts.

        Args:
            key: Key code from cv2.waitKey (0-255).

        Returns:
            True if key was an overlay shortcut and handled, False otherwise.
        """
        if key in (ord('v'), ord('V'), ord('m'), ord('M')):
            self.cycle_overlay_mode()
            return True
        elif key == ord('0'):
            self.toggle_all_solid_parts()
            return True
        elif key in (ord('1'), ord('f'), ord('F')):
            self.toggle_solid_part('forehead')
            return True
        elif key in (ord('2'), ord('c'), ord('C')):
            self.toggle_solid_part('cheeks')
            return True
        elif key == ord('3'):
            self.toggle_solid_part('chin')
            return True
        elif key in (ord('4'), ord('l'), ord('L')):
            self.toggle_solid_part('lips')
            return True
        elif key in (ord('5'), ord('e'), ord('E')):
            self.toggle_solid_part('eyes')
            return True
        elif key in (ord('6'), ord('n'), ord('N')):
            self.toggle_solid_part('nose')
            return True
        elif key in (ord('7'), ord('b'), ord('B')):
            self.toggle_solid_part('eyebrows')
            return True
        elif key == ord('8'):
            self.toggle_solid_part('face')
            return True
        elif key in (ord('r'), ord('R')):
            self.show_roi_box = not self.show_roi_box
            return True
        return False

    def draw(
        self,
        frame: np.ndarray,
        landmarks: Optional[np.ndarray],
        features=None,
        pnp_normalizer=None,
        pnp_success: Optional[bool] = None,
        assessment=None,
        bayesian_state=None,
        calibration_progress: Optional[float] = None,
        roi_bounds: Optional[Tuple[int, int, int, int]] = None,
        shift_vector: Optional[Tuple[float, float]] = None,
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
            assessment: Optional ImpairmentAssessment for status banner.
            bayesian_state: Optional BayesianState for ISO 26262 lockout status.
            calibration_progress: Optional baseline calibration progress (0.0 to 1.0).
            roi_bounds: Optional (x1, y1, x2, y2) bounding box of adaptive ROI crop.
            shift_vector: Optional (dx, dy) frame shift from kinematic stabilizer.

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

            # Overlay rendering by mode
            if self.overlay_mode == "solid":
                draw_solid_face_regions(frame, pts, self.solid_parts, dimmed=dimmed, alpha=self.solid_alpha)
            else:
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

        # ── Adaptive ROI & Kinematic Stabilizer Overlay (Hotkey [r]) ──
        if self.show_roi_box and roi_bounds is not None:
            self._draw_roi_and_shift(frame, roi_bounds, shift_vector)

        # ── Top-Center Diagnostic Status Banner ──
        if assessment is not None:
            self._draw_status_banner(frame, assessment, bayesian_state, calibration_progress)

        # ── Text HUD Panel (Top-Left) ──
        self._draw_text_panel(frame, features)

        return frame

    def _draw_legend(self, frame: np.ndarray, is_gated: bool) -> None:
        """Render color-coded region or solid parts legend in top-right corner."""
        h, w = frame.shape[:2]

        if self.overlay_mode == "solid":
            box_w = 205
            box_x = max(10, w - box_w - 12)
            box_y = 12
            n_items = len(SOLID_REGIONS)
            box_h = 28 + n_items * 18 + 22

            self._legend_item_boxes = []

            # Background box
            overlay = frame.copy()
            cv2.rectangle(overlay, (box_x, box_y), (box_x + box_w, box_y + box_h), (18, 18, 18), -1)
            cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)
            cv2.rectangle(frame, (box_x, box_y), (box_x + box_w, box_y + box_h), (80, 80, 80), 1)

            # Header title
            cv2.putText(frame, "LEGEND [SOLID PARTS]", (box_x + 8, box_y + 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1, cv2.LINE_AA)

            # Solid parts with hotkey badge, swatch, name, and ON/OFF indicator
            for idx, reg in enumerate(SOLID_REGIONS):
                item_y = box_y + 36 + idx * 18
                is_on = self.solid_parts.get(reg.key, True)
                swatch_color = (reg.dimmed_color if is_gated else reg.color) if is_on else (40, 40, 40)
                border_color = (180, 180, 180) if is_on else (70, 70, 70)

                # Store bounding box for mouse clicks
                self._legend_item_boxes.append((box_x, item_y - 12, box_x + box_w, item_y + 6, reg.key))

                # Hotkey badge [1]..[8]
                badge = f"[{reg.hotkey}]"
                cv2.putText(frame, badge, (box_x + 6, item_y - 1),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.34, (200, 200, 200), 1, cv2.LINE_AA)

                # Swatch
                cv2.rectangle(frame, (box_x + 28, item_y - 10), (box_x + 40, item_y + 1), swatch_color, -1)
                cv2.rectangle(frame, (box_x + 28, item_y - 10), (box_x + 40, item_y + 1), border_color, 1)

                # Label
                text_color = (220, 220, 220) if is_on else (110, 110, 110)
                cv2.putText(frame, reg.name, (box_x + 46, item_y - 1),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.36, text_color, 1, cv2.LINE_AA)

                # Status pill (ON / OFF)
                status_text = "ON" if is_on else "OFF"
                status_color = (0, 240, 100) if is_on else (120, 120, 120)
                cv2.putText(frame, status_text, (box_x + box_w - 32, item_y - 1),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.34, status_color, 1, cv2.LINE_AA)

            # Footer hint
            footer_y = box_y + 36 + n_items * 18 + 4
            self._legend_item_boxes.append((box_x, footer_y - 12, box_x + box_w, footer_y + 8, "__all__"))
            cv2.putText(frame, "[0] Toggle All | [1-8] Part", (box_x + 8, footer_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.32, (160, 160, 160), 1, cv2.LINE_AA)
            return

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
            rvec_to_use = getattr(normalizer, "aligned_rvec", None)
            if rvec_to_use is None:
                rvec_to_use = normalizer.rvec
            img_pts, _ = cv2.projectPoints(
                axes_3d, rvec_to_use, normalizer.tvec,
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

    def _draw_roi_and_shift(
        self,
        frame: np.ndarray,
        roi_bounds: Tuple[int, int, int, int],
        shift_vector: Optional[Tuple[float, float]] = None,
    ) -> None:
        """Draw AdaptiveROITracker bounding box and KinematicStabilizer shift vector."""
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = roi_bounds
        x1 = max(0, min(w - 1, int(x1)))
        y1 = max(0, min(h - 1, int(y1)))
        x2 = max(0, min(w - 1, int(x2)))
        y2 = max(0, min(h - 1, int(y2)))
        if x2 <= x1 or y2 <= y1:
            return

        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 1, cv2.LINE_AA)
        label = f"ROI [r] {x2 - x1}x{y2 - y1}"
        if shift_vector is not None:
            dx, dy = shift_vector
            label += f" | shift:({dx:+.1f},{dy:+.1f})px"
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            ex = int(np.clip(cx + dx * 4.0, 0, w - 1))
            ey = int(np.clip(cy + dy * 4.0, 0, h - 1))
            cv2.arrowedLine(frame, (cx, cy), (ex, ey), (0, 255, 255), 2, cv2.LINE_AA)

        cv2.putText(
            frame, label, (x1 + 4, max(16, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.40, (255, 255, 0), 1, cv2.LINE_AA
        )

    def _draw_status_banner(
        self,
        frame: np.ndarray,
        assessment,
        bayesian_state=None,
        calibration_progress: Optional[float] = None,
    ) -> None:
        """Draw top-center diagnostic classification and ISO 26262 Bayesian status banner."""
        h, w = frame.shape[:2]
        if w < 300 or h < 80:
            return

        banner_w = min(460, w - 20)
        banner_h = 48
        bx1 = max(10, (w - banner_w) // 2)
        by1 = 10
        bx2 = bx1 + banner_w
        by2 = by1 + banner_h

        state_str = getattr(assessment, "classification", "SOBER")
        bar_color = {
            "SOBER": (0, 180, 0),
            "MILD_IMPAIRMENT": (0, 165, 255),
            "HIGH_RISK_INTOXICATED": (0, 0, 220),
        }.get(state_str, (80, 80, 80))

        overlay = frame.copy()
        cv2.rectangle(overlay, (bx1, by1), (bx2, by2), (18, 18, 18), -1)
        cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)
        cv2.rectangle(frame, (bx1, by1), (bx2, by1 + 6), bar_color, -1)
        cv2.rectangle(frame, (bx1, by1), (bx2, by2), bar_color, 1)

        risk = getattr(assessment, "risk_score", 0.0)
        inv_risk = getattr(assessment, "involuntary_risk_score", 0.0)
        line1 = f"{state_str} | Risk:{risk:.1f}% (Invol:{inv_risk:.1f}%)"
        if getattr(assessment, "ml_model_active", False) and getattr(assessment, "ml_probability", None) is not None:
            line1 += f" | ML:{assessment.ml_probability:.0f}%"
        cv2.putText(
            frame, line1, (bx1 + 10, by1 + 23),
            cv2.FONT_HERSHEY_SIMPLEX, 0.44, (255, 255, 255), 1, cv2.LINE_AA
        )

        if bayesian_state is not None:
            post_p = getattr(bayesian_state, "posterior_probability", 0.0) * 100.0
            sust_s = getattr(bayesian_state, "sustained_high_confidence_s", 0.0)
            locked = getattr(bayesian_state, "lockout_confirmed", False)
            cal_pct = (calibration_progress if calibration_progress is not None else 1.0) * 100.0
            status_tag = "LOCKOUT ARMED" if locked else f"Calib:{cal_pct:.0f}%"
            line2 = f"Bayes P:{post_p:.1f}% ({sust_s:.1f}s) | {status_tag}"
            line2_color = (0, 100, 255) if locked else (200, 220, 200)
            cv2.putText(
                frame, line2, (bx1 + 10, by1 + 41),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, line2_color, 1, cv2.LINE_AA
            )

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
                f"Tremor: Lid {getattr(features, 'eyelid_tremor_hz', 0.0):.1f}Hz | Eye {getattr(features, 'eyeball_tremor_hz', 0.0):.1f}Hz | Brow:{getattr(features, 'brow_elevation_delta', 0.0)*100:+.0f}%",
                f"rPPG: {getattr(features, 'rppg_pulse_bpm', 0.0):.0f}bpm (HRV:{getattr(features, 'rppg_hrv_rmssd_ms', 0.0):.0f}ms) | Vib:{getattr(features, 'vibration_score', 0.0):.2f} | M_mask:{getattr(features, 'masking_divergence_score', 0.0):.1f}%",
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
            if getattr(features, "eyeglass_glare_detected", False):
                lines.append("[GUARD] EYEGLASS GLARE MONO-FALLBACK")
            if getattr(features, "sunglasses_occluded", False):
                lines.append("[GUARD] SUNGLASSES OCCLUDED")
            if getattr(features, "speech_active", False):
                lines.append("[GUARD] SPEECH / SINGING CADENCE")
            if getattr(features, "highway_fixation_exempt", False):
                lines.append("[GUARD] HIGHWAY FIXATION EXEMPT")

        for i, text in enumerate(lines):
            y = y_offset + i * 22
            (tw, th), _ = cv2.getTextSize(text, font, scale, thickness)
            cv2.rectangle(frame, (x - 2, y - th - 4), (x + tw + 2, y + 4), bg_color, -1)
            cv2.putText(frame, text, (x, y), font, scale, color, thickness, cv2.LINE_AA)
