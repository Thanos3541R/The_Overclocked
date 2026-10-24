"""Zero-blur kinematic video & landmark stabilizer for automotive road shocks.

Operates in coordinate and ROI-window space rather than warping image pixels
(preserving 100% sub-pixel sharpness for iris and eyelid contour extraction).
"""
import logging
from collections import deque
from typing import Deque, Dict, Optional, Tuple
import cv2
import numpy as np

logger = logging.getLogger(__name__)


class KinematicStabilizer:
    """Tracks instantaneous frame-to-frame rigid motion and high-frequency chassis vibration."""

    def __init__(self, history_len: int = 15):
        self._prev_gray_small: Optional[np.ndarray] = None
        self._prev_roi_pos: Tuple[int, int] = (0, 0)
        self._dy_history: Deque[float] = deque(maxlen=history_len)
        self._dx_history: Deque[float] = deque(maxlen=history_len)
        self._hann_windows: Dict[Tuple[int, int], Optional[np.ndarray]] = {}
        self.last_dx: float = 0.0
        self.last_dy: float = 0.0
        self.vibration_score: float = 0.0
        self.is_bumpy: bool = False

    def reset(self) -> None:
        self._prev_gray_small = None
        self._prev_roi_pos = (0, 0)
        self._dy_history.clear()
        self._dx_history.clear()
        self.last_dx = 0.0
        self.last_dy = 0.0
        self.vibration_score = 0.0
        self.is_bumpy = False

    def get_high_freq_jitter(self) -> Tuple[float, float]:
        """Return high-pass component of camera motion:
        (dx - mean(dx_history[-5:]), dy - mean(dy_history[-5:])).
        """
        if not self._dx_history or not self._dy_history:
            return 0.0, 0.0

        recent_dx = list(self._dx_history)[-5:]
        recent_dy = list(self._dy_history)[-5:]

        mean_dx = float(np.mean(recent_dx))
        mean_dy = float(np.mean(recent_dy))

        jitter_x = float(self.last_dx - mean_dx)
        jitter_y = float(self.last_dy - mean_dy)
        return jitter_x, jitter_y

    def estimate_frame_shift(self, frame: np.ndarray,
                             face_center: Optional[Tuple[Optional[float], Optional[float]]] = None,
                             face_scale: Optional[float] = None) -> Tuple[float, float, bool]:
        """Estimate instantaneous full-frame pixel shift (dx, dy) from previous frame.

        Uses fast downscaled phase correlation with a Hann window around the driver ROI
        (or full frame if face is not tracking) to predict sudden road shocks and cabin vibration.

        Returns:
            (dx_px, dy_px, is_bumpy)
        """
        if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
            return 0.0, 0.0, False

        h, w = frame.shape[:2]
        if h < 32 or w < 32:
            return 0.0, 0.0, False

        # Check if valid face ROI is available
        use_face_roi = False
        rx1, ry1 = 0, 0
        roi_w, roi_h = w, h
        target_patch = frame

        if face_center is not None and face_scale is not None:
            fc_x, fc_y = face_center
            if (fc_x is not None and fc_y is not None and
                    np.isfinite(fc_x) and np.isfinite(fc_y) and np.isfinite(face_scale) and
                    face_scale >= 10.0):
                # Expanded square ROI around face (1.8x face scale)
                roi_size = max(48, min(min(w, h), int(round(float(face_scale) * 1.8))))
                rx1 = max(0, min(w - roi_size, int(round(float(fc_x) - roi_size / 2.0))))
                ry1 = max(0, min(h - roi_size, int(round(float(fc_y) - roi_size / 2.0))))
                rx2 = rx1 + roi_size
                ry2 = ry1 + roi_size
                target_patch = frame[ry1:ry2, rx1:rx2]
                roi_w = rx2 - rx1
                roi_h = ry2 - ry1
                use_face_roi = True

        if use_face_roi:
            # 2x downscaling on face ROI for sub-0.2ms phase correlation
            small_w = max(32, roi_w // 2)
            small_h = max(32, roi_h // 2)
        else:
            # 4x downscaling on full frame
            small_w = max(32, w // 4)
            small_h = max(32, h // 4)

        # Crop & resize first, then convert tiny patch to grayscale to minimize memory bandwidth
        small_resized = cv2.resize(target_patch, (small_w, small_h), interpolation=cv2.INTER_AREA)
        if small_resized.ndim == 3:
            small_gray = cv2.cvtColor(small_resized, cv2.COLOR_BGR2GRAY)
        else:
            small_gray = small_resized
        small = small_gray.astype(np.float32)

        if self._prev_gray_small is None or self._prev_gray_small.shape != small.shape:
            self._prev_gray_small = small
            self._prev_roi_pos = (rx1, ry1)
            return 0.0, 0.0, False

        # Retrieve or create Hann window to eliminate spectral leakage at borders
        shape_key = (small.shape[1], small.shape[0])  # (cols, rows)
        if shape_key not in self._hann_windows:
            try:
                self._hann_windows[shape_key] = cv2.createHanningWindow(shape_key, cv2.CV_32F)
            except Exception:
                self._hann_windows[shape_key] = None

        hann = self._hann_windows.get(shape_key)

        try:
            if hann is not None:
                (sx, sy), response = cv2.phaseCorrelate(self._prev_gray_small, small, hann)
            else:
                (sx, sy), response = cv2.phaseCorrelate(self._prev_gray_small, small)

            if response > 0.15:
                scale_x = float(roi_w) / float(small_w)
                scale_y = float(roi_h) / float(small_h)
                # Compensate for any shift in ROI crop position between frames
                dx = float(sx * scale_x) + float(rx1 - self._prev_roi_pos[0])
                dy = float(sy * scale_y) + float(ry1 - self._prev_roi_pos[1])
            else:
                dx, dy = 0.0, 0.0
        except Exception as exc:
            logger.debug("cv2.phaseCorrelate failed in estimate_frame_shift: %s", exc, exc_info=True)
            dx, dy = 0.0, 0.0

        self._prev_gray_small = small
        self._prev_roi_pos = (rx1, ry1)

        # Clamp extreme scene-cut outliers
        if abs(dx) > w * 0.25 or abs(dy) > h * 0.25:
            dx, dy = 0.0, 0.0

        self.last_dx = dx
        self.last_dy = dy
        self._dx_history.append(dx)
        self._dy_history.append(dy)

        # Compute high-frequency vibration score (3-15 Hz sign-reversal oscillatory energy)
        if len(self._dy_history) >= 6:
            dys = np.array(self._dy_history, dtype=np.float64)
            # Zero-crossing rate of frame-to-frame vertical velocity (high during road vibration, 0 during smooth nod)
            signs = np.sign(dys[np.abs(dys) > 0.6])
            if len(signs) >= 4:
                reversals = float(np.sum(signs[1:] * signs[:-1] < 0)) / float(len(signs) - 1)
                rms_dy = float(np.sqrt(np.mean(dys ** 2)))
                self.vibration_score = float(np.clip(reversals * (rms_dy / 3.0), 0.0, 1.0))
            else:
                self.vibration_score = 0.0
        else:
            self.vibration_score = 0.0

        # Flag active speed-bump impulse (> 12px single-frame vertical jolt) or sustained rough road
        self.is_bumpy = bool(abs(dy) > 12.0 or self.vibration_score > 0.45)
        return dx, dy, self.is_bumpy
