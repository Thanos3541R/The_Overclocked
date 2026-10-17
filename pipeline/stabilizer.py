"""Zero-blur kinematic video & landmark stabilizer for automotive road shocks.

Operates in coordinate and ROI-window space rather than warping image pixels
(preserving 100% sub-pixel sharpness for iris and eyelid contour extraction).
"""
import logging
from collections import deque
from typing import Deque, Optional, Tuple
import cv2
import numpy as np

logger = logging.getLogger(__name__)


class KinematicStabilizer:
    """Tracks instantaneous frame-to-frame rigid motion and high-frequency chassis vibration."""

    def __init__(self, history_len: int = 15):
        self._prev_gray_small: Optional[np.ndarray] = None
        self._dy_history: Deque[float] = deque(maxlen=history_len)
        self._dx_history: Deque[float] = deque(maxlen=history_len)
        self.last_dx: float = 0.0
        self.last_dy: float = 0.0
        self.vibration_score: float = 0.0
        self.is_bumpy: bool = False

    def reset(self) -> None:
        self._prev_gray_small = None
        self._dy_history.clear()
        self._dx_history.clear()
        self.last_dx = 0.0
        self.last_dy = 0.0
        self.vibration_score = 0.0
        self.is_bumpy = False

    def estimate_frame_shift(self, frame: np.ndarray,
                             face_center: Optional[Tuple[float, float]] = None,
                             face_scale: Optional[float] = None) -> Tuple[float, float, bool]:
        """Estimate instantaneous full-frame pixel shift (dx, dy) from previous frame.

        Uses fast downscaled phase correlation around the driver region (or full frame)
        to predict sudden speed-bump jumps before MediaPipe ROI cropping.

        Returns:
            (dx_px, dy_px, is_bumpy)
        """
        if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
            return 0.0, 0.0, False

        h, w = frame.shape[:2]
        if h < 32 or w < 32:
            return 0.0, 0.0, False

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
        # Downscale by 4x for sub-0.3ms phase correlation
        small_w, small_h = max(32, w // 4), max(32, h // 4)
        small = cv2.resize(gray, (small_w, small_h), interpolation=cv2.INTER_AREA).astype(np.float32)

        if self._prev_gray_small is None or self._prev_gray_small.shape != small.shape:
            self._prev_gray_small = small
            return 0.0, 0.0, False

        try:
            (sx, sy), response = cv2.phaseCorrelate(self._prev_gray_small, small)
            dx = float(sx * (w / float(small_w))) if response > 0.15 else 0.0
            dy = float(sy * (h / float(small_h))) if response > 0.15 else 0.0
        except Exception as exc:
            logger.debug("cv2.phaseCorrelate failed in estimate_frame_shift: %s", exc, exc_info=True)
            dx, dy = 0.0, 0.0

        self._prev_gray_small = small

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
