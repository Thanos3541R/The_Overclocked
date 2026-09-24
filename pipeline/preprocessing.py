"""Multi-stage lighting normalization for driver monitoring.

Pipeline:
  1. CLAHE on luminance channel (YUV) for local contrast
  2. Adaptive gamma correction for dark frames (precomputed LUTs)

Notes:
  - No bilateral filter (costs 35-65ms/frame on 720p, kills real-time)
  - For NIR (grayscale) input, CLAHE applied directly to raw frame
  - Gamma < 1.0 brightens the image; gamma > 1.0 darkens
"""
import cv2
import numpy as np
from typing import Tuple


class LightingNormalizer:
    """Real-time lighting normalization with CLAHE and adaptive gamma."""

    def __init__(self, clip_limit: float = 2.0,
                 tile_grid_size: Tuple[int, int] = (8, 8),
                 low_light_threshold: float = 60.0,
                 gamma_dark: float = 0.5,
                 gamma_dim: float = 0.7):
        """
        Args:
            clip_limit: CLAHE contrast clip limit.
            tile_grid_size: CLAHE tile grid dimensions.
            low_light_threshold: Mean luminance below this triggers gamma lift.
            gamma_dark: Gamma for very dark frames (mean lum < 40). Must be < 1.0.
            gamma_dim: Gamma for dim frames (40 <= mean lum < threshold). Must be < 1.0.
        """
        self.clahe = cv2.createCLAHE(clipLimit=clip_limit,
                                     tileGridSize=tile_grid_size)
        self.low_light_threshold = low_light_threshold
        self.gamma_dark = gamma_dark
        self.gamma_dim = gamma_dim

        # Precompute gamma LUTs for fast lookup (gamma < 1 = brighten)
        self._lut_dark = self._build_gamma_lut(gamma_dark)
        self._lut_dim = self._build_gamma_lut(gamma_dim)

    @staticmethod
    def _build_gamma_lut(gamma: float) -> np.ndarray:
        """Build a 256-entry lookup table for gamma correction.

        Formula: output = (input / 255) ^ gamma * 255
        When gamma < 1.0, this brightens the image.
        When gamma > 1.0, this darkens the image.
        """
        table = np.array([
            np.clip(((i / 255.0) ** gamma) * 255.0, 0, 255)
            for i in range(256)
        ], dtype=np.uint8)
        return table

    def process(self, frame: np.ndarray, return_rgb: bool = False) -> np.ndarray:
        """Apply lighting normalization to a frame.

        Args:
            frame: Input frame (BGR color or single-channel grayscale).
            return_rgb: If True, returns normalized frame in RGB format
                        (optimized single-pass conversion for MediaPipe input).

        Returns:
            Normalized frame in the requested format (RGB if return_rgb=True,
            otherwise matching input color format).
        """
        if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0 or frame.ndim not in (2, 3):
            return frame

        # ── Grayscale / NIR path ──
        if len(frame.shape) == 2 or (len(frame.shape) == 3 and frame.shape[2] == 1):
            gray_input = frame if len(frame.shape) == 2 else frame[:, :, 0]
            if gray_input.dtype != np.uint8:
                if np.issubdtype(gray_input.dtype, np.floating) and np.nanmax(gray_input) <= 1.0:
                    gray_input = (gray_input * 255.0)
                gray_input = np.clip(gray_input, 0, 255).astype(np.uint8)
            enhanced = self.clahe.apply(gray_input)
            mean_lum = float(enhanced.mean())
            if mean_lum < self.low_light_threshold:
                lut = self._lut_dark if mean_lum < 40.0 else self._lut_dim
                enhanced = cv2.LUT(enhanced, lut)
            if return_rgb:
                return cv2.cvtColor(enhanced, cv2.COLOR_GRAY2RGB)
            return enhanced

        # ── BGR color path (single-pass CLAHE + Y-LUT + conversion) ──
        color_frame = frame
        if color_frame.dtype != np.uint8:
            if np.issubdtype(color_frame.dtype, np.floating) and np.nanmax(color_frame) <= 1.0:
                color_frame = (color_frame * 255.0)
            color_frame = np.clip(color_frame, 0, 255).astype(np.uint8)

        yuv = cv2.cvtColor(color_frame, cv2.COLOR_BGR2YUV)
        yuv[:, :, 0] = self.clahe.apply(yuv[:, :, 0])
        mean_lum = float(yuv[:, :, 0].mean())

        if mean_lum < self.low_light_threshold:
            lut = self._lut_dark if mean_lum < 40.0 else self._lut_dim
            yuv[:, :, 0] = cv2.LUT(yuv[:, :, 0], lut)

        if return_rgb:
            return cv2.cvtColor(yuv, cv2.COLOR_YUV2RGB)
        return cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)
