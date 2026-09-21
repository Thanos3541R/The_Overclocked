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

    def process(self, frame: np.ndarray) -> np.ndarray:
        """Apply lighting normalization to a frame.

        Args:
            frame: Input frame (BGR color or single-channel grayscale).

        Returns:
            Normalized frame in the same format as input.
        """
        # ── Grayscale / NIR path ──
        if len(frame.shape) == 2:
            enhanced = self.clahe.apply(frame)
            mean_lum = enhanced.mean()
            if mean_lum < self.low_light_threshold:
                lut = self._lut_dark if mean_lum < 40.0 else self._lut_dim
                enhanced = cv2.LUT(enhanced, lut)
            return enhanced

        # ── BGR color path ──
        yuv = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)
        yuv[:, :, 0] = self.clahe.apply(yuv[:, :, 0])
        mean_lum = yuv[:, :, 0].mean()
        enhanced = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)

        if mean_lum < self.low_light_threshold:
            lut = self._lut_dark if mean_lum < 40.0 else self._lut_dim
            enhanced = cv2.LUT(enhanced, lut)

        return enhanced
