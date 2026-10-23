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
from typing import Optional, Tuple


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

    @staticmethod
    def _compute_fast_mean(region: np.ndarray) -> float:
        """Compute mean luminance using 4x strided subsampling for frames >= 32x32."""
        if region is None or region.size == 0:
            return 0.0
        h, w = region.shape[:2]
        if h >= 32 and w >= 32:
            return float(region[::4, ::4].mean())
        return float(region.mean())

    @staticmethod
    def _get_eval_region(
        img_2d: np.ndarray,
        face_bbox: Optional[Tuple[int, int, int, int]],
    ) -> np.ndarray:
        """Extract evaluation region (face ROI) or return full frame if unavailable."""
        if face_bbox is None or img_2d is None or img_2d.size == 0:
            return img_2d
        h, w = img_2d.shape[:2]
        x1 = max(0, min(w, int(round(face_bbox[0]))))
        y1 = max(0, min(h, int(round(face_bbox[1]))))
        x2 = max(0, min(w, int(round(face_bbox[2]))))
        y2 = max(0, min(h, int(round(face_bbox[3]))))
        if (x2 - x1) >= 16 and (y2 - y1) >= 16:
            return img_2d[y1:y2, x1:x2]
        return img_2d

    def _balance_split_sunlight(
        self,
        img_2d: np.ndarray,
        face_bbox: Optional[Tuple[int, int, int, int]],
    ) -> np.ndarray:
        """Bilateral split-sunlight balancing when side-window sunlight saturates one half of face."""
        if face_bbox is None or img_2d is None or img_2d.size == 0:
            return img_2d
        h, w = img_2d.shape[:2]
        x1 = max(0, min(w, int(round(face_bbox[0]))))
        y1 = max(0, min(h, int(round(face_bbox[1]))))
        x2 = max(0, min(w, int(round(face_bbox[2]))))
        y2 = max(0, min(h, int(round(face_bbox[3]))))
        roi_w = x2 - x1
        roi_h = y2 - y1
        if roi_w < 16 or roi_h < 16:
            return img_2d

        roi = img_2d[y1:y2, x1:x2]
        lum_left = self._compute_fast_mean(roi[:, :roi_w // 2])
        lum_right = self._compute_fast_mean(roi[:, roi_w // 2:])

        if abs(lum_left - lum_right) > 45.0:
            x_mid = (x1 + x2) // 2
            w_trans = max(16, roi_w // 4)
            x_start = x_mid - w_trans / 2.0
            x_end = x_mid + w_trans / 2.0
            xs = np.arange(w, dtype=np.float32)

            if lum_left < lum_right:
                # Left side is darker -> ramp is 1.0 on left, transitioning to 0.0 on right
                ramp = np.clip((x_end - xs) / max(float(w_trans), 1.0), 0.0, 1.0)
            else:
                # Right side is darker -> ramp is 0.0 on left, transitioning to 1.0 on right
                ramp = np.clip((xs - x_start) / max(float(w_trans), 1.0), 0.0, 1.0)

            ramp = ramp.reshape(1, w)
            lifted = cv2.LUT(img_2d, self._lut_dim)
            balanced = np.clip(
                (1.0 - ramp) * img_2d.astype(np.float32) + ramp * lifted.astype(np.float32),
                0,
                255,
            ).astype(np.uint8)
            return balanced

        return img_2d

    def process(
        self,
        frame: np.ndarray,
        return_rgb: bool = False,
        face_bbox: Optional[Tuple[int, int, int, int]] = None,
    ) -> np.ndarray:
        """Apply lighting normalization to a frame.

        Args:
            frame: Input frame (BGR color or single-channel grayscale).
            return_rgb: If True, returns normalized frame in RGB format
                        (optimized single-pass conversion for MediaPipe input).
            face_bbox: Optional (x1, y1, x2, y2) face bounding box in frame pixel coords.
                       When provided, evaluates luminance within the face ROI to prevent
                       windshield backlighting distortion and balances split-sunlight.

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

            gray_input = self._balance_split_sunlight(gray_input, face_bbox)
            enhanced = self.clahe.apply(gray_input)
            mean_lum = self._compute_fast_mean(self._get_eval_region(enhanced, face_bbox))
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
        raw_y = yuv[:, :, 0]
        raw_y = self._balance_split_sunlight(raw_y, face_bbox)
        yuv[:, :, 0] = raw_y

        raw_lum = self._compute_fast_mean(self._get_eval_region(raw_y, face_bbox))
        clahe_y = self.clahe.apply(raw_y)
        if raw_lum >= 95.0:
            # Blend 35% CLAHE + 65% original Y in well-lit scenes to prevent eyeglass glare amplification
            yuv[:, :, 0] = cv2.addWeighted(clahe_y, 0.35, raw_y, 0.65, 0)
        else:
            yuv[:, :, 0] = clahe_y
        mean_lum = self._compute_fast_mean(self._get_eval_region(yuv[:, :, 0], face_bbox))

        if mean_lum < self.low_light_threshold:
            lut = self._lut_dark if mean_lum < 40.0 else self._lut_dim
            yuv[:, :, 0] = cv2.LUT(yuv[:, :, 0], lut)

        if return_rgb:
            return cv2.cvtColor(yuv, cv2.COLOR_YUV2RGB)
        return cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)

    def normalize(
        self,
        frame: np.ndarray,
        return_rgb: bool = False,
        face_bbox: Optional[Tuple[int, int, int, int]] = None,
    ) -> np.ndarray:
        """Alias for process()."""
        return self.process(frame, return_rgb=return_rgb, face_bbox=face_bbox)
