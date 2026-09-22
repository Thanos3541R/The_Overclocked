"""MediaPipe Face Mesh wrapper for driver monitoring.

Detects 478 facial landmarks (468 face mesh + 10 iris).
Supports both:
  1. Modern MediaPipe Tasks API (`vision.FaceLandmarker` with `face_landmarker.task`)
  2. Legacy `mp.solutions.face_mesh` (for older installations)

Critical implementation details:
  - MediaPipe expects RGB input. OpenCV reads BGR. This wrapper handles conversion internally.
  - AdaptiveROITracker: Maintains smoothed face ROI via EMA (alpha=0.35, margin=2.0)
    for an ~8x receptive-field resolution boost on distant faces.
  - Multi-Scale Pyramid Acquisition: Falls back to central driver quadrant crop
    ([0.2*W, 0.1*H] to [0.8*W, 0.85*H]) on frame 0 or when lost.
"""
import os
import urllib.request
import cv2
import numpy as np
import mediapipe as mp
from typing import Optional, Tuple, Union

MODEL_URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task"


class AdaptiveROITracker:
    """Adaptive Region-of-Interest (ROI) tracker for MediaPipe Face Landmarker.

    Maintains an Exponential Moving Average (EMA) of the face bounding box center (cx, cy)
    and scale S across frames. Crops an expanded high-resolution ROI (margin=2.0) directly
    from the raw frame, enabling distant faces to fill 60-80% of MediaPipe's receptive field (~8x resolution boost).
    """

    def __init__(self, alpha: float = 0.35, margin: float = 2.0):
        """
        Args:
            alpha: EMA smoothing coefficient (0.0 < alpha <= 1.0).
            margin: Multiplier for bounding box scale to determine crop ROI dimension (>= 1.0).
        """
        if not (0.0 < alpha <= 1.0):
            raise ValueError(f"alpha must be in (0.0, 1.0], got {alpha}")
        if margin < 1.0:
            raise ValueError(f"margin must be >= 1.0, got {margin}")

        self.alpha = float(alpha)
        self.margin = float(margin)
        self.cx: Optional[float] = None
        self.cy: Optional[float] = None
        self.scale: Optional[float] = None
        self.is_tracking: bool = False

    def reset(self) -> None:
        """Reset tracking state."""
        self.cx = None
        self.cy = None
        self.scale = None
        self.is_tracking = False

    @staticmethod
    def compute_bbox(landmarks: np.ndarray) -> Tuple[float, float, float, float]:
        """Compute 2D bounding box (x_min, y_min, x_max, y_max) from landmarks.

        Args:
            landmarks: Array of shape (N, >=2) in pixel coordinates.

        Returns:
            (x_min, y_min, x_max, y_max)
        """
        if landmarks is None:
            return 0.0, 0.0, 0.0, 0.0
        try:
            lms = np.asarray(landmarks)
        except Exception:
            return 0.0, 0.0, 0.0, 0.0

        if (lms.size == 0 or lms.ndim < 1 or
                not np.issubdtype(lms.dtype, np.number) or
                np.isnan(lms).any() or np.isinf(lms).any()):
            return 0.0, 0.0, 0.0, 0.0

        lms = np.atleast_2d(lms)
        if lms.shape[1] < 2:
            return 0.0, 0.0, 0.0, 0.0

        x_min = float(np.min(lms[:, 0]))
        x_max = float(np.max(lms[:, 0]))
        y_min = float(np.min(lms[:, 1]))
        y_max = float(np.max(lms[:, 1]))
        return x_min, y_min, x_max, y_max

    @classmethod
    def compute_center_scale(cls, landmarks: np.ndarray) -> Tuple[float, float, float]:
        """Compute bounding box center (cx, cy) and scale S = max(width, height).

        Args:
            landmarks: Array of shape (N, >=2) in pixel coordinates.

        Returns:
            (cx, cy, scale)
        """
        x_min, y_min, x_max, y_max = cls.compute_bbox(landmarks)
        cx = (x_min + x_max) / 2.0
        cy = (y_min + y_max) / 2.0
        scale = max(x_max - x_min, y_max - y_min)
        return cx, cy, max(scale, 1.0)

    def update(self, landmarks: Optional[np.ndarray]) -> None:
        """Update EMA smoothed center (cx, cy) and scale from global landmarks.

        Args:
            landmarks: Array of shape (N, >=2) in global pixel coordinates.
        """
        if landmarks is None:
            self.reset()
            return

        try:
            landmarks = np.asarray(landmarks)
        except Exception:
            self.reset()
            return

        if (landmarks.size == 0 or
                landmarks.ndim < 1 or
                not np.issubdtype(landmarks.dtype, np.number) or
                np.isnan(landmarks).any() or np.isinf(landmarks).any()):
            self.reset()
            return

        lms = np.atleast_2d(landmarks)
        if lms.shape[1] < 2:
            self.reset()
            return

        raw_cx, raw_cy, raw_scale = self.compute_center_scale(lms)
        if np.isnan(raw_cx) or np.isnan(raw_cy) or np.isnan(raw_scale):
            self.reset()
            return

        if not self.is_tracking or self.cx is None or self.cy is None or self.scale is None:
            self.cx = raw_cx
            self.cy = raw_cy
            self.scale = raw_scale
            self.is_tracking = True
        else:
            self.cx = self.alpha * raw_cx + (1.0 - self.alpha) * self.cx
            self.cy = self.alpha * raw_cy + (1.0 - self.alpha) * self.cy
            self.scale = self.alpha * raw_scale + (1.0 - self.alpha) * self.scale

    def get_crop_bounds(self, frame_shape: Tuple[int, ...]) -> Tuple[int, int, int, int]:
        """Compute pixel bounds (x_crop_min, y_crop_min, x_crop_max, y_crop_max) for the ROI.

        Maintains the expanded crop dimension (scale * margin) with an exact square aspect ratio
        while shifting within frame boundaries, or clamped to the minimum frame dimension.

        Args:
            frame_shape: (height, width, ...) of raw frame.

        Returns:
            (x_crop_min, y_crop_min, x_crop_max, y_crop_max)
        """
        if len(frame_shape) < 2:
            return 0, 0, 0, 0

        h_frame, w_frame = frame_shape[:2]
        if w_frame <= 0 or h_frame <= 0:
            return 0, 0, 0, 0

        if not self.is_tracking or self.cx is None or self.cy is None or self.scale is None:
            return 0, 0, w_frame, h_frame

        max_size = min(w_frame, h_frame)
        crop_size = max(1, min(int(round(float(self.scale * self.margin))), max_size))

        # Center the square crop around (self.cx, self.cy)
        x1 = int(round(self.cx - crop_size / 2.0))
        y1 = int(round(self.cy - crop_size / 2.0))

        # Shift / clamp within frame boundaries while preserving exact square dimension
        if x1 < 0:
            x1 = 0
        elif x1 + crop_size > w_frame:
            x1 = max(0, w_frame - crop_size)

        if y1 < 0:
            y1 = 0
        elif y1 + crop_size > h_frame:
            y1 = max(0, h_frame - crop_size)

        x2 = x1 + crop_size
        y2 = y1 + crop_size

        return x1, y1, x2, y2

    def crop_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
        """Extract ROI crop from frame.

        Args:
            frame: Input image array.

        Returns:
            Tuple of (cropped_image, (x_crop_min, y_crop_min, x_crop_max, y_crop_max)).
        """
        bounds = self.get_crop_bounds(frame.shape)
        x1, y1, x2, y2 = bounds
        crop = frame[y1:y2, x1:x2]
        return crop, bounds

    @staticmethod
    def get_driver_quadrant_bounds(frame_shape: Tuple[int, ...]) -> Tuple[int, int, int, int]:
        """Compute bounding box for central driver quadrant crop:
        [0.2*W, 0.1*H] to [0.8*W, 0.85*H].

        Args:
            frame_shape: (height, width, ...) of raw frame.

        Returns:
            (x_crop_min, y_crop_min, x_crop_max, y_crop_max)
        """
        if len(frame_shape) < 2:
            return 0, 0, 0, 0

        h_frame, w_frame = frame_shape[:2]
        if w_frame <= 0 or h_frame <= 0:
            return 0, 0, 0, 0

        x_min = max(0, min(w_frame - 1, int(round(0.2 * w_frame))))
        y_min = max(0, min(h_frame - 1, int(round(0.1 * h_frame))))
        x_max = max(x_min + 1, min(w_frame, int(round(0.8 * w_frame))))
        y_max = max(y_min + 1, min(h_frame, int(round(0.85 * h_frame))))
        return x_min, y_min, x_max, y_max

    @staticmethod
    def remap_to_global(landmarks_norm: np.ndarray,
                        crop_box: Tuple[int, int, int, int]) -> Optional[np.ndarray]:
        """Remap crop-relative normalized coordinates back to global full-frame pixel coordinates
        with exact subpixel precision:
            x_global = x_crop_min + x_norm * W_crop
            y_global = y_crop_min + y_norm * H_crop
            z_global = z_norm * W_crop

        Args:
            landmarks_norm: Normalized landmarks array of shape (N, >=2) or (>=2,).
            crop_box: (x_crop_min, y_crop_min, x_crop_max, y_crop_max).

        Returns:
            Global coordinates array matching input dimensions.
        """
        if landmarks_norm is None:
            return None
        if len(landmarks_norm) == 0:
            return np.empty_like(landmarks_norm, dtype=np.float64)

        x_crop_min, y_crop_min, x_crop_max, y_crop_max = crop_box
        w_crop = max(float(x_crop_max - x_crop_min), 1.0)
        h_crop = max(float(y_crop_max - y_crop_min), 1.0)

        is_1d = (landmarks_norm.ndim == 1)
        lms = np.atleast_2d(landmarks_norm)

        landmarks_global = np.array(lms, dtype=np.float64, copy=True)
        landmarks_global[:, 0] = x_crop_min + lms[:, 0] * w_crop
        landmarks_global[:, 1] = y_crop_min + lms[:, 1] * h_crop
        if lms.shape[1] >= 3:
            landmarks_global[:, 2] = lms[:, 2] * w_crop

        return landmarks_global[0] if is_1d else landmarks_global

    @staticmethod
    def global_to_normalized(landmarks_global: np.ndarray,
                             crop_box: Tuple[int, int, int, int]) -> Optional[np.ndarray]:
        """Inverse of remap_to_global: converts global pixel coordinates to crop-normalized coordinates.

        Args:
            landmarks_global: Global landmarks array of shape (N, >=2) or (>=2,).
            crop_box: (x_crop_min, y_crop_min, x_crop_max, y_crop_max).

        Returns:
            Normalized landmarks array matching input dimensions.
        """
        if landmarks_global is None:
            return None
        if len(landmarks_global) == 0:
            return np.empty_like(landmarks_global, dtype=np.float64)

        x_crop_min, y_crop_min, x_crop_max, y_crop_max = crop_box
        w_crop = max(float(x_crop_max - x_crop_min), 1.0)
        h_crop = max(float(y_crop_max - y_crop_min), 1.0)

        is_1d = (landmarks_global.ndim == 1)
        lms = np.atleast_2d(landmarks_global)

        landmarks_norm = np.array(lms, dtype=np.float64, copy=True)
        landmarks_norm[:, 0] = (lms[:, 0] - x_crop_min) / w_crop
        landmarks_norm[:, 1] = (lms[:, 1] - y_crop_min) / h_crop
        if lms.shape[1] >= 3:
            landmarks_norm[:, 2] = lms[:, 2] / w_crop

        return landmarks_norm[0] if is_1d else landmarks_norm


class FaceMeshDetector:
    """Face landmark detector configured for driver monitoring."""

    get_driver_quadrant_bounds = staticmethod(AdaptiveROITracker.get_driver_quadrant_bounds)

    def __init__(self, max_faces: int = 1,
                 refine_landmarks: bool = True,
                 min_detection_confidence: float = 0.35,
                 min_tracking_confidence: float = 0.35,
                 enable_adaptive_roi: bool = True,
                 roi_margin: float = 2.0,
                 roi_alpha: float = 0.35):
        """
        Args:
            max_faces: Maximum faces to detect (1 for driver-only).
            refine_landmarks: If True, enables iris landmarks (468-477).
            min_detection_confidence: Minimum face detection confidence (default: 0.35).
            min_tracking_confidence: Minimum landmark tracking confidence (default: 0.35).
            enable_adaptive_roi: Enable high-resolution adaptive ROI tracking.
            roi_margin: Margin multiplier for adaptive ROI crop.
            roi_alpha: EMA smoothing coefficient for adaptive ROI tracker (default: 0.35).
        """
        self._n_landmarks = 478 if refine_landmarks else 468
        self._mode = "unknown"
        self.enable_adaptive_roi = enable_adaptive_roi
        self.roi_margin = roi_margin
        self.roi_tracker = AdaptiveROITracker(alpha=roi_alpha, margin=roi_margin)

        # Check if legacy mp.solutions is present
        if hasattr(mp, "solutions") and hasattr(mp.solutions, "face_mesh"):
            self._mode = "legacy"
            self.face_mesh = mp.solutions.face_mesh.FaceMesh(
                static_image_mode=False,
                max_num_faces=max_faces,
                refine_landmarks=refine_landmarks,
                min_detection_confidence=min_detection_confidence,
                min_tracking_confidence=min_tracking_confidence,
            )
        else:
            # Modern Tasks API (vision.FaceLandmarker)
            self._mode = "tasks"
            from mediapipe.tasks import python
            from mediapipe.tasks.python import vision

            # Locate or auto-download face_landmarker.task
            model_dir = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models"
            )
            os.makedirs(model_dir, exist_ok=True)
            model_path = os.path.join(model_dir, "face_landmarker.task")

            if not os.path.exists(model_path) or os.path.getsize(model_path) == 0:
                print("[DMS] Downloading face_landmarker.task model bundle...")
                urllib.request.urlretrieve(MODEL_URL, model_path)
                print(f"[DMS] Model ready at {model_path}")

            base_options = python.BaseOptions(model_asset_path=model_path)
            options = vision.FaceLandmarkerOptions(
                base_options=base_options,
                output_face_blendshapes=False,
                output_facial_transformation_matrixes=False,
                num_faces=max_faces,
                min_face_detection_confidence=min_detection_confidence,
                min_face_presence_confidence=min_detection_confidence,
                min_tracking_confidence=min_tracking_confidence,
            )
            self.face_landmarker = vision.FaceLandmarker.create_from_options(options)

    def _detect_raw(self, image: np.ndarray) -> Optional[np.ndarray]:
        """Detect face on an arbitrary image or crop.

        Args:
            image: Image in BGR or Grayscale format.

        Returns:
            Normalized landmarks array of shape (N, 3) in [0, 1] relative to image,
            or None if no face detected.
        """
        if (image is None or not isinstance(image, np.ndarray) or
                image.size == 0 or
                image.ndim not in (2, 3) or
                image.shape[0] < 10 or image.shape[1] < 10):
            return None

        if image.ndim == 3 and image.shape[2] not in (1, 3, 4):
            return None

        # Convert float or uint16 to uint8 for MediaPipe
        if image.dtype != np.uint8:
            if np.issubdtype(image.dtype, np.floating):
                if np.isnan(image).any() or np.isinf(image).any():
                    return None
                if np.nanmax(image) <= 1.0:
                    image = (image * 255.0)
                image = np.clip(image, 0, 255).astype(np.uint8)
            elif image.dtype == np.uint16:
                image = (image >> 8).astype(np.uint8)
            else:
                image = np.clip(image, 0, 255).astype(np.uint8)

        # Prepare RGB image for MediaPipe
        if len(image.shape) == 2 or (len(image.shape) == 3 and image.shape[2] == 1):
            rgb_frame = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        else:
            rgb_frame = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self._mode == "legacy":
            results = self.face_mesh.process(rgb_frame)
            if not results.multi_face_landmarks:
                return None
            face = results.multi_face_landmarks[0]
            landmarks = np.array(
                [[lm.x, lm.y, lm.z] for lm in face.landmark[:self._n_landmarks]],
                dtype=np.float64
            )
            return landmarks
        else:
            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=np.ascontiguousarray(rgb_frame)
            )
            results = self.face_landmarker.detect(mp_image)
            if not results.face_landmarks:
                return None
            face = results.face_landmarks[0]
            landmarks = np.array(
                [[lm.x, lm.y, lm.z] for lm in face[:self._n_landmarks]],
                dtype=np.float64
            )
            return landmarks

    def detect(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """Detect face and extract landmarks with adaptive ROI tracking and multi-scale acquisition.

        Args:
            frame: Input frame in BGR (color) or grayscale format.

        Returns:
            Landmarks as numpy array of shape (478, 3) in pixel coordinates
            (x_px, y_px, z_relative), or None if no face detected.
        """
        if (frame is None or not isinstance(frame, np.ndarray) or
                frame.size == 0 or
                frame.ndim not in (2, 3) or
                frame.shape[0] < 10 or frame.shape[1] < 10):
            return None

        h, w = frame.shape[:2]

        # 1. High-resolution ROI tracking if active
        if self.enable_adaptive_roi and self.roi_tracker.is_tracking:
            crop, crop_box = self.roi_tracker.crop_frame(frame)
            norm_lms = self._detect_raw(crop)
            if norm_lms is not None:
                landmarks = self.roi_tracker.remap_to_global(norm_lms, crop_box)
                self.roi_tracker.update(landmarks)
                return landmarks
            else:
                # Crop detection failed on this frame: immediately reset and fallback to full-frame
                self.roi_tracker.reset()

        # 2. Full-frame detection
        full_box = (0, 0, w, h)
        norm_lms = self._detect_raw(frame)
        if norm_lms is not None:
            landmarks = self.roi_tracker.remap_to_global(norm_lms, full_box)
            if self.enable_adaptive_roi:
                self.roi_tracker.update(landmarks)
            return landmarks

        # 3. Multi-Scale Pyramid Acquisition:
        # On frame 0 or when lost: if full-frame detection fails, evaluate central driver quadrant crop
        if self.enable_adaptive_roi:
            quadrant_box = self.roi_tracker.get_driver_quadrant_bounds(frame.shape)
            qx1, qy1, qx2, qy2 = quadrant_box
            quadrant_crop = frame[qy1:qy2, qx1:qx2]
            norm_lms = self._detect_raw(quadrant_crop)
            if norm_lms is not None:
                landmarks = self.roi_tracker.remap_to_global(norm_lms, quadrant_box)
                self.roi_tracker.update(landmarks)
                return landmarks

        return None

    def reset(self) -> None:
        """Reset internal ROI tracker state."""
        self.roi_tracker.reset()

    @property
    def n_landmarks(self) -> int:
        """Number of landmarks per face."""
        return self._n_landmarks

    def close(self):
        """Release MediaPipe resources."""
        if self._mode == "legacy" and hasattr(self, "face_mesh"):
            self.face_mesh.close()
        elif hasattr(self, "face_landmarker"):
            self.face_landmarker.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
