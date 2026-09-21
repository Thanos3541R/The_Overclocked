"""MediaPipe Face Mesh wrapper for driver monitoring.

Detects 478 facial landmarks (468 face mesh + 10 iris).
Supports both:
  1. Modern MediaPipe Tasks API (`vision.FaceLandmarker` with `face_landmarker.task`)
  2. Legacy `mp.solutions.face_mesh` (for older installations)

Critical implementation detail:
  MediaPipe expects RGB input. OpenCV reads BGR.
  This wrapper handles the conversion internally.
"""
import os
import urllib.request
import cv2
import numpy as np
import mediapipe as mp
from typing import Optional

MODEL_URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task"


class FaceMeshDetector:
    """Face landmark detector configured for driver monitoring."""

    def __init__(self, max_faces: int = 1,
                 refine_landmarks: bool = True,
                 min_detection_confidence: float = 0.5,
                 min_tracking_confidence: float = 0.5):
        """
        Args:
            max_faces: Maximum faces to detect (1 for driver-only).
            refine_landmarks: If True, enables iris landmarks (468-477).
            min_detection_confidence: Minimum face detection confidence.
            min_tracking_confidence: Minimum landmark tracking confidence.
        """
        self._n_landmarks = 478 if refine_landmarks else 468
        self._mode = "unknown"

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

    def detect(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """Detect face and extract landmarks.

        Args:
            frame: Input frame in BGR (color) or grayscale format.

        Returns:
            Landmarks as numpy array of shape (478, 3) in pixel coordinates
            (x_px, y_px, z_relative), or None if no face detected.
        """
        # Prepare RGB image for MediaPipe
        if len(frame.shape) == 2:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
        else:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        h, w = frame.shape[:2]

        if self._mode == "legacy":
            results = self.face_mesh.process(rgb_frame)
            if not results.multi_face_landmarks:
                return None
            face = results.multi_face_landmarks[0]
            return np.array(
                [[lm.x * w, lm.y * h, lm.z * w] for lm in face.landmark],
                dtype=np.float64
            )
        else:
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            results = self.face_landmarker.detect(mp_image)
            if not results.face_landmarks:
                return None
            face = results.face_landmarks[0]
            return np.array(
                [[lm.x * w, lm.y * h, lm.z * w] for lm in face],
                dtype=np.float64
            )

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
