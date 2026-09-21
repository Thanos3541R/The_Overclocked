"""Polymorphic frame capture for RGB webcams and grayscale NIR cameras.

Handles:
- USB webcam (default, BGR output)
- Video file playback (.mp4, .avi)
- Single-channel NIR camera (grayscale output)

Auto-detects input type and sets the `is_grayscale` flag accordingly.
"""
import cv2
import numpy as np
from typing import Optional, Tuple, Union


class FrameCapture:
    """Polymorphic video capture abstraction."""

    def __init__(self, source: Union[int, str] = 0,
                 width: int = 1280, height: int = 720):
        """
        Args:
            source: Camera index (int) or video file path (str).
            width: Requested frame width.
            height: Requested frame height.
        """
        self.source = source
        self.cap = cv2.VideoCapture(source)

        if isinstance(source, int):
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open video source: {source}")

        # Read one test frame to detect grayscale vs color
        ret, test_frame = self.cap.read()
        if not ret:
            raise RuntimeError(f"Cannot read from video source: {source}")

        self.is_grayscale = (len(test_frame.shape) == 2 or
                            (len(test_frame.shape) == 3 and test_frame.shape[2] == 1))

        self.frame_width = test_frame.shape[1]
        self.frame_height = test_frame.shape[0]
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0

        # Reset to beginning for video files
        if isinstance(source, str):
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        self._frame_count = 0

    def read(self) -> Optional[np.ndarray]:
        """Read the next frame.

        Returns:
            Frame as numpy array (BGR for color, single-channel for NIR),
            or None if no frame available.
        """
        ret, frame = self.cap.read()
        if not ret:
            return None

        # Normalize single-channel to 2D array
        if self.is_grayscale and len(frame.shape) == 3:
            frame = frame[:, :, 0]

        self._frame_count += 1
        return frame

    @property
    def frame_count(self) -> int:
        """Total frames read so far."""
        return self._frame_count

    def is_open(self) -> bool:
        """Check if capture is still open."""
        return self.cap.isOpened()

    def release(self):
        """Release the video capture resource."""
        self.cap.release()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.release()
