"""Polymorphic frame capture for RGB webcams and grayscale NIR cameras.

Handles:
- USB webcam (default, BGR output)
- Video file playback (.mp4, .avi)
- Single-channel NIR camera (grayscale output)

Auto-detects input type and sets the `is_grayscale` flag accordingly.
"""
import logging
import sys
import threading
import time
import cv2
import numpy as np
from typing import Optional, Tuple, Union

logger = logging.getLogger(__name__)


class FrameCapture:
    """Polymorphic video capture abstraction with threaded async capture for live cameras."""

    def __init__(self, source: Union[int, str] = 0,
                 width: int = 1280, height: int = 720,
                 target_fps: float = 60.0,
                 anti_motion_blur: bool = False):
        """
        Args:
            source: Camera index (int) or video file path (str).
            width: Requested frame width.
            height: Requested frame height.
            target_fps: Target camera capture framerate (default 60.0).
            anti_motion_blur: Enable low-exposure / anti-motion blur capture settings.
        """
        self.source = source
        self.target_fps = target_fps
        self.anti_motion_blur = anti_motion_blur

        is_camera = isinstance(source, int)
        if isinstance(source, str) and source.strip().isdigit():
            source = int(source.strip())
            self.source = source
            is_camera = True

        self.is_camera = is_camera

        if is_camera:
            if sys.platform.startswith("win"):
                self.cap = cv2.VideoCapture(source, cv2.CAP_DSHOW)
                if not self.cap.isOpened():
                    self.cap = cv2.VideoCapture(source)
            else:
                self.cap = cv2.VideoCapture(source)
            self._cap = self.cap
            try:
                self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
            except Exception:
                pass
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            self.cap.set(cv2.CAP_PROP_FPS, target_fps)
            try:
                self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            except Exception:
                pass
            if anti_motion_blur:
                try:
                    self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
                    self.cap.set(cv2.CAP_PROP_EXPOSURE, -6)
                except Exception:
                    pass
        else:
            self.cap = cv2.VideoCapture(source)
            self._cap = self.cap

        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open video source: {source}")

        # Read one test frame to detect grayscale vs color
        ret, test_frame = self.cap.read()
        if not ret:
            self.cap.release()
            raise RuntimeError(f"Cannot read from video source: {source}")

        self.is_grayscale = (len(test_frame.shape) == 2 or
                            (len(test_frame.shape) == 3 and test_frame.shape[2] == 1))

        self.frame_width = test_frame.shape[1]
        self.frame_height = test_frame.shape[0]
        fps_val = self.cap.get(cv2.CAP_PROP_FPS)
        self.fps = float(fps_val) if (fps_val and fps_val > 0 and not np.isnan(fps_val)) else 30.0
        self.actual_capture_fps = float(self.fps)

        # Log negotiated resolution
        print(f"[DMS Capture] Camera opened: {self.frame_width}x{self.frame_height} @ {self.fps} fps")

        # Reset to beginning for video files
        if not is_camera and isinstance(self.source, str):
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        self._frame_count = 0
        self._lock = threading.Lock()
        self._condition = threading.Condition(self._lock)
        self._raw_frame_seq: int = 1
        self._consumed_frame_seq: int = 0
        self._latest_frame_timestamp: float = time.perf_counter()
        if self.is_camera:
            self.last_frame_timestamp: float = self._latest_frame_timestamp
            self._last_video_timestamp: float = -1.0
        else:
            pos_msec = self.cap.get(cv2.CAP_PROP_POS_MSEC)
            init_ts = (pos_msec / 1000.0) if (pos_msec is not None and pos_msec > 0) else (1.0 / max(self.fps, 1.0))
            self.last_frame_timestamp: float = init_ts
            self._last_video_timestamp: float = init_ts
        self.last_frame_is_new: bool = False
        self._running = False
        self._latest_frame: Optional[np.ndarray] = test_frame.copy()
        self._thread: Optional[threading.Thread] = None

        if self.is_camera:
            self._running = True
            self._thread = threading.Thread(target=self._capture_worker, daemon=True)
            self._thread.start()

    def _capture_worker(self):
        """Asynchronous background worker continuously polling live camera frames."""
        t_prev = time.perf_counter()
        fps_ema = float(self.fps)
        while self._running:
            try:
                if not hasattr(self, "cap") or not self.cap.isOpened():
                    break
                ret, frame = self.cap.read()
            except Exception as exc:
                logger.debug("Exception in FrameCapture._capture_worker: %s", exc, exc_info=True)
                break

            if not ret or frame is None:
                time.sleep(0.005)
                continue
            t_now = time.perf_counter()
            dt = t_now - t_prev
            if dt > 1e-4:
                fps_ema = 0.92 * fps_ema + 0.08 * (1.0 / dt)
            t_prev = t_now
            self.actual_capture_fps = fps_ema
            with self._condition:
                self._latest_frame = frame
                self._latest_frame_timestamp = t_now
                self._raw_frame_seq += 1
                self._condition.notify_all()
            time.sleep(0.001)

    def read(self) -> Optional[np.ndarray]:
        """Read the next frame.

        Returns:
            Frame as numpy array (BGR for color, single-channel for NIR),
            or None if no frame available.
        """
        if not self.is_camera:
            ret, frame = self.cap.read()
            if not ret or frame is None:
                self.last_frame_is_new = False
                return None
            if self.is_grayscale and len(frame.shape) == 3:
                frame = frame[:, :, 0]
            pos_msec = self.cap.get(cv2.CAP_PROP_POS_MSEC)
            if pos_msec is not None and pos_msec > 0 and (pos_msec / 1000.0) >= self._last_video_timestamp:
                self.last_frame_timestamp = pos_msec / 1000.0
                self._last_video_timestamp = self.last_frame_timestamp
            else:
                self.last_frame_timestamp = max(
                    self._last_video_timestamp + (1.0 / max(self.fps, 1.0)),
                    self._frame_count / max(self.fps, 1.0)
                )
                self._last_video_timestamp = self.last_frame_timestamp
            self.last_frame_is_new = True
            self._frame_count += 1
            return frame

        with self._condition:
            prev_consumed = self._consumed_frame_seq
            if self._raw_frame_seq == self._consumed_frame_seq and self._running:
                # Wait up to one frame interval (max 50ms) for a fresh hardware frame
                wait_timeout = min(0.05, 1.5 / max(self.fps, 15.0))
                self._condition.wait(timeout=wait_timeout)
            frame = self._latest_frame if self._latest_frame is not None else None
            self.last_frame_timestamp = self._latest_frame_timestamp
            self.last_frame_is_new = (self._raw_frame_seq != prev_consumed) and (frame is not None)
            self._consumed_frame_seq = self._raw_frame_seq

        if frame is None:
            return None

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
        """Release the video capture resource and background thread."""
        with self._condition:
            self._running = False
            self._condition.notify_all()
        if hasattr(self, "_thread") and self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=0.5)
        if hasattr(self, "cap") and self.cap.isOpened():
            try:
                self.cap.release()
            except Exception as exc:
                logger.debug("Exception while releasing capture: %s", exc, exc_info=True)
        if hasattr(self, "_thread") and self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=0.2)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.release()
