"""
training_buffer.py

Assembles per-frame features into training-ready windowed tensors.
Accumulates FrameFeatures into a rolling buffer and extracts sliding
windows of fixed duration for sequence models.
"""
import json
import uuid
import numpy as np
from enum import Enum
from collections import deque
from dataclasses import dataclass
from typing import List, Optional

from pipeline.feature_extractor import FrameFeatures


class GatedFrameStrategy(Enum):
    DROP = "DROP"
    HOLD_LAST = "HOLD_LAST"
    INTERPOLATE = "INTERPOLATE"


@dataclass
class TrainingWindow:
    features: np.ndarray  # (window_frames, n_features)
    timestamps: np.ndarray  # (window_frames,)
    subject_id: str
    session_id: str
    window_start_time: float
    window_end_time: float
    n_gated_frames: int
    label: Optional[str] = None


class TrainingBuffer:
    FEATURE_NAMES = (
        "ear_left", "ear_right", "ear_avg", "mar", "perclos",
        "left_iris_x", "left_iris_y", "right_iris_x", "right_iris_y",
        "head_yaw", "head_pitch", "head_roll",
        "saccade_velocity", "saccade_latency_ms",
        "blink_closing_velocity", "blink_opening_velocity",
        "gaze_yaw_dispersion",
        "gen_slow_phase_vel", "gen_beat_count",
        "pursuit_fragmentation_ratio",
        "vergence_angle_deg", "vor_gain",
        "blink_detected", "microsleep_detected", "yawn_detected", 
        "saccade_detected", "gen_detected", "lack_of_convergence", "imu_gated"
    )

    def __init__(
        self,
        subject_id: str,
        fps: float = 30.0,
        window_size_sec: float = 5.0,
        stride_sec: float = 2.5,
        gated_strategy: GatedFrameStrategy = GatedFrameStrategy.HOLD_LAST,
        max_session_hours: float = 1.0
    ):
        """
        Initialize the training data buffer.
        
        Args:
            subject_id: String identifier for the subject.
            fps: Expected frames per second.
            window_size_sec: Duration of each extracted window in seconds.
            stride_sec: Stride duration between windows in seconds.
            gated_strategy: Strategy for handling gated/frozen frames.
            max_session_hours: Maximum session duration to keep in the buffer.
        """
        self.subject_id = subject_id
        self.fps = fps
        self.window_size_sec = window_size_sec
        self.stride_sec = stride_sec
        self.gated_strategy = gated_strategy
        
        self.window_frames = int(window_size_sec * fps)
        self.stride_frames = int(stride_sec * fps)
        self.max_frames = int(max_session_hours * 3600 * fps)
        
        self.session_id = str(uuid.uuid4())
        
        # Buffer to store frame data: (timestamp, feature_vector, is_gated)
        self._buffer: deque = deque(maxlen=self.max_frames)
        
        self._lookahead_buffer: deque = deque(maxlen=2)
        self._last_valid_vector = None

    def reset(self):
        """Clear the buffer but keep subject_id and config."""
        self._buffer.clear()
        self._lookahead_buffer.clear()
        self._last_valid_vector = None
        self.session_id = str(uuid.uuid4())

    def __len__(self) -> int:
        """Return number of accumulated samples."""
        return len(self._buffer)
        
    def _extract_vector(self, ff: FrameFeatures) -> np.ndarray:
        numeric = [
            ff.ear_left, ff.ear_right, ff.ear_avg, ff.mar, ff.perclos,
            ff.left_iris_x, ff.left_iris_y, ff.right_iris_x, ff.right_iris_y,
            ff.head_yaw, ff.head_pitch, ff.head_roll,
            ff.saccade_velocity, ff.saccade_latency_ms,
            ff.blink_closing_velocity, ff.blink_opening_velocity,
            ff.gaze_yaw_dispersion,
            ff.gen_slow_phase_vel, float(ff.gen_beat_count),
            ff.pursuit_fragmentation_ratio,
            ff.vergence_angle_deg, ff.vor_gain
        ]
        bools = [
            int(ff.blink_detected), int(ff.microsleep_detected), 
            int(ff.yawn_detected), int(ff.saccade_detected), 
            int(ff.gen_detected), int(ff.lack_of_convergence), int(ff.imu_gated)
        ]
        return np.array(numeric + bools, dtype=np.float32)

    def add_sample(self, ff: FrameFeatures):
        """
        Add a FrameFeatures sample to the buffer, applying the gated frame strategy.
        """
        is_gated = ff.imu_gated or not ff.face_valid
        vec = self._extract_vector(ff)
        
        if self.gated_strategy == GatedFrameStrategy.DROP:
            if is_gated:
                return
            self._buffer.append((ff.timestamp, vec, is_gated))
            
        elif self.gated_strategy == GatedFrameStrategy.HOLD_LAST:
            if is_gated:
                if self._last_valid_vector is not None:
                    held_vec = self._last_valid_vector.copy()
                    # Retain boolean values from current frame
                    held_vec[-7:] = vec[-7:]
                    vec = held_vec
            else:
                self._last_valid_vector = vec
            self._buffer.append((ff.timestamp, vec, is_gated))
            
        elif self.gated_strategy == GatedFrameStrategy.INTERPOLATE:
            if not is_gated:
                # Interpolate delayed frames if any
                if len(self._lookahead_buffer) > 0 and self._last_valid_vector is not None:
                    n_gated = len(self._lookahead_buffer)
                    for i, (g_ts, g_vec, g_is_gated) in enumerate(self._lookahead_buffer):
                        alpha = (i + 1) / (n_gated + 1)
                        interp_vec = (1 - alpha) * self._last_valid_vector + alpha * vec
                        interp_vec[-7:] = g_vec[-7:]
                        self._buffer.append((g_ts, interp_vec, g_is_gated))
                elif len(self._lookahead_buffer) > 0:
                    for (g_ts, g_vec, g_is_gated) in self._lookahead_buffer:
                        g_vec[:-7] = vec[:-7]
                        self._buffer.append((g_ts, g_vec, g_is_gated))
                self._lookahead_buffer.clear()
                self._last_valid_vector = vec
                self._buffer.append((ff.timestamp, vec, is_gated))
            else:
                self._lookahead_buffer.append((ff.timestamp, vec, is_gated))
                if len(self._lookahead_buffer) > 2:
                    oldest_ts, oldest_vec, oldest_is_gated = self._lookahead_buffer.popleft()
                    if self._last_valid_vector is not None:
                        oldest_vec[:-7] = self._last_valid_vector[:-7]
                    self._buffer.append((oldest_ts, oldest_vec, oldest_is_gated))

    def extract_windows(self) -> List[TrainingWindow]:
        """
        Extract fixed-size windows from the accumulated buffer.
        
        Returns:
            A list of TrainingWindow dataclasses.
        """
        if not self._buffer:
            return []
            
        windows = []
        buf_list = list(self._buffer)
        n_samples = len(buf_list)
        
        min_required_frames = int(self.window_frames * 0.8)
        
        start_time = buf_list[0][0]
        end_time = buf_list[-1][0]
        
        current_window_start = start_time
        
        left_idx = 0
        while current_window_start + self.window_size_sec <= end_time + 1e-3:
            current_window_end = current_window_start + self.window_size_sec
            
            while left_idx < n_samples and buf_list[left_idx][0] < current_window_start - 1e-4:
                left_idx += 1
                
            right_idx = left_idx
            while right_idx < n_samples and buf_list[right_idx][0] <= current_window_end + 1e-4:
                right_idx += 1
                
            window_slice = buf_list[left_idx:right_idx]
            
            if len(window_slice) >= min_required_frames:
                feats = np.zeros((self.window_frames, len(self.FEATURE_NAMES)), dtype=np.float32)
                timestamps = np.zeros(self.window_frames, dtype=np.float64)
                
                n_copy = min(len(window_slice), self.window_frames)
                
                feats[:n_copy] = np.array([x[1] for x in window_slice[:n_copy]], dtype=np.float32)
                timestamps[:n_copy] = np.array([x[0] for x in window_slice[:n_copy]], dtype=np.float64)
                
                n_gated = sum(1 for x in window_slice if x[2])
                
                win = TrainingWindow(
                    features=feats,
                    timestamps=timestamps,
                    subject_id=self.subject_id,
                    session_id=self.session_id,
                    window_start_time=current_window_start,
                    window_end_time=current_window_end,
                    n_gated_frames=n_gated,
                    label=None
                )
                windows.append(win)
                
            current_window_start += self.stride_sec
            
        return windows

    def save_to_disk(self, path: str):
        """
        Save all accumulated windows to a .npz file.
        
        Args:
            path: Target file path.
        """
        windows = self.extract_windows()
        if not windows:
            return
            
        features_arr = np.stack([w.features for w in windows])
        timestamps_arr = np.stack([w.timestamps for w in windows])
        subject_ids_arr = np.array([w.subject_id for w in windows], dtype=str)
        labels_arr = np.array([w.label if w.label is not None else "" for w in windows], dtype=str)
        
        meta = {
            "fps": self.fps,
            "window_size_sec": self.window_size_sec,
            "stride_sec": self.stride_sec,
            "gated_strategy": self.gated_strategy.value
        }
        
        np.savez(
            path,
            features=features_arr,
            timestamps=timestamps_arr,
            feature_names=np.array(self.FEATURE_NAMES, dtype=str),
            subject_ids=subject_ids_arr,
            session_id=self.session_id,
            labels=labels_arr,
            metadata=json.dumps(meta)
        )
