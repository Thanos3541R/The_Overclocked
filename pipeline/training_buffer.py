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
    valid_mask: Optional[np.ndarray] = None
    audio_data: Optional[np.ndarray] = None  # (window_frames, 7) supplementary audio channels


AUDIO_FEATURE_NAMES: List[str] = [
    "audio_valid",
    "vocal_jitter_pct",
    "vocal_shimmer_pct",
    "vocal_hnr_db",
    "formant_f2_slope_hz_ms",
    "speech_pause_ratio",
    "voice_impairment_score",
]


# Shared physiological outlier caps for train-serve consistency
MAX_SACCADE_LATENCY_MS: float = 10000.0
MAX_STARE_DURATION_S: float = 30.0

CORE_NUMERIC_NAMES = (
    "ear_left", "ear_right", "ear_avg", "mar", "perclos",
    "left_iris_x", "left_iris_y", "right_iris_x", "right_iris_y",
    "head_yaw", "head_pitch", "head_roll",
    "saccade_velocity", "saccade_latency_ms",
    "blink_closing_velocity", "blink_opening_velocity",
    "gaze_yaw_dispersion",
    "gen_slow_phase_vel", "gen_beat_count",
    "pursuit_fragmentation_ratio",
    "vergence_angle_deg", "vor_gain",
)  # 22 core numeric

EXTENDED_NUMERIC_NAMES = CORE_NUMERIC_NAMES + (
    "raw_iris_dx",
    "head_postural_sway", "facial_flushing_ratio", "flushing_delta",
    "stare_fixation_duration_s", "masking_divergence_score",
    "brow_eye_distance_ratio", "brow_elevation_delta", "eyelid_tremor_hz",
    "rppg_pulse_bpm", "rppg_hrv_rmssd_ms",
)  # 33 extended numeric

CORE_BOOL_NAMES = (
    "blink_detected", "microsleep_detected", "yawn_detected",
    "saccade_detected", "gen_detected", "lack_of_convergence", "imu_gated",
)  # 7 core booleans (imu_gated at -1)

EXTENDED_BOOL_NAMES = (
    "blink_detected", "microsleep_detected", "yawn_detected",
    "saccade_detected", "gen_detected", "lack_of_convergence",
    "flushing_valid", "deliberate_blink_suppression", "voluntary_eye_widening",
    "compensatory_stare_active", "rigid_head_stabilization", "masking_detected",
    "frontalis_compensation_active", "rppg_valid", "imu_gated",
)  # 15 extended booleans (imu_gated at -1)

FEATURE_NAMES = CORE_NUMERIC_NAMES + CORE_BOOL_NAMES              # 29 channels
EXTENDED_FEATURE_NAMES = EXTENDED_NUMERIC_NAMES + EXTENDED_BOOL_NAMES  # 48 channels

VOICE_NUMERIC_NAMES: List[str] = [
    "vocal_jitter_pct",
    "vocal_shimmer_pct",
    "vocal_hnr_db",
    "formant_f2_slope_hz_ms",
    "speech_pause_ratio",
    "voice_impairment_score",
]

VOICE_BOOL_NAMES: List[str] = [
    "audio_valid",
    "voice_slurred_detected",
]


class TrainingBuffer:
    CORE_NUMERIC_NAMES = CORE_NUMERIC_NAMES
    EXTENDED_NUMERIC_NAMES = EXTENDED_NUMERIC_NAMES
    CORE_BOOL_NAMES = CORE_BOOL_NAMES
    EXTENDED_BOOL_NAMES = EXTENDED_BOOL_NAMES
    VOICE_NUMERIC_NAMES = VOICE_NUMERIC_NAMES
    VOICE_BOOL_NAMES = VOICE_BOOL_NAMES
    AUDIO_FEATURE_NAMES = AUDIO_FEATURE_NAMES
    FEATURE_NAMES = FEATURE_NAMES
    EXTENDED_FEATURE_NAMES = EXTENDED_FEATURE_NAMES
    MAX_SACCADE_LATENCY_MS = MAX_SACCADE_LATENCY_MS
    MAX_STARE_DURATION_S = MAX_STARE_DURATION_S

    def __init__(
        self,
        subject_id: str,
        fps: float = 30.0,
        window_size_sec: float = 3.0,
        stride_sec: float = 1.5,
        min_fill_ratio: float = 0.70,
        gated_strategy: GatedFrameStrategy = GatedFrameStrategy.HOLD_LAST,
        max_session_hours: float = 1.0,
        default_label: Optional[str] = None,
        include_extended_features: bool = False,
        include_voice_features: bool = False,
    ):
        """
        Initialize the training data buffer.
        
        Args:
            subject_id: String identifier for the subject.
            fps: Expected frames per second.
            window_size_sec: Duration of each extracted window in seconds (default 3.0s).
            stride_sec: Stride duration between windows in seconds (default 1.5s).
            min_fill_ratio: Minimum fraction of non-empty window frames required (default 0.70).
            gated_strategy: Strategy for handling gated/frozen frames.
            max_session_hours: Maximum session duration to keep in the buffer.
            default_label: Optional default ground-truth label for extracted windows.
            include_extended_features: If True, uses all 48 features (33 numeric + 15 boolean);
                                       if False (default), uses the 29 core features.
            include_voice_features: If True, appends 8 voice features (6 numeric + 2 boolean).
        """
        self.subject_id = subject_id
        self.fps = fps
        self.window_size_sec = window_size_sec
        self.stride_sec = stride_sec
        self.min_fill_ratio = min_fill_ratio
        self.gated_strategy = gated_strategy
        self.default_label = default_label
        self.include_extended_features = bool(include_extended_features)
        self.include_voice_features = bool(include_voice_features)

        if self.include_extended_features:
            self._numeric_names = list(self.EXTENDED_NUMERIC_NAMES)
            self._bool_names = list(self.EXTENDED_BOOL_NAMES)
        else:
            self._numeric_names = list(self.CORE_NUMERIC_NAMES)
            self._bool_names = list(self.CORE_BOOL_NAMES)

        if self.include_voice_features:
            self._numeric_names = self._numeric_names + list(self.VOICE_NUMERIC_NAMES)
            self._bool_names = self._bool_names + list(self.VOICE_BOOL_NAMES)

        self.numeric_names = list(self._numeric_names)
        self.bool_names = list(self._bool_names)
        self.feature_names = tuple(self._numeric_names + self._bool_names)
        self._n_bool_features = len(self._bool_names)
        
        self.window_frames = int(window_size_sec * fps)
        self.stride_frames = int(stride_sec * fps)
        self.max_frames = int(max_session_hours * 3600 * fps)
        
        self.session_id = str(uuid.uuid4())
        
        # Buffer to store frame data: (timestamp, feature_vector, is_gated)
        self._buffer: deque = deque(maxlen=self.max_frames)
        self._audio_buffer: deque = deque(maxlen=self.max_frames)
        
        self._lookahead_buffer: deque = deque(maxlen=2)
        self._audio_lookahead_buffer: deque = deque(maxlen=2)
        self._last_valid_vector = None

    def reset(self):
        """Clear the buffer but keep subject_id and config."""
        self._buffer.clear()
        self._audio_buffer.clear()
        self._lookahead_buffer.clear()
        self._audio_lookahead_buffer.clear()
        self._last_valid_vector = None
        self.session_id = str(uuid.uuid4())

    clear = reset

    def push(self, features: FrameFeatures):
        """Alias for add_sample."""
        self.add_sample(features)

    def __len__(self) -> int:
        """Return number of accumulated samples."""
        return len(self._buffer)

    @property
    def n_features(self) -> int:
        """Return total number of features in the assembled vector."""
        return len(self.feature_names)

    @classmethod
    def clamp_numeric_feature(cls, name: str, val: float) -> float:
        """Apply identical physiological outlier bounds across training and live inference."""
        if name == "saccade_latency_ms":
            return min(val, cls.MAX_SACCADE_LATENCY_MS)
        if name == "stare_fixation_duration_s":
            return min(val, cls.MAX_STARE_DURATION_S)
        return val

    def _extract_vector(self, ff: FrameFeatures) -> np.ndarray:
        numeric = [
            self.clamp_numeric_feature(name, float(getattr(ff, name, 0.0)))
            for name in self._numeric_names
        ]
        bools = [int(bool(getattr(ff, name, False))) for name in self._bool_names]
        return np.array(numeric + bools, dtype=np.float32)

    _features_to_vector = _extract_vector

    def _extract_audio_vector(self, ff: FrameFeatures) -> np.ndarray:
        return np.array([
            float(bool(getattr(ff, "audio_valid", False))),
            float(getattr(ff, "vocal_jitter_pct", 0.0)),
            float(getattr(ff, "vocal_shimmer_pct", 0.0)),
            float(getattr(ff, "vocal_hnr_db", 0.0)),
            float(getattr(ff, "formant_f2_slope_hz_ms", 0.0)),
            float(getattr(ff, "speech_pause_ratio", 0.0)),
            float(getattr(ff, "voice_impairment_score", 0.0)),
        ], dtype=np.float32)

    def add_sample(self, ff: FrameFeatures):
        """
        Add a FrameFeatures sample to the buffer, applying the gated frame strategy.
        """
        is_gated = ff.imu_gated or not ff.face_valid
        vec = self._extract_vector(ff)
        audio_vec = self._extract_audio_vector(ff)
        n_b = self._n_bool_features
        
        if self.gated_strategy == GatedFrameStrategy.DROP:
            if is_gated:
                return
            self._buffer.append((ff.timestamp, vec, is_gated))
            self._audio_buffer.append(audio_vec)
            
        elif self.gated_strategy == GatedFrameStrategy.HOLD_LAST:
            if is_gated:
                if self._last_valid_vector is not None:
                    held_vec = self._last_valid_vector.copy()
                    # Retain boolean values from current frame
                    held_vec[-n_b:] = vec[-n_b:]
                    vec = held_vec
            else:
                self._last_valid_vector = vec
            self._buffer.append((ff.timestamp, vec, is_gated))
            self._audio_buffer.append(audio_vec)
            
        elif self.gated_strategy == GatedFrameStrategy.INTERPOLATE:
            if not is_gated:
                # Interpolate delayed frames if any
                if len(self._lookahead_buffer) > 0 and self._last_valid_vector is not None:
                    n_gated = len(self._lookahead_buffer)
                    for i, (g_ts, g_vec, g_is_gated) in enumerate(self._lookahead_buffer):
                        alpha = (i + 1) / (n_gated + 1)
                        interp_vec = (1 - alpha) * self._last_valid_vector + alpha * vec
                        interp_vec[-n_b:] = g_vec[-n_b:]
                        self._buffer.append((g_ts, interp_vec, g_is_gated))
                    for a_vec in self._audio_lookahead_buffer:
                        self._audio_buffer.append(a_vec)
                elif len(self._lookahead_buffer) > 0:
                    for (g_ts, g_vec, g_is_gated) in self._lookahead_buffer:
                        g_vec[:-n_b] = vec[:-n_b]
                        self._buffer.append((g_ts, g_vec, g_is_gated))
                    for a_vec in self._audio_lookahead_buffer:
                        self._audio_buffer.append(a_vec)
                self._lookahead_buffer.clear()
                self._audio_lookahead_buffer.clear()
                self._last_valid_vector = vec
                self._buffer.append((ff.timestamp, vec, is_gated))
                self._audio_buffer.append(audio_vec)
            else:
                self._lookahead_buffer.append((ff.timestamp, vec, is_gated))
                self._audio_lookahead_buffer.append(audio_vec)
                if len(self._lookahead_buffer) > 2:
                    oldest_ts, oldest_vec, oldest_is_gated = self._lookahead_buffer.popleft()
                    oldest_audio_vec = self._audio_lookahead_buffer.popleft()
                    if self._last_valid_vector is not None:
                        oldest_vec[:-n_b] = self._last_valid_vector[:-n_b]
                    self._buffer.append((oldest_ts, oldest_vec, oldest_is_gated))
                    self._audio_buffer.append(oldest_audio_vec)

    def extract_windows(self, label: Optional[str] = None) -> List[TrainingWindow]:
        """
        Extract fixed-size windows from the accumulated buffer.
        
        Args:
            label: Optional ground truth label to assign to all extracted windows.
                   If None, defaults to self.default_label.
        
        Returns:
            A list of TrainingWindow dataclasses.
        """
        if not self._buffer:
            return []
            
        windows = []
        buf_list = list(self._buffer)
        audio_list = list(self._audio_buffer)
        n_samples = len(buf_list)
        win_label = label if label is not None else self.default_label
        
        min_required_frames = int(self.window_frames * self.min_fill_ratio)
        
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
                n_copy = min(len(window_slice), self.window_frames)
                n_valid = sum(1 for x in window_slice[:n_copy] if not x[2])
                if n_valid < min_required_frames:
                    current_window_start += self.stride_sec
                    continue

                feats = np.zeros((self.window_frames, len(self.feature_names)), dtype=np.float32)
                timestamps = np.zeros(self.window_frames, dtype=np.float64)
                valid_mask = np.zeros(self.window_frames, dtype=bool)
                audio_feats = np.zeros((self.window_frames, len(AUDIO_FEATURE_NAMES)), dtype=np.float32)

                feats[:n_copy] = np.array([x[1] for x in window_slice[:n_copy]], dtype=np.float32)
                timestamps[:n_copy] = np.array([x[0] for x in window_slice[:n_copy]], dtype=np.float64)
                valid_mask[:n_copy] = np.array([not x[2] for x in window_slice[:n_copy]], dtype=bool)

                if len(audio_list) >= right_idx:
                    audio_slice = audio_list[left_idx:right_idx]
                    n_audio_copy = min(len(audio_slice), self.window_frames)
                    if n_audio_copy > 0:
                        audio_feats[:n_audio_copy] = np.array(audio_slice[:n_audio_copy], dtype=np.float32)

                n_gated = sum(1 for x in window_slice if x[2])

                win = TrainingWindow(
                    features=feats,
                    timestamps=timestamps,
                    subject_id=self.subject_id,
                    session_id=self.session_id,
                    window_start_time=current_window_start,
                    window_end_time=current_window_end,
                    n_gated_frames=n_gated,
                    label=win_label,
                    valid_mask=valid_mask,
                    audio_data=audio_feats,
                )
                windows.append(win)

            current_window_start += self.stride_sec

        return windows

    def save_to_disk(self, path: str, label: Optional[str] = None):
        """
        Save all accumulated windows to a .npz file.

        Args:
            path: Target file path.
            label: Optional ground truth label to assign to windows.
        """
        windows = self.extract_windows(label=label)
        if not windows:
            return

        features_arr = np.stack([w.features for w in windows])
        timestamps_arr = np.stack([w.timestamps for w in windows])
        valid_masks_arr = np.stack([
            w.valid_mask if w.valid_mask is not None else np.ones(w.features.shape[0], dtype=bool)
            for w in windows
        ])
        subject_ids_arr = np.array([w.subject_id for w in windows], dtype=str)
        labels_arr = np.array([w.label if w.label is not None else "" for w in windows], dtype=str)

        meta = {
            "fps": self.fps,
            "window_size_sec": self.window_size_sec,
            "stride_sec": self.stride_sec,
            "min_fill_ratio": self.min_fill_ratio,
            "gated_strategy": self.gated_strategy.value
        }

        save_dict = {
            "features": features_arr,
            "timestamps": timestamps_arr,
            "valid_masks": valid_masks_arr,
            "feature_names": np.array(self.feature_names, dtype=str),
            "subject_ids": subject_ids_arr,
            "session_id": self.session_id,
            "labels": labels_arr,
            "metadata": json.dumps(meta),
        }

        has_audio = self.include_voice_features or any(
            getattr(w, "audio_data", None) is not None
            and np.any(w.audio_data[:, 0] > 0.0)
            for w in windows
        )
        if has_audio:
            audio_arr = np.stack([
                w.audio_data if getattr(w, "audio_data", None) is not None
                else np.zeros((w.features.shape[0], len(AUDIO_FEATURE_NAMES)), dtype=np.float32)
                for w in windows
            ])
            save_dict["X_audio"] = audio_arr
            save_dict["audio_features"] = audio_arr
            save_dict["audio_feature_names"] = np.array(AUDIO_FEATURE_NAMES, dtype=str)

        np.savez_compressed(path, **save_dict)

    @staticmethod
    def load_from_disk(path: str) -> List[TrainingWindow]:
        """Load TrainingWindow objects from a saved .npz file."""
        data = np.load(path, allow_pickle=True)
        features = data["features"]
        timestamps = data["timestamps"]
        valid_masks = data["valid_masks"] if "valid_masks" in data else None
        subject_ids = data["subject_ids"] if "subject_ids" in data else np.array([""] * len(features))
        labels = data["labels"] if "labels" in data else np.array([""] * len(features))
        session_id = str(data.get("session_id", ""))
        x_audio = data["X_audio"] if "X_audio" in data else (data["audio_features"] if "audio_features" in data else None)

        windows = []
        for i in range(len(features)):
            feats = features[i]
            ts = timestamps[i]
            vm = valid_masks[i] if valid_masks is not None else (ts > 0.0)
            n_gated = int(np.sum(~vm))
            a_data = x_audio[i] if x_audio is not None and i < len(x_audio) else None
            w = TrainingWindow(
                features=feats,
                timestamps=ts,
                subject_id=str(subject_ids[i]) if i < len(subject_ids) else "",
                session_id=session_id,
                window_start_time=float(ts[0]) if len(ts) > 0 else 0.0,
                window_end_time=float(ts[-1]) if len(ts) > 0 else 0.0,
                n_gated_frames=n_gated,
                label=str(labels[i]) if i < len(labels) else None,
                valid_mask=vm,
                audio_data=a_data,
            )
            windows.append(w)
        return windows
