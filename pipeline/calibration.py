"""Personal baseline auto-calibration engine for driver monitoring.

Establishes driver-specific biometric baselines during the initial drive period (30-60 seconds)
to prevent false positives from natural anatomical variation (e.g. resting eyelid asymmetry,
monolid vs double eyelid morphology, natural facial chromaticity, and unique blink kinetics).

Baseline metrics tracked:
  - awake_ear_baseline: median EAR when eyes are open (non-blink)
  - resting_mar_baseline: median mouth aspect ratio when mouth is closed
  - blink_upstroke_baseline: median eyelid reopening velocity (EAR/s)
  - facial_flushing_baseline: median cheek chromaticity ratio R/((G+B)/2)
  - resting_asymmetry_baseline: median |EAR_left - EAR_right|
  - resting_head_pitch / roll: median natural seating posture
"""
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, List, Optional
import numpy as np

from pipeline.feature_extractor import FrameFeatures


@dataclass
class PersonalBaselines:
    """Calibrated biometric baseline parameters for an individual driver."""
    ear_baseline: float = 0.28
    mar_baseline: float = 0.15
    blink_upstroke_baseline: float = 2.5
    flushing_baseline: float = 1.35
    asymmetry_baseline: float = 0.0
    head_pitch_baseline: float = 0.0
    head_roll_baseline: float = 0.0
    is_calibrated: bool = False
    samples_collected: int = 0
    duration_s: float = 0.0


class PersonalBaselineCalibrator:
    """Accumulates driver-specific biometric features over initial drive time to compute median baselines."""

    def __init__(self,
                 calibration_duration_sec: float = 45.0,
                 min_samples: int = 150,
                 fps: float = 30.0):
        """
        Args:
            calibration_duration_sec: Duration in seconds over which to accumulate baseline samples.
            min_samples: Minimum valid frame samples required before marking calibration as ready.
            fps: Expected video stream frame rate.
        """
        self.calibration_duration_sec = calibration_duration_sec
        self.min_samples = min_samples
        self.fps = fps

        max_history = int(calibration_duration_sec * fps * 1.5)
        self._ear_samples: Deque[float] = deque(maxlen=max_history)
        self._mar_samples: Deque[float] = deque(maxlen=max_history)
        self._blink_up_samples: Deque[float] = deque(maxlen=100)
        self._flushing_samples: Deque[float] = deque(maxlen=max_history)
        self._asymmetry_samples: Deque[float] = deque(maxlen=max_history)
        self._pitch_samples: Deque[float] = deque(maxlen=max_history)
        self._roll_samples: Deque[float] = deque(maxlen=max_history)

        self._start_timestamp: Optional[float] = None
        self._last_timestamp: float = 0.0
        self._is_calibrated: bool = False
        self._frozen_baselines: Optional[PersonalBaselines] = None

    @property
    def is_calibrated(self) -> bool:
        """Returns True if the calibrator has finalized baseline estimates."""
        return self._is_calibrated

    @property
    def progress(self) -> float:
        """Calibration progress fraction from 0.0 to 1.0."""
        if self._is_calibrated:
            return 1.0
        if self._start_timestamp is None:
            return 0.0
        elapsed = max(0.0, self._last_timestamp - self._start_timestamp)
        time_ratio = min(1.0, elapsed / max(1.0, self.calibration_duration_sec))
        sample_ratio = min(1.0, len(self._ear_samples) / max(1, self.min_samples))
        return float(min(time_ratio, sample_ratio))

    def update(self, features: FrameFeatures) -> None:
        """Ingest a single frame features object into the calibration buffer.

        Gated frames (imu_gated or not face_valid) are excluded to prevent corrupted samples.
        """
        if self._is_calibrated:
            return

        if not features.face_valid or features.imu_gated:
            return

        if self._start_timestamp is None:
            self._start_timestamp = features.timestamp
        self._last_timestamp = features.timestamp

        # Only accumulate open-eye EAR (exclude blinks and microsleeps)
        if features.ear_avg > 0.18 and not features.blink_detected:
            self._ear_samples.append(features.ear_avg)
            asym = abs(features.ear_left - features.ear_right)
            self._asymmetry_samples.append(asym)

        # Only accumulate resting MAR (exclude yawns and mouth open speech)
        if 0.02 < features.mar < 0.40 and not features.yawn_detected:
            self._mar_samples.append(features.mar)

        # Accumulate completed blink upstroke velocities
        if features.blink_opening_velocity > 0.20:
            self._blink_up_samples.append(features.blink_opening_velocity)

        # Accumulate valid RGB facial flushing ratios
        if features.flushing_valid and features.facial_flushing_ratio > 0.5:
            self._flushing_samples.append(features.facial_flushing_ratio)

        # Accumulate head pose
        self._pitch_samples.append(features.head_pitch)
        self._roll_samples.append(features.head_roll)

        # Check if calibration criteria met
        elapsed = features.timestamp - self._start_timestamp
        if elapsed >= self.calibration_duration_sec and len(self._ear_samples) >= self.min_samples:
            self.finalize()

    def finalize(self) -> PersonalBaselines:
        """Lock in calibration and calculate final baseline medians."""
        ear_base = float(np.median(self._ear_samples)) if self._ear_samples else 0.28
        mar_base = float(np.median(self._mar_samples)) if self._mar_samples else 0.15
        blink_up_base = float(np.median(self._blink_up_samples)) if self._blink_up_samples else 2.5
        flush_base = float(np.median(self._flushing_samples)) if self._flushing_samples else 1.35
        asym_base = float(np.median(self._asymmetry_samples)) if self._asymmetry_samples else 0.0
        pitch_base = float(np.median(self._pitch_samples)) if self._pitch_samples else 0.0
        roll_base = float(np.median(self._roll_samples)) if self._roll_samples else 0.0

        elapsed = (self._last_timestamp - self._start_timestamp) if self._start_timestamp else 0.0

        self._frozen_baselines = PersonalBaselines(
            ear_baseline=ear_base,
            mar_baseline=mar_base,
            blink_upstroke_baseline=blink_up_base,
            flushing_baseline=flush_base,
            asymmetry_baseline=asym_base,
            head_pitch_baseline=pitch_base,
            head_roll_baseline=roll_base,
            is_calibrated=True,
            samples_collected=len(self._ear_samples),
            duration_s=float(elapsed)
        )
        self._is_calibrated = True
        return self._frozen_baselines

    def get_baselines(self) -> PersonalBaselines:
        """Return current baselines (or default baselines if calibration not yet complete)."""
        if self._frozen_baselines is not None:
            return self._frozen_baselines

        ear_base = float(np.median(self._ear_samples)) if len(self._ear_samples) >= 30 else 0.28
        mar_base = float(np.median(self._mar_samples)) if len(self._mar_samples) >= 30 else 0.15
        blink_up_base = float(np.median(self._blink_up_samples)) if len(self._blink_up_samples) >= 3 else 2.5
        flush_base = float(np.median(self._flushing_samples)) if len(self._flushing_samples) >= 30 else 1.35
        asym_base = float(np.median(self._asymmetry_samples)) if len(self._asymmetry_samples) >= 30 else 0.0
        pitch_base = float(np.median(self._pitch_samples)) if len(self._pitch_samples) >= 30 else 0.0
        roll_base = float(np.median(self._roll_samples)) if len(self._roll_samples) >= 30 else 0.0

        elapsed = (self._last_timestamp - self._start_timestamp) if self._start_timestamp else 0.0

        return PersonalBaselines(
            ear_baseline=ear_base,
            mar_baseline=mar_base,
            blink_upstroke_baseline=blink_up_base,
            flushing_baseline=flush_base,
            asymmetry_baseline=asym_base,
            head_pitch_baseline=pitch_base,
            head_roll_baseline=roll_base,
            is_calibrated=self._is_calibrated,
            samples_collected=len(self._ear_samples),
            duration_s=float(elapsed)
        )

    def apply_to(self, extractor: Any, classifier: Optional[Any] = None) -> None:
        """Inject calibrated personal baselines into FeatureExtractor and ImpairmentClassifier instances."""
        b = self.get_baselines()
        if hasattr(extractor, "sober_ear_baseline"):
            extractor.sober_ear_baseline = b.ear_baseline
        if hasattr(extractor, "sober_blink_up_baseline"):
            extractor.sober_blink_up_baseline = b.blink_upstroke_baseline
        if hasattr(extractor, "baseline_flushing"):
            extractor.baseline_flushing = b.flushing_baseline

        if classifier is not None:
            if hasattr(classifier, "sober_ear"):
                classifier.sober_ear = b.ear_baseline
            if hasattr(classifier, "sober_blink_up"):
                classifier.sober_blink_up = b.blink_upstroke_baseline
            if hasattr(classifier, "baseline_asymmetry"):
                classifier.baseline_asymmetry = b.asymmetry_baseline
