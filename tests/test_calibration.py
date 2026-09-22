"""Unit tests for online personal baseline auto-calibration (pipeline/calibration.py)."""
import pytest
import numpy as np

from pipeline.calibration import PersonalBaselineCalibrator, PersonalBaselines
from pipeline.feature_extractor import FeatureExtractor, FrameFeatures
from pipeline.classifier import ImpairmentClassifier


def test_calibrator_initial_state():
    calibrator = PersonalBaselineCalibrator(calibration_duration_sec=30.0, min_samples=100, fps=30.0)
    assert not calibrator.is_calibrated
    assert calibrator.progress == 0.0

    b = calibrator.get_baselines()
    assert b.ear_baseline == 0.28
    assert b.blink_upstroke_baseline == 2.5
    assert b.flushing_baseline == 1.35
    assert b.asymmetry_baseline == 0.0


def test_calibrator_accumulates_and_finalizes():
    calibrator = PersonalBaselineCalibrator(calibration_duration_sec=5.0, min_samples=50, fps=30.0)

    # Ingest 150 frames spanning 5 seconds of healthy, alert subject
    # Subject has resting EAR ~0.33, blink upstroke ~2.9, flushing ~1.42
    for i in range(160):
        t = i * (1.0 / 30.0)
        ff = FrameFeatures(
            timestamp=t,
            frame_id=i,
            face_valid=True,
            ear_left=0.33,
            ear_right=0.31,
            ear_avg=0.32,
            mar=0.12,
            blink_opening_velocity=2.9 if i % 25 == 0 else 0.0,
            facial_flushing_ratio=1.42,
            flushing_valid=True,
            head_pitch=-1.5,
            head_roll=0.8
        )
        calibrator.update(ff)

    assert calibrator.is_calibrated
    assert calibrator.progress == 1.0

    b = calibrator.get_baselines()
    assert b.is_calibrated
    assert abs(b.ear_baseline - 0.32) < 0.01
    assert abs(b.asymmetry_baseline - 0.02) < 0.01
    assert abs(b.blink_upstroke_baseline - 2.9) < 0.05
    assert abs(b.flushing_baseline - 1.42) < 0.02


def test_calibrator_excludes_gated_and_blink_frames():
    calibrator = PersonalBaselineCalibrator(calibration_duration_sec=2.0, min_samples=20, fps=30.0)

    # Feed frames with blink (EAR < 0.15) and IMU gated frames
    for i in range(30):
        t = i * 0.033
        # Frame with eyes shut
        ff_shut = FrameFeatures(
            timestamp=t,
            ear_avg=0.08,
            blink_detected=True,
            face_valid=True
        )
        calibrator.update(ff_shut)

        # Gated frame
        ff_gated = FrameFeatures(
            timestamp=t + 0.01,
            ear_avg=0.55,
            imu_gated=True,
            face_valid=False
        )
        calibrator.update(ff_gated)

    # None of the invalid frames should have been added to ear samples
    assert len(calibrator._ear_samples) == 0


def test_calibrator_apply_to_extractor_and_classifier():
    calibrator = PersonalBaselineCalibrator(calibration_duration_sec=1.0, min_samples=10, fps=10.0)
    for i in range(15):
        ff = FrameFeatures(
            timestamp=i * 0.1,
            face_valid=True,
            ear_left=0.35,
            ear_right=0.31,
            ear_avg=0.33,
            blink_opening_velocity=3.1,
            facial_flushing_ratio=1.48,
            flushing_valid=True
        )
        calibrator.update(ff)

    extractor = FeatureExtractor()
    classifier = ImpairmentClassifier()

    calibrator.apply_to(extractor, classifier)

    assert abs(extractor.sober_ear_baseline - 0.33) < 0.01
    assert abs(extractor.sober_blink_up_baseline - 3.1) < 0.01
    assert abs(extractor.baseline_flushing - 1.48) < 0.01
    assert abs(classifier.sober_ear - 0.33) < 0.01
    assert abs(classifier.sober_blink_up - 3.1) < 0.01
    assert abs(classifier.baseline_asymmetry - 0.04) < 0.01
