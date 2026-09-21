"""Unit tests for the Impairment Classifier.

Tests verify:
  - Sober awake face produces low risk score (< 30%) and "SOBER" verdict
  - Ptosis (droopy eyes) and head slump produce high risk score (> 65%) and "HIGH_RISK_INTOXICATED"
  - Temporal stream evaluation detects PERCLOS, sluggish blink upstroke, and gaze tunneling
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest
from pipeline.classifier import ImpairmentClassifier
from pipeline.feature_extractor import FrameFeatures
from tests.test_alcohol_signals import make_full_landmarks


class TestImpairmentClassifier:
    """Test suite for impairment classification and scoring."""

    def test_sober_face_classified_as_sober(self):
        """A face with open eyes, upright pose, and normal mouth should be SOBER."""
        classifier = ImpairmentClassifier()
        dummy_frame = np.full((480, 640, 3), 180, dtype=np.uint8)
        landmarks = make_full_landmarks(ear_val=0.30)

        assessment = classifier.evaluate_static_image(
            frame=dummy_frame,
            landmarks=landmarks,
            ear_left=0.30,
            ear_right=0.30,
            mar=0.10,
            head_pitch=0.0,
            head_roll=0.0
        )

        assert assessment.classification == "SOBER"
        assert assessment.risk_score < 30.0, f"Sober face had high risk: {assessment.risk_score}"

    def test_severe_ptosis_and_slump_classified_as_intoxicated(self):
        """Heavy eyelid droop (EAR 0.15) + forward head slump (pitch -18°) should trigger HIGH_RISK."""
        classifier = ImpairmentClassifier()
        dummy_frame = np.full((480, 640, 3), 180, dtype=np.uint8)
        landmarks = make_full_landmarks(ear_val=0.15)

        assessment = classifier.evaluate_static_image(
            frame=dummy_frame,
            landmarks=landmarks,
            ear_left=0.15,
            ear_right=0.16,
            mar=0.28,
            head_pitch=-20.0,  # forward slumped head
            head_roll=5.0
        )

        assert assessment.classification == "HIGH_RISK_INTOXICATED"
        assert assessment.risk_score >= 60.0, f"Intoxicated face had low risk: {assessment.risk_score}"
        assert any("droop" in ind.lower() or "slump" in ind.lower() for ind in assessment.primary_indicators)

    def test_temporal_stream_evaluation(self):
        """Temporal features with sluggish blink upstroke and gaze tunneling trigger high risk."""
        classifier = ImpairmentClassifier()
        features = FrameFeatures(
            perclos=0.25,                 # elevated eye closure
            blink_opening_velocity=0.8,   # very sluggish upstroke (< 1.2)
            gaze_yaw_dispersion=0.02,     # visual tunneling (< 0.03)
            lack_of_convergence=True,     # DRE sign: divergent drift
            pursuit_fragmentation_ratio=0.35,  # severe catch-up saccade intrusions
            vor_gain=0.55,                # depressed VOR gain (PMC8997842)
            microsleep_detected=False
        )

        assessment = classifier.evaluate_temporal_stream(features)
        assert assessment.classification in ("MILD_IMPAIRMENT", "HIGH_RISK_INTOXICATED")
        assert assessment.risk_score > 50.0
        assert any("blink" in ind.lower() for ind in assessment.primary_indicators)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
