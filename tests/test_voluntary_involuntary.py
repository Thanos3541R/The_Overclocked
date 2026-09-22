"""Unit tests for Section A Involuntary, Section B Voluntary, and Anti-Masking Divergence.

Tests verify:
  1. Section A: Low-frequency postural sway (micro-tremor)
  2. Section A: Micro-vascular facial flushing (cheek chromaticity ratio)
  3. Section B: Voluntary eye-widening spikes (ptosis counter-effort)
  4. Section B: Deliberate blink suppression (prolonged inter-blink interval)
  5. Section B: Compensatory stare fixation duration
  6. Anti-Masking Divergence Metric (M_mask)
  7. Primary Lockout Decision anchored on Involuntary Evidence
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest
from pipeline.feature_extractor import FeatureExtractor, FrameFeatures
from pipeline.classifier import ImpairmentClassifier
from tests.test_alcohol_signals import make_full_landmarks
from utils.landmarks import RIGHT_CHEEK, LEFT_CHEEK


class TestVoluntaryInvoluntaryFeatures:
    """Test suite for voluntary vs involuntary split and anti-masking divergence."""

    def test_postural_sway_computation(self):
        """Postural sway should distinguish steady head from low-frequency cerebellar wobble."""
        ext = FeatureExtractor(fps=30.0)

        # 1. High sway / micro-tremor: pitch and roll wobbling
        for i in range(40):
            t = i / 30.0
            pitch = 2.5 * np.sin(2 * np.pi * 1.5 * t)  # 1.5 Hz wobble, amplitude 2.5 deg
            roll = 2.0 * np.cos(2 * np.pi * 1.5 * t)
            lm = make_full_landmarks(ear_val=0.30)
            feat_wobble = ext.compute(lm, timestamp=t, frame_id=i, face_valid=True,
                                      head_pitch=pitch, head_roll=roll)

        assert feat_wobble.head_postural_sway > 1.4, f"Expected sway > 1.4 deg, got {feat_wobble.head_postural_sway}"
        assert feat_wobble.rigid_head_stabilization is False

        # 2. Rigid head stabilization: conscious neck locking (pitch 0, roll 0)
        ext_rigid = FeatureExtractor(fps=30.0)
        for i in range(40):
            t = i / 30.0
            lm = make_full_landmarks(ear_val=0.30)
            feat_rigid = ext_rigid.compute(lm, timestamp=t, frame_id=i, face_valid=True,
                                           head_pitch=0.0, head_roll=0.0)

        assert feat_rigid.head_postural_sway < 0.10
        assert feat_rigid.rigid_head_stabilization is True

    def test_facial_flushing_ratio(self):
        """Facial flushing should detect elevated red chromaticity in cheek regions."""
        ext = FeatureExtractor(fps=30.0)
        lm = make_full_landmarks(ear_val=0.30)

        # Add mock cheek landmarks
        for idx in RIGHT_CHEEK:
            lm[idx] = [180, 280, 0]
        for idx in LEFT_CHEEK:
            lm[idx] = [420, 280, 0]

        # 1. Neutral frame (gray: equal B, G, R)
        frame_neutral = np.full((480, 640, 3), 128, dtype=np.uint8)
        feat_neutral = ext.compute(lm, timestamp=0.0, frame_id=0, face_valid=True, frame=frame_neutral)
        assert 0.95 <= feat_neutral.facial_flushing_ratio <= 1.05

        # 2. Flushed face (high red component on cheeks)
        frame_flushed = np.full((480, 640, 3), 100, dtype=np.uint8)
        # Boost Red channel (index 2 in BGR)
        frame_flushed[:, :, 2] = 180  # R=180, G=100, B=100 -> ratio = 180/100 = 1.8
        for i in range(10):
            t = (i + 1) / 30.0
            feat_flushed = ext.compute(lm, timestamp=t, frame_id=i+1, face_valid=True, frame=frame_flushed)

        assert feat_flushed.facial_flushing_ratio > 1.20, f"Expected flushing ratio > 1.20, got {feat_flushed.facial_flushing_ratio}"

    def test_voluntary_eye_widening_spike(self):
        """Surge in EAR after a droop episode indicates voluntary frontalis/levator contraction."""
        ext = FeatureExtractor(fps=30.0)

        # Phase 1: Driver has ptosis / droop (EAR = 0.18) for 2 seconds
        for i in range(60):
            t = i / 30.0
            lm = make_full_landmarks(ear_val=0.18)
            feat = ext.compute(lm, timestamp=t, frame_id=i, face_valid=True)
            assert feat.voluntary_eye_widening is False

        # Phase 2: Driver attempts to fight droop by forcing eyes wide open (EAR = 0.38)
        for i in range(60, 70):
            t = i / 30.0
            lm = make_full_landmarks(ear_val=0.38)
            feat = ext.compute(lm, timestamp=t, frame_id=i, face_valid=True)

        assert feat.voluntary_eye_widening is True, "Expected voluntary eye-widening spike after ptosis"

    def test_deliberate_blink_suppression(self):
        """Driver withholding blinks (>6 seconds with eyes open) triggers deliberate blink suppression."""
        ext = FeatureExtractor(fps=30.0)

        # Awake open eyes for 7 seconds without blinking
        for i in range(210):  # 210 frames / 30 fps = 7.0s
            t = i / 30.0
            lm = make_full_landmarks(ear_val=0.30)
            feat = ext.compute(lm, timestamp=t, frame_id=i, face_valid=True)

        assert feat.deliberate_blink_suppression is True

        # Now simulate a blink at t=7.1s (EAR drops below 0.20, ear_val=0.05 -> EAR=0.10)
        lm_blink = make_full_landmarks(ear_val=0.05)
        ext.compute(lm_blink, timestamp=7.1, frame_id=211, face_valid=True)

        # Eyes reopen at t=7.2s -> blink_detected is True, suppression counter resets
        lm_open = make_full_landmarks(ear_val=0.15)
        feat_after_blink = ext.compute(lm_open, timestamp=7.2, frame_id=212, face_valid=True)
        assert feat_after_blink.blink_detected is True
        assert feat_after_blink.deliberate_blink_suppression is False

    def test_compensatory_stare_fixation(self):
        """Unbroken gaze fixation without micro-saccades accumulates stare duration."""
        ext = FeatureExtractor(fps=30.0)

        # Hold central gaze steady for 4.0 seconds
        for i in range(120):  # 120 frames = 4.0s
            t = i / 30.0
            lm = make_full_landmarks(ear_val=0.30, iris_x_offset=0.0)
            feat = ext.compute(lm, timestamp=t, frame_id=i, face_valid=True)

        assert feat.stare_fixation_duration_s >= 3.5, f"Expected stare duration >= 3.5s, got {feat.stare_fixation_duration_s}"

        # Rapid saccade breaks fixation
        lm_jump = make_full_landmarks(ear_val=0.30, iris_x_offset=15.0)  # fast jump
        feat_broken = ext.compute(lm_jump, timestamp=4.05, frame_id=121, face_valid=True)
        assert feat_broken.stare_fixation_duration_s < 0.5

    def test_anti_masking_divergence_metric(self):
        """High voluntary effort paired with involuntary reflex decay triggers M_mask >= 50%."""
        ext = FeatureExtractor(fps=30.0)

        # Case 1: Driver masking intoxication:
        # Involuntary: GEN detected, sluggish blink upstroke, low VOR gain, LOC
        # Voluntary: Eye-widening spike, deliberate blink suppression, compensatory stare
        features_masking = FrameFeatures(
            gen_detected=True,
            vor_gain=0.60,
            blink_opening_velocity=0.9,
            pursuit_fragmentation_ratio=0.30,
            lack_of_convergence=True,
            voluntary_eye_widening=True,
            deliberate_blink_suppression=True,
            stare_fixation_duration_s=4.0,
            ear_avg=0.36
        )

        m_mask, detected = ext._compute_anti_masking_divergence(features_masking)
        assert m_mask >= 50.0, f"Expected M_mask >= 50%, got {m_mask}"
        assert detected is True

        # Case 2: Sober alert person
        features_sober = FrameFeatures(
            gen_detected=False,
            vor_gain=0.98,
            blink_opening_velocity=2.5,
            pursuit_fragmentation_ratio=0.02,
            lack_of_convergence=False,
            voluntary_eye_widening=False,
            deliberate_blink_suppression=False,
            stare_fixation_duration_s=1.0,
            ear_avg=0.30
        )
        m_mask_sober, detected_sober = ext._compute_anti_masking_divergence(features_sober)
        assert m_mask_sober == 0.0
        assert detected_sober is False

    def test_lockout_recommendation_anchoring(self):
        """Primary lockout is strictly anchored on Section A Involuntary evidence and Anti-Masking."""
        classifier = ImpairmentClassifier()

        # Scenario A: High Involuntary impairment (GEN + low VOR + slow blink) -> Lockout TRIGGERED
        feat_invol = FrameFeatures(
            gen_detected=True,
            pursuit_fragmentation_ratio=0.35,
            blink_opening_velocity=0.9,
            vor_gain=0.55,
            lack_of_convergence=True
        )
        assess_invol = classifier.evaluate_temporal_stream(feat_invol)
        assert assess_invol.lockout_recommended is True
        assert assess_invol.classification == "HIGH_RISK_INTOXICATED"
        assert assess_invol.involuntary_risk_score >= 50.0

        # Scenario B: Pure voluntary activity (e.g. driver looking surprised or staring) without involuntary failure -> NO Lockout
        feat_vol_only = FrameFeatures(
            voluntary_eye_widening=True,
            deliberate_blink_suppression=True,
            stare_fixation_duration_s=4.0,
            gen_detected=False,
            vor_gain=1.0,
            blink_opening_velocity=2.5,
            lack_of_convergence=False
        )
        assess_vol = classifier.evaluate_temporal_stream(feat_vol_only)
        assert assess_vol.lockout_recommended is False
        assert assess_vol.classification != "HIGH_RISK_INTOXICATED"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
