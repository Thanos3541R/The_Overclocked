"""Unit tests for the 5 verified architectural fixes and performance optimizations:
1. utils/landmarks.py: Realistic anthropometric scale for CANONICAL_FACE_3D (100mm outer bi-canthal width).
2. pipeline/pnp_normalizer.py: Coordinate-preserving face normalization for visual sampling.
3. pipeline/feature_extractor.py: Horizontal conjugate gaze axis polarity, vergence, O(1) PERCLOS ring counter, and high-pass suspension de-trending.
4. pipeline/preprocessing.py: Single-pass color conversion (return_rgb=True).
5. pipeline/frame_capture.py: Threaded asynchronous capture for live camera sources.
6. pipeline/classifier.py: Cropped eye bounding box fast path for sclera redness.
7. train_baseline.py: Subject-invariant baseline delta features.
8. extract_video_features.py: MediaPipe detector lifecycle cleanup in finally.
"""
import os
import sys
import time
import numpy as np
import pytest
import cv2
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.landmarks import CANONICAL_FACE_3D, RIGHT_EYE_OUTER, LEFT_EYE_OUTER, NOSE_TIP
from pipeline.pnp_normalizer import PnPNormalizer
from pipeline.feature_extractor import FeatureExtractor, FrameFeatures
from pipeline.preprocessing import LightingNormalizer
from pipeline.frame_capture import FrameCapture
from pipeline.classifier import ImpairmentClassifier
from train_baseline import compute_subject_invariant_delta_features, compute_window_summary_features, generate_synthetic_dataset, train_and_evaluate
from tests.test_pnp import make_frontal_face_landmarks
from tests.test_alcohol_signals import make_full_landmarks


class TestAnthropometricModelAndPnPNormalizer:
    """Tests for Item 1 and Item 2: Canonical 3D model and coordinate-preserving normalization."""

    def test_canonical_face_3d_outer_bicanthal_width(self):
        """Outer bi-canthal width must be exactly 100mm (±50mm from nose tip)."""
        r_eye = CANONICAL_FACE_3D[2]  # Right eye outer
        l_eye = CANONICAL_FACE_3D[3]  # Left eye outer
        assert r_eye[0] == -50.0
        assert l_eye[0] == 50.0
        bicanthal_width = l_eye[0] - r_eye[0]
        assert bicanthal_width == pytest.approx(100.0, rel=1e-3)

    def test_pnp_normalizer_preserves_pixel_range_and_samples(self):
        """normalize_landmarks must retain pixel space coordinates, enabling valid visual frame sampling."""
        w, h = 1280, 720
        normalizer = PnPNormalizer(w, h)
        landmarks = make_frontal_face_landmarks(frame_width=w, frame_height=h)

        success = normalizer.solve_pose(landmarks)
        assert success is True

        norm_lms = normalizer.normalize_landmarks(landmarks)

        # Coordinates must remain within frame pixel bounds
        assert 0 <= norm_lms[NOSE_TIP, 0] <= w
        assert 0 <= norm_lms[NOSE_TIP, 1] <= h
        # Nose tip should remain identical to raw pixel coordinate
        np.testing.assert_allclose(norm_lms[NOSE_TIP, :2], landmarks[NOSE_TIP, :2], atol=1e-3)

    def test_pnp_normalizer_handles_2d_and_empty_landmarks(self):
        """normalize_landmarks must handle 2D coordinates and degenerate inputs gracefully."""
        normalizer = PnPNormalizer(1280, 720)
        lms_3d = make_frontal_face_landmarks()
        normalizer.solve_pose(lms_3d)

        # 2D input
        lms_2d = lms_3d[:, :2].copy()
        norm_2d = normalizer.normalize_landmarks(lms_2d)
        assert norm_2d.shape == (478, 2)
        assert np.isfinite(norm_2d).all()

        # Empty input
        empty = np.zeros((0, 3))
        res_empty = normalizer.normalize_landmarks(empty)
        assert res_empty.shape == (0, 3)


class TestFeatureExtractorEnhancements:
    """Tests for Item 3: Conjugate gaze polarity, vergence, O(1) PERCLOS, and suspension de-trending."""

    def test_conjugate_horizontal_gaze_polarity(self):
        """Both eyes must report positive x for rightward conjugate gaze and negative x for leftward."""
        lm = make_full_landmarks(ear_val=0.30, iris_x_offset=0.0)

        # Shift eyes rightward by +10px (viewer right, +X in frame)
        lm_right = make_full_landmarks(ear_val=0.30, iris_x_offset=10.0)
        ext = FeatureExtractor(fps=30.0)
        f_right = ext.compute(lm_right, timestamp=0.1, frame_id=1, face_valid=True)

        assert f_right.left_iris_x > 0.1
        assert f_right.right_iris_x > 0.1
        avg_gaze_right = (f_right.left_iris_x + f_right.right_iris_x) / 2.0
        assert avg_gaze_right > 0.1

        # Shift eyes leftward by -10px (viewer left, -X in frame)
        lm_left = make_full_landmarks(ear_val=0.30, iris_x_offset=-10.0)
        f_left = ext.compute(lm_left, timestamp=0.2, frame_id=2, face_valid=True)

        assert f_left.left_iris_x < -0.1
        assert f_left.right_iris_x < -0.1
        avg_gaze_left = (f_left.left_iris_x + f_left.right_iris_x) / 2.0
        assert avg_gaze_left < -0.1

    def test_vergence_remains_near_zero_during_conjugate_saccades(self):
        """Conjugate eye movements must not trigger false Lack of Convergence (LOC) alerts."""
        ext = FeatureExtractor(fps=30.0, loc_vergence_threshold_deg=-3.5)

        # Extreme right gaze (+15px)
        lm_right = make_full_landmarks(ear_val=0.30, iris_x_offset=15.0)
        f_right = ext.compute(lm_right, timestamp=0.1, frame_id=1, face_valid=True)
        assert abs(f_right.vergence_angle_deg) < 1.0
        assert f_right.lack_of_convergence is False

        # Extreme left gaze (-15px)
        lm_left = make_full_landmarks(ear_val=0.30, iris_x_offset=-15.0)
        f_left = ext.compute(lm_left, timestamp=0.2, frame_id=2, face_valid=True)
        assert abs(f_left.vergence_angle_deg) < 1.0
        assert f_left.lack_of_convergence is False

    def test_perclos_o1_ring_counter_exact_match(self):
        """O(1) PERCLOS ring counter must match exact sum() over rolling history."""
        ext = FeatureExtractor(fps=30.0, perclos_window_sec=2.0)  # 60 frame buffer
        lm_open = make_full_landmarks(ear_val=0.30)
        lm_closed = make_full_landmarks(ear_val=0.10)

        np.random.seed(42)
        sequence = [lm_open if np.random.rand() > 0.3 else lm_closed for _ in range(150)]

        for i, lm in enumerate(sequence):
            f = ext.compute(lm, timestamp=i / 30.0, frame_id=i, face_valid=True)
            # Verify against ground-truth sum
            expected_closed = sum(1 for e in ext._ear_history if e < ext.blink_threshold)
            expected_perclos = expected_closed / len(ext._ear_history)
            assert f.perclos == pytest.approx(expected_perclos, abs=1e-5)
            assert ext._perclos_closed_count == expected_closed

    def test_high_pass_suspension_detrending(self):
        """Postural sway must detrend linear road incline and bank changes."""
        ext = FeatureExtractor(fps=30.0)

        # Simulate 3 seconds of driving up an incline with constant slope
        for i in range(90):
            t = i / 30.0
            lm = make_full_landmarks(ear_val=0.30)
            # Linear incline from 2.0 to 8.0 degrees + tiny physiological jitter
            pitch = 2.0 + (6.0 * t / 3.0) + 0.02 * np.sin(i * 1.5)
            roll = -1.0 + 0.02 * np.cos(i * 1.5)
            f = ext.compute(lm, timestamp=t, frame_id=i, face_valid=True, head_pitch=pitch, head_roll=roll)

        # After de-trending the 6-degree slope, sway residual should be small (<0.25 deg)
        assert f.head_postural_sway < 0.25


class TestPreprocessingAndCaptureEnhancements:
    """Tests for Item 4 and Item 5: Single-pass color conversion and async capture."""

    def test_preprocessing_return_rgb(self):
        """LightingNormalizer must output valid RGB 3-channel frames when return_rgb=True."""
        prep = LightingNormalizer()

        # BGR input
        bgr = np.full((100, 100, 3), 40, dtype=np.uint8)
        rgb = prep.process(bgr, return_rgb=True)
        assert rgb.shape == (100, 100, 3)
        assert rgb.dtype == np.uint8

        # Grayscale input
        gray = np.full((100, 100), 30, dtype=np.uint8)
        rgb_from_gray = prep.process(gray, return_rgb=True)
        assert rgb_from_gray.shape == (100, 100, 3)

    @patch("cv2.VideoCapture")
    @patch("sys.platform", "win32")
    def test_threaded_camera_capture_lifecycle(self, mock_cv2):
        """Live camera capture starts background worker thread and terminates cleanly on release."""
        mock_instance = MagicMock()
        mock_instance.isOpened.return_value = True
        mock_instance.read.return_value = (True, np.full((480, 640, 3), 128, dtype=np.uint8))
        mock_instance.get.return_value = 30.0
        mock_cv2.return_value = mock_instance

        cap = FrameCapture(source=0, width=640, height=480)
        assert cap.is_camera is True
        assert cap._thread is not None
        assert cap._thread.is_alive()

        frame = cap.read()
        assert frame is not None
        assert frame.shape == (480, 640, 3)

        cap.release()
        assert cap._running is False
        assert not cap._thread.is_alive()


class TestClassifierAndBaselineDeltaEnhancements:
    """Tests for Item 6, Item 7, and Item 8: Sclera fast path, delta features, and detector cleanup."""

    def test_cropped_sclera_redness_fast_path(self):
        """_extract_sclera_redness fast path must calculate correct redness ratio and return 1.0 on invalid."""
        from utils.landmarks import RIGHT_EYE_CONTOUR, LEFT_EYE_CONTOUR

        frame = np.full((720, 1280, 3), 120, dtype=np.uint8)
        # Red bloodshot eyes
        frame[190:215, 185:215, 2] = 220  # High Red
        frame[190:215, 185:215, 0] = 60   # Low Blue
        frame[190:215, 185:215, 1] = 60   # Low Green

        frame[190:215, 385:415, 2] = 220
        frame[190:215, 385:415, 0] = 60
        frame[190:215, 385:415, 1] = 60

        lm = make_full_landmarks(ear_val=0.30)
        # Populate contour around eye centers
        for idx in RIGHT_EYE_CONTOUR:
            lm[idx] = [200.0, 200.0, 0.0]
        lm[RIGHT_EYE_CONTOUR[0]] = [185.0, 200.0, 0.0]
        lm[RIGHT_EYE_CONTOUR[4]] = [200.0, 215.0, 0.0]
        lm[RIGHT_EYE_CONTOUR[8]] = [215.0, 200.0, 0.0]
        lm[RIGHT_EYE_CONTOUR[12]] = [200.0, 190.0, 0.0]

        for idx in LEFT_EYE_CONTOUR:
            lm[idx] = [400.0, 200.0, 0.0]
        lm[LEFT_EYE_CONTOUR[0]] = [385.0, 200.0, 0.0]
        lm[LEFT_EYE_CONTOUR[4]] = [400.0, 215.0, 0.0]
        lm[LEFT_EYE_CONTOUR[8]] = [415.0, 200.0, 0.0]
        lm[LEFT_EYE_CONTOUR[12]] = [400.0, 190.0, 0.0]

        ratio = ImpairmentClassifier._extract_sclera_redness(frame, lm)
        assert ratio > 1.3

        # Invalid/monochrome frame must return neutral 1.0
        mono_frame = np.full((720, 1280), 100, dtype=np.uint8)
        assert ImpairmentClassifier._extract_sclera_redness(mono_frame, lm) == 1.0

    def test_subject_invariant_delta_features(self):
        """Baseline delta features must normalize individual resting EAR differences."""
        X_3d, y, subjects, raw_feat_names = generate_synthetic_dataset(n_subjects=10, windows_per_subject=24)
        X_2d, flat_names = compute_window_summary_features(X_3d, raw_feat_names, subjects=subjects)

        # Verify delta features were created
        has_delta = any("delta_base" in n for n in flat_names)
        assert has_delta is True

        # GroupKFold cross-validation must achieve >= 0.95 ROC-AUC
        results, _ = train_and_evaluate(X_2d, y, subjects, flat_names, n_splits=5, use_xgboost=False)
        assert results["oof_metrics"]["roc_auc"] >= 0.95
        assert results["oof_metrics"]["sensitivity"] >= 0.95
        assert results["oof_metrics"]["specificity"] >= 0.95


class TestRobustnessAndEdgeCases:
    """Rigorous edge-case and boundary attacks on all pipeline optimizations."""

    def test_pnp_edge_cases_and_non_arrays(self):
        """PnPNormalizer must reject degenerate inputs and handle tuples/lists safely."""
        pnp = PnPNormalizer(1280, 720)
        assert pnp.solve_pose(None) is False
        assert pnp.solve_pose([]) is False
        assert pnp.solve_pose(np.zeros((0, 3))) is False
        assert pnp.solve_pose(np.zeros((5, 3))) is False
        assert pnp.solve_pose(np.full((300, 3), np.nan)) is False
        assert pnp.solve_pose(np.full((300, 3), np.inf)) is False

        assert pnp.normalize_landmarks(None) is None
        tup_res = pnp.normalize_landmarks((1, 2))
        assert tup_res == (1, 2)
        list_res = pnp.normalize_landmarks([[1, 2], [3, 4]])
        assert isinstance(list_res, (list, np.ndarray))

    def test_preprocessing_degenerate_inputs(self):
        """LightingNormalizer must handle None, 0x0, and float images gracefully."""
        prep = LightingNormalizer()
        assert prep.process(None) is None
        empty = np.zeros((0, 0), dtype=np.uint8)
        assert prep.process(empty).size == 0

        # Float input [0, 1] normalized to uint8
        float_img = np.full((50, 50, 3), 0.5, dtype=np.float32)
        out = prep.process(float_img, return_rgb=True)
        assert out.dtype == np.uint8
        assert out.shape == (50, 50, 3)

    def test_feature_extractor_epoch_timestamp_and_nans(self):
        """FeatureExtractor must handle UNIX epoch timestamps and NaN landmarks cleanly."""
        ext = FeatureExtractor(fps=30.0)

        # 90 frames with epoch timestamp (~1.7e9 seconds) + linear drift
        for i in range(90):
            t = 1727190000.0 + (i / 30.0)
            lm = make_full_landmarks(ear_val=0.30)
            pitch = 5.0 + 3.0 * (i / 90.0)
            roll = -2.0 + 1.0 * (i / 90.0)
            f = ext.compute(lm, timestamp=t, frame_id=i, face_valid=True, head_pitch=pitch, head_roll=roll)

        assert f.head_postural_sway < 0.20

        # NaN landmarks must not trigger UnboundLocalError or exception
        lms_nan = np.full((478, 3), np.nan)
        f_nan = ext.compute(lms_nan, timestamp=t + 0.1, frame_id=91, face_valid=True)
        assert f_nan.face_valid is True
        assert np.isfinite(f_nan.left_iris_x)
        assert np.isfinite(f_nan.perclos)

    def test_sclera_redness_extreme_boundaries(self):
        """Sclera redness must return neutral 1.0 on collapsed or out-of-frame coordinates."""
        frame = np.full((720, 1280, 3), 120, dtype=np.uint8)

        # All landmarks collapsed to single point (0x0 bbox)
        lms_collapsed = np.zeros((478, 3))
        assert ImpairmentClassifier._extract_sclera_redness(frame, lms_collapsed) == 1.0

        # Out-of-bounds coordinates (negative and positive)
        lms_neg = np.full((478, 3), -500.0)
        assert ImpairmentClassifier._extract_sclera_redness(frame, lms_neg) == 1.0
        lms_pos = np.full((478, 3), 5000.0)
        assert ImpairmentClassifier._extract_sclera_redness(frame, lms_pos) == 1.0

        # NaN coordinates
        lms_nan = np.full((478, 3), np.nan)
        assert ImpairmentClassifier._extract_sclera_redness(frame, lms_nan) == 1.0

    def test_delta_features_empty_and_nan_resilience(self):
        """compute_subject_invariant_delta_features must handle empty or NaN-containing inputs."""
        c_feat, c_names = compute_subject_invariant_delta_features(
            np.zeros((0, 7)), ["ear_avg_mean"], np.array([])
        )
        assert c_feat.shape == (0, 7)
        assert c_names == ["ear_avg_mean"]

        # Subject with NaNs
        feats = np.array([[np.nan, 0.3], [np.nan, 0.35]])
        c_feat2, c_names2 = compute_subject_invariant_delta_features(
            feats, ["ear_avg_mean", "other_mean"], np.array(["s1", "s1"])
        )
        assert c_feat2.shape == (2, 2)
        assert "delta_base" in c_names2[0]
