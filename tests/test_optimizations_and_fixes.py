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
import json
import threading
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
from pipeline.training_buffer import TrainingBuffer
from pipeline.head_pose_estimator import HeadPoseGate
from pipeline.imu_gate import SoftwareIMUGate
from pipeline.face_mesh_detector import FaceMeshDetector
from config import PipelineConfig
from train_baseline import (
    compute_subject_invariant_delta_features,
    compute_window_summary_features,
    generate_synthetic_dataset,
    train_and_evaluate,
    load_dataset_from_npz,
)
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

        # NaN landmarks must not trigger UnboundLocalError or exception and must be safely invalidated
        lms_nan = np.full((478, 3), np.nan)
        f_nan = ext.compute(lms_nan, timestamp=t + 0.1, frame_id=91, face_valid=True)
        assert f_nan.face_valid is False
        assert f_nan.gating_reason == "invalid_landmarks"
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


class TestAuditRemediation:
    """Tests for the 6 technical audit findings and non-regression invariants."""

    def test_pnp_custom_camera_intrinsics_and_json_load(self, tmp_path):
        """PnPNormalizer must support focal length, JSON/NPZ calibration files, and aspect scaling."""
        # 1. Custom focal length
        pnp_fl = PnPNormalizer(1280, 720, focal_length=1500.0)
        assert pnp_fl.camera_matrix[0, 0] == pytest.approx(1500.0)
        assert pnp_fl.camera_matrix[1, 1] == pytest.approx(1500.0)
        assert pnp_fl.camera_matrix[0, 2] == pytest.approx(640.0)
        assert pnp_fl.camera_matrix[1, 2] == pytest.approx(360.0)

        # 2. JSON calibration file loading
        calib_data = {
            "camera_matrix": [
                [1100.0, 0.0, 640.0],
                [0.0, 1100.0, 360.0],
                [0.0, 0.0, 1.0],
            ],
            "dist_coeffs": [0.01, -0.02, 0.0, 0.0, 0.0],
        }
        json_path = tmp_path / "camera_calib.json"
        with open(json_path, "w") as f:
            json.dump(calib_data, f)

        pnp_json = PnPNormalizer(1280, 720, calibration_path=str(json_path))
        assert pnp_json.camera_matrix[0, 0] == pytest.approx(1100.0)
        assert pnp_json.dist_coeffs[0, 0] == pytest.approx(0.01)

        # Classmethod constructor
        pnp_cls = PnPNormalizer.from_calibration_file(str(json_path), 1280, 720)
        assert pnp_cls.camera_matrix[0, 0] == pytest.approx(1100.0)

        # Scaling on frame size change
        pnp_json.update_frame_size(1920, 1080)
        assert pnp_json.camera_matrix[0, 0] == pytest.approx(1100.0 * 1.5)
        assert pnp_json.camera_matrix[1, 1] == pytest.approx(1100.0 * 1.5)
        assert pnp_json.camera_matrix[0, 2] == pytest.approx(640.0 * 1.5)
        assert pnp_json.camera_matrix[1, 2] == pytest.approx(360.0 * 1.5)

        # 3. NPZ calibration file loading
        npz_path = tmp_path / "camera_calib.npz"
        np.savez(
            npz_path,
            camera_matrix=np.array([[1200.0, 0.0, 640.0], [0.0, 1200.0, 360.0], [0.0, 0.0, 1.0]]),
            dist_coeffs=np.array([0.05, -0.01, 0.0, 0.0, 0.0]),
        )
        pnp_npz = PnPNormalizer(1280, 720, calibration_path=str(npz_path))
        assert pnp_npz.camera_matrix[0, 0] == pytest.approx(1200.0)
        assert pnp_npz.dist_coeffs[0, 0] == pytest.approx(0.05)

    def test_feature_extractor_configurable_thresholds_and_thread_safety(self):
        """FeatureExtractor must expose configurable thresholds matching config and be thread-safe."""
        # 1. Configurable threshold default values
        ext = FeatureExtractor()
        assert ext.deg_per_iris_unit == pytest.approx(15.0)
        assert ext.saccade_deg_per_unit == pytest.approx(22.0)
        assert ext.saccade_subpixel_deadzone_dx == pytest.approx(0.026)
        assert ext.saccade_subpixel_deadzone_hypot == pytest.approx(0.030)
        assert ext.saccade_head_motion_dx == pytest.approx(0.055)
        assert ext.eye_socket_vel_threshold == pytest.approx(0.75)
        assert ext.blink_phase_timeout_frames == 8
        assert ext.sway_yaw_gate_deg_s == pytest.approx(5.0)
        assert ext.sway_nod_reset_deg_s == pytest.approx(35.0)
        assert ext.rigid_head_sway_threshold == pytest.approx(0.12)

        # 2. Custom values override
        ext_custom = FeatureExtractor(
            saccade_deg_per_unit=30.0,
            eye_socket_vel_threshold=0.85,
            blink_phase_timeout_frames=20,
        )
        assert ext_custom.saccade_deg_per_unit == pytest.approx(30.0)
        assert ext_custom.eye_socket_vel_threshold == pytest.approx(0.85)
        assert ext_custom.blink_phase_timeout_frames == 20

        # 3. Thread-safety: RLock present and concurrent compute() calls succeed
        assert hasattr(ext, "_lock")
        lms = make_full_landmarks()
        errors = []

        def worker(thread_id: int):
            try:
                for i in range(25):
                    t = float(i) * 0.033 + float(thread_id)
                    res = ext.compute(lms, timestamp=t, frame_id=i, face_valid=True)
                    assert res is not None
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker, args=(k,)) for k in range(4)]
        for th in threads:
            th.start()
        for th in threads:
            th.join()

        assert len(errors) == 0, f"Concurrent execution errors: {errors}"

    def test_training_buffer_core_vs_extended_modes(self, tmp_path):
        """TrainingBuffer must default to 29 channels and support 48 channels in extended mode."""
        # 1. Channel counts
        tb_core = TrainingBuffer(subject_id="sub_core", window_size_sec=0.5, stride_sec=0.25, fps=30.0)
        assert tb_core.include_extended_features is False
        assert len(tb_core.feature_names) == 29
        assert tb_core.n_features == 29

        tb_ext = TrainingBuffer(subject_id="sub_ext", include_extended_features=True, window_size_sec=0.5, stride_sec=0.25, fps=30.0)
        assert tb_ext.include_extended_features is True
        assert len(tb_ext.feature_names) == 48
        assert tb_ext.n_features == 48

        # 2. Vector shapes and encoding
        f = FrameFeatures(timestamp=1.0, frame_id=1, face_valid=True, ear_avg=0.27, blink_detected=True)
        v_core = tb_core._features_to_vector(f)
        v_ext = tb_ext._features_to_vector(f)

        assert v_core.shape == (29,)
        assert v_ext.shape == (48,)
        assert v_core[2] == pytest.approx(0.27)
        assert v_ext[2] == pytest.approx(0.27)
        # blink_detected boolean flag
        assert v_core[22] == 1.0
        assert v_ext[33] == 1.0

        # 3. Populate samples and extract windows
        for i in range(30):
            sample = FrameFeatures(
                timestamp=float(i) * 0.033,
                frame_id=i,
                face_valid=True,
                ear_avg=0.28,
                head_yaw=1.0,
            )
            tb_core.add_sample(sample)
            tb_ext.add_sample(sample)

        wins_core = tb_core.extract_windows()
        wins_ext = tb_ext.extract_windows()
        assert len(wins_core) > 0
        assert len(wins_ext) > 0
        assert wins_core[0].features.shape[1] == 29
        assert wins_ext[0].features.shape[1] == 48

        # 4. Save and load mixed archives via load_dataset_from_npz
        p_core = tmp_path / "test_core.npz"
        p_ext = tmp_path / "test_ext.npz"
        tb_core.save_to_disk(p_core)
        tb_ext.save_to_disk(p_ext)

        X_3d, y, subs, fnames = load_dataset_from_npz([str(p_core), str(p_ext)])
        assert X_3d.shape[2] == 48  # Aligned union schema
        assert len(fnames) == 48
        assert len(subs) == len(wins_core) + len(wins_ext)


class TestExternalReviewFixes:
    """Verification tests for the 7 external review fixes and reliability upgrades."""

    def test_atomic_model_download_failure_handling(self, tmp_path):
        """Model downloader must clean up .tmp on failure and raise clear RuntimeError."""
        fake_model = tmp_path / "subdir" / "face_landmarker.task"
        tmp_target = str(fake_model) + ".tmp"

        # 1. Network failure raises RuntimeError and leaves no .tmp
        with patch("urllib.request.urlopen", side_effect=Exception("Connection refused")):
            with pytest.raises(RuntimeError) as exc_info:
                FaceMeshDetector._ensure_model_bundle(str(fake_model), timeout_sec=0.1)
            assert "offline/network download failed" in str(exc_info.value)
            assert not os.path.exists(tmp_target)
            assert not fake_model.exists()

        # 2. Truncated download (< min_bytes) raises RuntimeError and removes .tmp
        import io
        mock_resp = MagicMock()
        mock_resp.__enter__.return_value = io.BytesIO(b"short payload")
        with patch("urllib.request.urlopen", return_value=mock_resp):
            with pytest.raises(RuntimeError) as exc_info:
                FaceMeshDetector._ensure_model_bundle(str(fake_model), min_bytes=1000)
            assert "incomplete or corrupted" in str(exc_info.value) or "offline/network download failed" in str(exc_info.value)
            assert not os.path.exists(tmp_target)
            assert not fake_model.exists()

        # 3. Existing valid file bypasses network completely
        valid_model = tmp_path / "valid_face_landmarker.task"
        valid_model.write_bytes(b"X" * 1500)
        with patch("urllib.request.urlopen") as mock_url:
            FaceMeshDetector._ensure_model_bundle(str(valid_model), min_bytes=1000)
            mock_url.assert_not_called()

    def test_feature_extractor_none_and_all_zero_landmarks_safety(self):
        """FeatureExtractor must safely handle landmarks=None and all-zero landmarks without crashing,
        invalidating face_valid, setting gating_reason='invalid_landmarks', and holding last valid state."""
        ext = FeatureExtractor()

        # First pass a valid frame to establish baseline values
        lm_valid = make_full_landmarks(ear_val=0.28)
        f_valid = ext.compute(
            lm_valid, timestamp=1.0, frame_id=1, face_valid=True,
            head_yaw=12.0, head_pitch=5.0, head_roll=-3.0
        )
        assert f_valid.face_valid is True
        saved_ear = f_valid.ear_avg
        assert saved_ear > 0.0
        assert f_valid.head_yaw == 12.0
        assert f_valid.head_pitch == 5.0
        assert f_valid.head_roll == -3.0

        # Now test with landmarks=None
        f_none = ext.compute(
            landmarks=None, timestamp=1.033, frame_id=2, face_valid=True,
            head_yaw=0.0, head_pitch=0.0, head_roll=0.0
        )
        assert f_none.face_valid is False
        assert f_none.gating_reason == "invalid_landmarks"
        # Must hold last valid values
        assert f_none.ear_avg == pytest.approx(saved_ear)
        assert f_none.head_yaw == 12.0
        assert f_none.head_pitch == 5.0
        assert f_none.head_roll == -3.0

        # Now test with all-zeros landmarks
        lms_zero = np.zeros((478, 3), dtype=np.float64)
        f_zero = ext.compute(
            landmarks=lms_zero, timestamp=1.066, frame_id=3, face_valid=True,
            head_yaw=0.0, head_pitch=0.0, head_roll=0.0
        )
        assert f_zero.face_valid is False
        assert f_zero.gating_reason == "invalid_landmarks"
        assert f_zero.ear_avg == pytest.approx(saved_ear)
        assert f_zero.head_yaw == 12.0
        assert f_zero.head_pitch == 5.0
        assert f_zero.head_roll == -3.0

    def test_imu_gate_missing_tvec_does_not_corrupt_dt_or_false_freeze(self):
        """IMU gate must not corrupt dt or trigger false shock freezes after gaps in tracking."""
        gate = SoftwareIMUGate()
        t = 0.0
        dt = 0.033

        # 5 stationary frames
        for _ in range(5):
            res = gate.update(t, np.array([0.0, 0.0, 500.0]))
            assert res is False
            t += dt

        # 4 missing frames (tvec=None)
        for _ in range(4):
            res = gate.update(t, None)
            assert res is False
            t += dt

        # Gentle motion frame: should not false freeze
        res = gate.update(t, np.array([1.0, 0.0, 500.0]))
        assert res is False

    def test_head_pose_gate_holds_last_valid_euler_on_no_pose(self):
        """HeadPoseGate must hold last valid Euler angles when PnP loses tracking (euler=None)."""
        gate = HeadPoseGate()

        # Valid pose
        state1 = gate.check((5.0, 10.0, -2.0))
        assert state1.face_valid is True
        assert state1.pitch == 5.0
        assert state1.yaw == 10.0
        assert state1.roll == -2.0

        # Lost tracking (None)
        state2 = gate.check(None)
        assert state2.face_valid is False
        assert state2.gating_reason == "no_pose"
        assert state2.pitch == 5.0
        assert state2.yaw == 10.0
        assert state2.roll == -2.0

        # Reset clears held values
        gate.reset()
        state3 = gate.check(None)
        assert state3.face_valid is False
        assert state3.gating_reason == "no_pose"
        assert state3.pitch == 0.0
        assert state3.yaw == 0.0
        assert state3.roll == 0.0

    def test_pipeline_config_from_json(self, tmp_path):
        """PipelineConfig.from_json must load known fields, ignore unknown fields, and preserve defaults."""
        cfg_file = tmp_path / "test_config.json"
        data = {
            "camera_source": 2,
            "ear_blink_threshold": 0.22,
            "unexpected_future_key": True,
        }
        cfg_file.write_text(json.dumps(data))

        cfg = PipelineConfig.from_json(str(cfg_file))
        assert cfg.camera_source == 2
        assert cfg.ear_blink_threshold == 0.22
        # Check defaults are preserved
        assert cfg.frame_width == 1280
        assert cfg.frame_height == 720
        assert cfg.max_faces == 1

    def test_all_landmark_indices_within_mediapipe_bounds(self):
        """All defined landmark groups must contain valid integer indices within 0 <= idx < 478
        (and < 468 for non-iris facial landmarks)."""
        from utils.landmarks import (
            PNP_LANDMARK_INDICES,
            PNP_LANDMARK_INDICES_8,
            RIGHT_EYE_EAR_FLAT,
            LEFT_EYE_EAR_FLAT,
            RIGHT_EYE_CONTOUR,
            LEFT_EYE_CONTOUR,
            DYNAMIC_EYELID_INDICES,
            DYNAMIC_IRIS_INDICES,
            FOREHEAD_CONTOUR,
            FOREHEAD_PATCH,
            FULL_CHEEK_LANDMARKS,
            VASODILATION_LANDMARKS,
            EYEBROW_ARCH_LANDMARKS,
        )

        non_iris_groups = [
            PNP_LANDMARK_INDICES,
            PNP_LANDMARK_INDICES_8,
            RIGHT_EYE_EAR_FLAT,
            LEFT_EYE_EAR_FLAT,
            RIGHT_EYE_CONTOUR,
            LEFT_EYE_CONTOUR,
            DYNAMIC_EYELID_INDICES,
            FOREHEAD_CONTOUR,
            FOREHEAD_PATCH,
            FULL_CHEEK_LANDMARKS,
            VASODILATION_LANDMARKS,
            EYEBROW_ARCH_LANDMARKS,
        ]

        for group in non_iris_groups:
            for idx in group:
                assert isinstance(idx, (int, np.integer))
                assert 0 <= idx < 468, f"Index {idx} out of non-iris bounds [0, 468)"

        for idx in DYNAMIC_IRIS_INDICES:
            assert isinstance(idx, (int, np.integer))
            assert 0 <= idx < 478, f"Index {idx} out of bounds [0, 478)"
            assert 468 <= idx < 478, f"Iris index {idx} not in range [468, 478)"

    def test_main_cli_overlay_mode_and_mesh_style_resolution(self):
        """CLI arguments for overlay mode and mesh style must resolve cleanly without SystemExit."""
        import main

        test_cases = [
            ([], "regions"),
            (["--mesh-style", "off"], "off"),
            (["--mesh-style", "contours"], "regions"),
            (["--mesh-style", "solid"], "solid"),
            (["--overlay-mode", "full"], "full"),
            (["--overlay-mode", "solid"], "solid"),
            (["--no-mesh"], "off"),
        ]

        for cli_args, expected_mode in test_cases:
            with patch.object(sys, "argv", ["main.py"] + cli_args):
                args = main.parse_args()
                if args.no_mesh:
                    resolved = "off"
                elif getattr(args, "mesh_style", None) is not None:
                    resolved = "regions" if args.mesh_style in ("contours", "dots") else args.mesh_style
                else:
                    resolved = getattr(args, "overlay_mode", "regions")
                assert resolved == expected_mode

    def test_mediapipe_missing_import_guard(self):
        """FaceMeshDetector must raise actionable RuntimeError when mediapipe is None."""
        with patch("pipeline.face_mesh_detector.mp", None):
            with pytest.raises(RuntimeError) as exc_info:
                FaceMeshDetector()
            assert "MediaPipe is not installed or failed to import" in str(exc_info.value)

    def test_frame_capture_timestamp_property(self):
        """FrameCapture must maintain valid positive last_frame_timestamp on read."""
        with patch("cv2.VideoCapture") as mock_cap_cls:
            mock_inst = MagicMock()
            mock_inst.isOpened.return_value = True
            dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
            mock_inst.read.return_value = (True, dummy_frame)
            mock_inst.get.return_value = 30.0
            mock_cap_cls.return_value = mock_inst

            cap = FrameCapture(source="dummy_video.mp4")
            t0 = cap.last_frame_timestamp
            assert t0 > 0.0
            frame = cap.read()
            assert frame is not None
            assert cap.last_frame_timestamp >= t0
            cap.release()

    def test_extract_video_features_summary_and_report_new_micromotions(self, tmp_path):
        """extract_video_features output summary and text report include the 4 new micromotion fields."""
        from extract_video_features import extract_video_features

        summary = extract_video_features(
            "tests/sample_driver_video.mp4",
            output_dir=str(tmp_path),
            max_frames=5,
            headless=True,
        )

        sa = summary["section_a_involuntary_biomarkers"]
        sb = summary["section_b_voluntary_masking_biomarkers"]

        assert "mean_eyelid_tremor_hz" in sa
        assert "mean_rppg_pulse_bpm" in sa
        assert "mean_rppg_hrv_rmssd_ms" in sa
        assert "frontalis_compensation_frames" in sb

        # Check text report
        report_path = tmp_path / "video_features_report.txt"
        assert report_path.exists()
        report_text = report_path.read_text(encoding="utf-8")
        assert "Palpebral Eyelid Tremor" in report_text
        assert "T-Zone Autonomic Pulse (rPPG)" in report_text
        assert "Frontalis Brow Elevation" in report_text

    def test_backlight_face_roi_evaluation(self):
        """LightingNormalizer lifts dark face ROI despite bright windshield background."""
        prep = LightingNormalizer(low_light_threshold=60.0)
        # Create a 200x200 frame with bright background (240) and dark face ROI (25)
        frame = np.full((200, 200, 3), 240, dtype=np.uint8)
        face_bbox = (50, 50, 150, 150)
        frame[50:150, 50:150] = 25

        # Without face_bbox, full-frame mean lum is high (~186 > 95), so gamma lift is NOT triggered
        out_no_bbox = prep.process(frame.copy(), face_bbox=None)
        # With face_bbox, face ROI luminance (<60.0) is evaluated, triggering gamma lift on dark face
        out_with_bbox = prep.process(frame.copy(), face_bbox=face_bbox)

        # The face ROI in out_with_bbox must be brighter than in out_no_bbox due to gamma lift
        face_no_bbox_mean = out_no_bbox[50:150, 50:150].mean()
        face_with_bbox_mean = out_with_bbox[50:150, 50:150].mean()
        assert face_with_bbox_mean > face_no_bbox_mean

    def test_split_sunlight_bilateral_balancing(self):
        """Bilateral split-sunlight balancing equalizes face when |lum_left - lum_right| > 45."""
        prep = LightingNormalizer()
        frame = np.full((120, 160, 3), 100, dtype=np.uint8)
        # Face ROI from x=40 to 120, y=20 to 100
        # Left half of face (x=40..80) is dark (20) in shadow; right half (x=80..120) is sunlit (180)
        frame[20:100, 40:80] = 20
        frame[20:100, 80:120] = 180
        face_bbox = (40, 20, 120, 100)

        # Process with split-sunlight balancing
        out = prep.process(frame.copy(), face_bbox=face_bbox)

        # Check that the dark left side was lifted
        out_roi_left = out[20:100, 40:80].mean()
        orig_roi_left = frame[20:100, 40:80].mean()
        assert out_roi_left > orig_roi_left + 15.0

    def test_lighting_normalizer_face_bbox_none_and_degenerate(self):
        """LightingNormalizer preserves bit-for-bit invariance when face_bbox=None and handles degenerate bboxes."""
        prep = LightingNormalizer()
        bgr = np.random.randint(10, 240, (100, 100, 3), dtype=np.uint8)
        gray = np.random.randint(10, 240, (100, 100), dtype=np.uint8)

        # Exact bit-for-bit equality with face_bbox=None
        np.testing.assert_array_equal(prep.process(bgr), prep.process(bgr, face_bbox=None))
        np.testing.assert_array_equal(prep.process(gray), prep.process(gray, face_bbox=None))

        # Degenerate bboxes: out of bounds, inverted, tiny (<16px)
        for degen_bbox in [(-50, -50, -10, -10), (50, 50, 30, 30), (10, 10, 20, 20), (500, 500, 600, 600)]:
            out_degen_bgr = prep.process(bgr, face_bbox=degen_bbox)
            out_degen_gray = prep.process(gray, face_bbox=degen_bbox)
            assert out_degen_bgr.shape == bgr.shape
            assert out_degen_gray.shape == gray.shape

    def test_head_pose_gate_mounting_offsets(self):
        """HeadPoseGate with mounting offset evaluates relative angles without false gating."""
        from pipeline.head_pose_estimator import HeadPoseGate

        # Camera mounted at 20 deg yaw offset (e.g. A-pillar or center console)
        gate = HeadPoseGate(yaw_limit=35.0, yaw_offset=20.0, pitch_offset=-5.0)

        # Forward driver gaze: yaw=20.0, pitch=-5.0 from camera frame
        state = gate.check((-5.0, 20.0, 0.0))
        assert state.face_valid is True
        assert state.yaw == 20.0  # Raw reporting preserved
        assert state.pitch == -5.0

        # Looking left 30 deg relative to road: yaw = 20 + 30 = 50.0 deg
        state_left = gate.check((-5.0, 50.0, 0.0))
        assert state_left.face_valid is True  # rel_yaw = 30.0 <= 35.0

        # Looking right 40 deg relative to road: yaw = 20 - 40 = -20.0 deg
        state_right = gate.check((-5.0, -20.0, 0.0))
        assert state_right.face_valid is False  # rel_yaw = -40.0 > 35.0
        assert "yaw=-40.0° exceeds ±35.0°" in state_right.gating_reason

    def test_classifier_head_slump_relative_posture(self):
        """ImpairmentClassifier evaluates head slump relative to driver baseline posture."""
        from pipeline.classifier import ImpairmentClassifier

        clf = ImpairmentClassifier()
        dummy_frame = np.zeros((100, 100, 3), dtype=np.uint8)
        dummy_lms = np.zeros((478, 3))

        # Driver with -16 deg camera pitch (downward looking camera)
        # Without baseline calibration: pitch=-16.0 triggers false head slump
        res_uncal = clf.evaluate_static_image(
            frame=dummy_frame, landmarks=dummy_lms,
            ear_left=0.30, ear_right=0.30, mar=0.10,
            head_pitch=-16.0, head_roll=0.0,
        )
        assert res_uncal.marker_scores["head_slump"] > 0.0

        # With baseline calibration: pitch_baseline=-16.0 -> rel_pitch = 0.0
        clf.pitch_baseline = -16.0
        res_cal = clf.evaluate_static_image(
            frame=dummy_frame, landmarks=dummy_lms,
            ear_left=0.30, ear_right=0.30, mar=0.10,
            head_pitch=-16.0, head_roll=0.0,
        )
        assert res_cal.marker_scores["head_slump"] == 0.0

    def test_occlusion_survival_holdover_config(self):
        """PipelineConfig specifies max_occlusion_hold_frames to survive transient hand occlusions."""
        from config import PipelineConfig
        cfg = PipelineConfig()
        assert cfg.max_occlusion_hold_frames == 15


class TestPhase8Enhancements:
    """Unit tests for Phase 8: Eyeglass Glare, Sunglass Occlusion, and Rolling-Shutter Normalization."""

    @staticmethod
    def _make_landmarks_with_contours(ear_l=0.30, ear_r=0.30):
        from utils.landmarks import RIGHT_EYE_CONTOUR, LEFT_EYE_CONTOUR
        lm = np.zeros((478, 3), dtype=np.float64)

        # Right eye (subject's right, viewer's left) around (200, 200)
        cx_r, cy_r = 200.0, 200.0
        w_r, h_r = 40.0, ear_r * 40.0
        for i, idx in enumerate(RIGHT_EYE_CONTOUR):
            ang = 2.0 * np.pi * i / len(RIGHT_EYE_CONTOUR)
            lm[idx] = [cx_r + (w_r / 2.0) * np.cos(ang), cy_r + (h_r / 2.0) * np.sin(ang), 0.0]

        # Left eye (subject's left, viewer's right) around (400, 200)
        cx_l, cy_l = 400.0, 200.0
        w_l, h_l = 40.0, ear_l * 40.0
        for i, idx in enumerate(LEFT_EYE_CONTOUR):
            ang = 2.0 * np.pi * i / len(LEFT_EYE_CONTOUR)
            lm[idx] = [cx_l + (w_l / 2.0) * np.cos(ang), cy_l + (h_l / 2.0) * np.sin(ang), 0.0]

        # Explicit EAR 6-points for exact EAR Soukupová formula: EAR = (v1 + v2)/(2*h) = h_r/w_r = ear_r
        lm[33]  = [cx_r - w_r / 2, cy_r, 0]
        lm[160] = [cx_r - w_r * 0.15, cy_r - h_r / 2, 0]
        lm[158] = [cx_r + w_r * 0.15, cy_r - h_r / 2, 0]
        lm[133] = [cx_r + w_r / 2, cy_r, 0]
        lm[153] = [cx_r + w_r * 0.15, cy_r + h_r / 2, 0]
        lm[144] = [cx_r - w_r * 0.15, cy_r + h_r / 2, 0]

        lm[263] = [cx_l + w_l / 2, cy_l, 0]
        lm[385] = [cx_l + w_l * 0.15, cy_l - h_l / 2, 0]
        lm[387] = [cx_l - w_l * 0.15, cy_l - h_l / 2, 0]
        lm[362] = [cx_l - w_l / 2, cy_l, 0]
        lm[373] = [cx_l - w_l * 0.15, cy_l + h_l / 2, 0]
        lm[380] = [cx_l + w_l * 0.15, cy_l + h_l / 2, 0]

        # Irises
        lm[468] = [cx_r, cy_r, 0]
        lm[473] = [cx_l, cy_l, 0]

        # Nose bridge and tip (for rolling shutter)
        lm[168] = [300.0, 200.0, 0.0]  # Mid-nose bridge
        lm[4]   = [300.0, 280.0, 0.0]  # Nose tip
        return lm

    def test_uniform_synthetic_frames_no_glare_or_sunglasses(self):
        """Uniform synthetic test frames (std < 1.0) must never trigger glare or sunglasses."""
        ext = FeatureExtractor(fps=30.0)
        lm = self._make_landmarks_with_contours(0.30, 0.30)

        # Zero frame (black)
        f_black = np.zeros((720, 1280, 3), dtype=np.uint8)
        gl_l, gl_r, any_gl, sunglass = ext._evaluate_eye_glare_and_occlusion(f_black, lm, 0.30, 0.30)
        assert (gl_l, gl_r, any_gl, sunglass) == (False, False, False, False)

        feat_black = ext.compute(lm, timestamp=0.0, frame_id=0, face_valid=True, frame=f_black)
        assert feat_black.eyeglass_glare_detected is False
        assert feat_black.sunglasses_occluded is False

        # Gray frame (uniform 128)
        f_gray = np.full((720, 1280, 3), 128, dtype=np.uint8)
        gl_l, gl_r, any_gl, sunglass = ext._evaluate_eye_glare_and_occlusion(f_gray, lm, 0.30, 0.30)
        assert (gl_l, gl_r, any_gl, sunglass) == (False, False, False, False)

    def test_eyeglass_glare_monocular_fallback(self):
        """Specular reflection on one eyeglass lens triggers monocular fallback from clean eye."""
        ext = FeatureExtractor(fps=30.0)
        # Left eye: corrupted EAR 0.45; Right eye: clean EAR 0.25
        lm = self._make_landmarks_with_contours(ear_l=0.45, ear_r=0.25)
        # Offset irises to verify copying
        lm[468] = [200.0 - 8.0, 200.0 + 5.0, 0]  # Left eye in viewer frame (corners 33 & 133)
        lm[473] = [400.0 + 4.0, 200.0 - 2.0, 0]  # Right eye in viewer frame (corners 263 & 362)

        # Base frame with natural texture (std ~ 6.0)
        np.random.seed(42)
        frame = np.random.randint(90, 115, (720, 1280, 3), dtype=np.uint8)

        # Add specular glare (>242 lum) to left eye (subject's left, around 400, 200)
        # 18x18 patch gives ~31% contour glare (>25%) and crop mean ~136 (<210)
        frame[191:209, 391:409] = 255

        feat = ext.compute(lm, timestamp=0.0, frame_id=0, face_valid=True, frame=frame)
        assert feat.eyeglass_glare_detected is True
        # Monocular fallback: clean right eye (ear_right=0.25) used for ear_avg
        assert feat.ear_avg == pytest.approx(feat.ear_right, rel=1e-3)
        # Left iris coordinates copied from clean right iris
        assert feat.left_iris_x == pytest.approx(feat.right_iris_x, abs=1e-5)
        assert feat.left_iris_y == pytest.approx(feat.right_iris_y, abs=1e-5)

        # Opposite case: right eye has glare, left eye is clean
        ext.reset()
        frame_r = np.random.randint(90, 115, (720, 1280, 3), dtype=np.uint8)
        frame_r[191:209, 191:209] = 255  # Right eye contour around (200, 200)

        feat_r = ext.compute(lm, timestamp=0.0, frame_id=0, face_valid=True, frame=frame_r)
        assert feat_r.eyeglass_glare_detected is True
        # Monocular fallback: clean left eye used for ear_avg
        assert feat_r.ear_avg == pytest.approx(feat_r.ear_left, rel=1e-3)
        assert feat_r.right_iris_x == pytest.approx(feat_r.left_iris_x, abs=1e-5)
        assert feat_r.right_iris_y == pytest.approx(feat_r.left_iris_y, abs=1e-5)

    def test_sunglasses_occlusion_and_masking_suppression(self):
        """Dark sunglasses with static landmarks suppress false deliberate blink suppression and stare."""
        ext = FeatureExtractor(fps=30.0)
        lm = self._make_landmarks_with_contours(ear_l=0.30, ear_r=0.30)

        # Create color frame with dark eye crops (lum < 45, std in [1.0, 8.0))
        np.random.seed(123)
        frame = np.random.randint(22, 34, (720, 1280, 3), dtype=np.uint8)
        frame[:, :, 0] += 5  # Non-monochrome chromaticity

        # Feed 65 frames of static landmarks (inter-blink interval > 2.0s)
        feat = None
        for i in range(65):
            t = i / 30.0
            feat = ext.compute(lm, timestamp=t, frame_id=i, face_valid=True, frame=frame)

        assert feat is not None
        assert feat.sunglasses_occluded is True
        assert feat.deliberate_blink_suppression is False
        assert feat.compensatory_stare_active is False

    def test_rolling_shutter_vertical_midface_compensation(self):
        """Rolling shutter vertical elongation under vibration is canceled out by mid-face ratio."""
        ext = FeatureExtractor(fps=30.0)
        lm = self._make_landmarks_with_contours(ear_l=0.30, ear_r=0.30)
        # Distance nose: ||[300, 280] - [300, 200]|| = 80.0 px
        # Distance canthal: ||[420, 200] - [180, 200]|| = 240.0 px
        # Baseline ratio = 80 / 240 = 0.3333

        # Step 1: Calibrate baseline under low vibration (<= 0.15) over 30 frames
        for i in range(30):
            ext.compute(lm, timestamp=i / 30.0, frame_id=i, face_valid=True,
                        head_pitch=0.0, vibration_score=0.10)

        assert ext._baseline_nose_ratio == pytest.approx(80.0 / 240.0, rel=1e-3)

        # Step 2: High cabin vibration (> 0.25) with 10% rolling-shutter vertical stretch
        # Nose bridge stretches by 1.10x: 80 -> 88 px (tip moves from 280 to 288)
        # Eye landmarks also stretch vertically by 1.10x, inflating raw EAR to 0.33
        lm_stretched = self._make_landmarks_with_contours(ear_l=0.33, ear_r=0.33)
        lm_stretched[4] = [300.0, 288.0, 0.0]  # Nose length 88 px -> ratio 88/240 = 0.3667

        feat = ext.compute(lm_stretched, timestamp=1.5, frame_id=31, face_valid=True,
                           head_pitch=0.0, vibration_score=0.35)

        # Stretch ratio = 0.3667 / 0.3333 = 1.10.
        # EAR should be compensated by dividing by 1.10: 0.33 / 1.10 = 0.30
        assert feat.ear_avg == pytest.approx(0.30, abs=0.005)





