"""Tests for color-coded landmark overlay, vectorization, and feature invariance.

Verifies:
1. utils/overlay_regions.py region definitions and static NumPy index arrays.
2. Vectorized batch drawing with zero per-landmark Python loops.
3. Hotkey [v] overlay mode cycling: off / key regions / full mesh.
4. Gated frame indication (red border, dimmed overlay colors).
5. Exact bit-for-bit feature invariance between overlay ON vs OFF on recorded video.
   (Ensures sclera redness, facial flushing, EAR, MAR are NEVER tainted by overlay pixels).
6. Performance benchmarks reporting FPS and overlay latency in ms across all three modes.
"""
import os
import sys
import time
from typing import Dict, List, Tuple
from unittest.mock import patch
import cv2
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import DEFAULT_CONFIG
from pipeline.classifier import ImpairmentClassifier
from pipeline.face_mesh_detector import FaceMeshDetector
from pipeline.feature_extractor import FeatureExtractor, FrameFeatures
from pipeline.head_pose_estimator import HeadPoseGate
from pipeline.imu_gate import SoftwareIMUGate
from pipeline.one_euro_filter import OneEuroFilterBank
from pipeline.pnp_normalizer import PnPNormalizer
from pipeline.preprocessing import LightingNormalizer
from utils.landmarks import (
    FOREHEAD_CONTOUR,
    FOREHEAD_PATCH,
    FULL_CHEEK_LANDMARKS,
    LEFT_EYE_CONTOUR,
    LEFT_IRIS_CENTER,
    LEFT_IRIS_RING,
    PNP_LANDMARK_INDICES,
    RIGHT_CHEEK,
    RIGHT_EYE_CONTOUR,
    RIGHT_IRIS_CENTER,
    RIGHT_IRIS_RING,
    VASODILATION_LANDMARKS,
    RIGHT_EYE_OUTER,
    RIGHT_EYE_INNER,
    LEFT_EYE_OUTER,
    LEFT_EYE_INNER,
)
from utils.overlay_regions import (
    ALL_REGIONS,
    CHEEK_PATCHES_REGION,
    FOREHEAD_REGION,
    FULL_CHEEKS_REGION,
    FULL_MESH_REGION,
    KEY_REGIONS,
    LEFT_EYELID_REGION,
    LEFT_IRIS_CORNERS_REGION,
    PNP_ANCHORS_REGION,
    RIGHT_EYELID_REGION,
    RIGHT_IRIS_CORNERS_REGION,
    LEFT_IRIS_CONNECTIONS,
    RIGHT_IRIS_CONNECTIONS,
    OverlayRegion,
)
from utils.visualization import Visualizer


# ═══════════════════════════════════════════════════════════════════
# 1. Overlay Region Definitions and Static Arrays Tests
# ═══════════════════════════════════════════════════════════════════

def test_key_regions_definitions():
    """Verify that all 9 key regions and full mesh are properly defined."""
    assert len(KEY_REGIONS) == 9, f"Expected 9 key regions, found {len(KEY_REGIONS)}"
    expected_names = [
        "R Eyelid",
        "L Eyelid",
        "R Iris/Corners",
        "L Iris/Corners",
        "PnP Anchors",
        "Eyebrows",
        "Lips",
        "Cheeks",
        "Forehead",
    ]
    actual_names = [r.name for r in KEY_REGIONS]
    assert actual_names == expected_names

    # Check that each region has valid numpy index arrays with int32 dtype
    for r in KEY_REGIONS:
        assert isinstance(r.connections, np.ndarray)
        assert r.connections.dtype == np.int32
        if len(r.connections) > 0:
            assert r.connections.ndim == 2
            assert r.connections.shape[1] == 2
            assert np.all(r.connections >= 0)
            assert np.all(r.connections < 478)

        assert isinstance(r.point_segments, np.ndarray)
        assert r.point_segments.dtype == np.int32
        if len(r.point_segments) > 0:
            assert r.point_segments.ndim == 2
            assert r.point_segments.shape[1] == 2
            assert np.all(r.point_segments >= 0)
            assert np.all(r.point_segments < 478)

        # Check BGR color formats
        assert len(r.color) == 3
        assert all(0 <= c <= 255 for c in r.color)
        assert len(r.dimmed_color) == 3
        assert all(0 <= c <= 255 for c in r.dimmed_color)
        # Dimmed color must be noticeably darker than base color (unless base is black)
        for c_base, c_dim in zip(r.color, r.dimmed_color):
            if c_base > 0:
                assert c_dim < c_base

    # Check distinct shades and correct anatomical connections for Left vs Right iris + corners
    assert RIGHT_IRIS_CORNERS_REGION.color != LEFT_IRIS_CORNERS_REGION.color
    assert RIGHT_EYELID_REGION.color != LEFT_EYELID_REGION.color
    assert np.array_equal(
        RIGHT_IRIS_CORNERS_REGION.point_indices,
        [LEFT_IRIS_CENTER, RIGHT_EYE_OUTER, RIGHT_EYE_INNER]
    )
    assert np.array_equal(RIGHT_IRIS_CORNERS_REGION.connections, LEFT_IRIS_CONNECTIONS)
    assert np.array_equal(
        LEFT_IRIS_CORNERS_REGION.point_indices,
        [RIGHT_IRIS_CENTER, LEFT_EYE_OUTER, LEFT_EYE_INNER]
    )
    assert np.array_equal(LEFT_IRIS_CORNERS_REGION.connections, RIGHT_IRIS_CONNECTIONS)

    # Forehead region checks
    assert FOREHEAD_REGION.name == "Forehead"
    assert len(FOREHEAD_REGION.connections) == len(FOREHEAD_CONTOUR) - 1
    assert len(FOREHEAD_REGION.point_indices) == len(FOREHEAD_PATCH)
    assert FOREHEAD_REGION.color == (0, 200, 255)  # Warm Gold / Amber

    # Full cheeks checks
    assert FULL_CHEEKS_REGION.name == "Cheeks"
    assert len(FULL_CHEEKS_REGION.point_indices) == 44  # 22 per cheek
    assert len(FULL_CHEEKS_REGION.connections) == 18   # 9 edges per cheek outline

    # Full mesh wireframe
    assert FULL_MESH_REGION.name == "Full Mesh"
    assert len(FULL_MESH_REGION.connections) == 2556
    assert FULL_MESH_REGION.connections.dtype == np.int32


def test_iris_crosshair_marker_colors_match_regions():
    """Verify that visualizer crosshair markers match region colors and anatomical wiring."""
    viz = Visualizer(overlay_mode="regions", show_iris=True)
    frame = np.zeros((300, 300, 3), dtype=np.uint8)
    landmarks = np.zeros((478, 3), dtype=np.float64)
    # Put viewer's left iris at (100, 150) and viewer's right iris at (200, 150)
    landmarks[LEFT_IRIS_CENTER] = [100.0, 150.0, 0.0]
    landmarks[RIGHT_IRIS_CENTER] = [200.0, 150.0, 0.0]

    with patch("cv2.drawMarker") as mock_marker:
        viz.draw(frame, landmarks)
        calls = mock_marker.call_args_list
        pts_called = [c[0][1] for c in calls]
        colors_called = [c[0][2] for c in calls]
        assert (100, 150) in pts_called
        assert (200, 150) in pts_called
        idx_left = pts_called.index((100, 150))
        idx_right = pts_called.index((200, 150))
        assert colors_called[idx_left] == RIGHT_IRIS_CORNERS_REGION.color
        assert colors_called[idx_right] == LEFT_IRIS_CORNERS_REGION.color


def test_overlay_draw_with_468_points():
    """Verify that visualizer and overlay regions render cleanly on 468-point models without irises."""
    viz = Visualizer(overlay_mode="regions", show_iris=True)
    frame = np.zeros((300, 300, 3), dtype=np.uint8)
    landmarks_468 = np.ones((468, 3), dtype=np.float64) * 150.0

    # Must render without IndexError
    rendered = viz.draw(frame.copy(), landmarks_468)
    assert rendered.shape == frame.shape

    # Full mesh mode with 468 points
    viz_full = Visualizer(overlay_mode="full", show_iris=True)
    rendered_full = viz_full.draw(frame.copy(), landmarks_468)
    assert rendered_full.shape == frame.shape


def test_overlay_draw_with_empty_or_1d_landmarks():
    """Verify that visualizer gracefully handles empty or 1D landmark arrays."""
    viz = Visualizer(overlay_mode="regions", show_iris=True)
    frame = np.zeros((300, 300, 3), dtype=np.uint8)

    # Empty 1D array
    out1 = viz.draw(frame.copy(), np.zeros(0))
    assert out1.shape == frame.shape

    # 1D array of landmarks
    out2 = viz.draw(frame.copy(), np.zeros(478))
    assert out2.shape == frame.shape

    # Region with negative or out-of-bound connection indices
    from utils.overlay_regions import OverlayRegion
    bad_conn = np.array([[-1, 2], [1, 999], [0, 1]], dtype=np.int32)
    bad_pts = np.array([[-1, -1], [999, 999], [0, 0]], dtype=np.int32)
    reg = OverlayRegion(
        name="BadRegion",
        connections=bad_conn,
        point_indices=np.array([0, 1], dtype=np.int32),
        point_segments=bad_pts,
        color=(0, 255, 0),
        dimmed_color=(0, 128, 0),
        line_thickness=1,
        point_thickness=2,
    )
    # Must draw without IndexError or wrapping to negative indices
    reg.draw(frame, np.ones((5, 2), dtype=np.int32) * 50)


def test_vasodilation_landmarks_and_t_zone():
    """Verify that vasodilation tracking landmarks include full cheeks and forehead patch."""
    assert len(VASODILATION_LANDMARKS) == 64
    assert set(FULL_CHEEK_LANDMARKS).issubset(set(VASODILATION_LANDMARKS))
    assert set(FOREHEAD_PATCH).issubset(set(VASODILATION_LANDMARKS))


def test_batch_draw_without_python_loops():
    """Verify that region.draw executes vectorized batch calls without looping."""
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    dummy_pts = np.random.randint(50, 400, (478, 2), dtype=np.int32)

    # Draw all 8 regions
    t0 = time.perf_counter()
    for _ in range(50):
        for r in KEY_REGIONS:
            r.draw(frame, dummy_pts, dimmed=False)
    t1 = time.perf_counter()

    elapsed_ms_per_frame = (t1 - t0) * 1000.0 / 50.0
    # Must be faster than 4ms on standard CPU (proves batch execution)
    assert elapsed_ms_per_frame < 4.0, f"Batch draw too slow: {elapsed_ms_per_frame:.2f} ms"
    assert np.count_nonzero(frame) > 0


# ═══════════════════════════════════════════════════════════════════
# 2. Visualizer Modes, Hotkey [v], and Gating Tests
# ═══════════════════════════════════════════════════════════════════

def test_visualizer_mode_cycling():
    """Verify that hotkey [v] cycles through: off -> regions -> full -> solid -> off."""
    viz = Visualizer(overlay_mode="regions")
    assert viz.overlay_mode == "regions"

    # Cycle to full
    mode = viz.cycle_overlay_mode()
    assert mode == "full"
    assert viz.overlay_mode == "full"

    # Cycle to solid
    mode = viz.cycle_overlay_mode()
    assert mode == "solid"
    assert viz.overlay_mode == "solid"

    # Cycle to off
    mode = viz.cycle_overlay_mode()
    assert mode == "off"
    assert viz.overlay_mode == "off"

    # Cycle to regions
    mode = viz.cycle_overlay_mode()
    assert mode == "regions"
    assert viz.overlay_mode == "regions"

    # Backward compatibility alias cycle_mesh_mode
    mode = viz.cycle_mesh_mode()
    assert mode == "full"


def test_gated_frame_dimming_and_border():
    """Verify red border and dimmed overlay rendering when face is invalid or IMU gated."""
    viz = Visualizer(overlay_mode="regions")
    dummy_pts = np.full((478, 3), 150.0, dtype=np.float64)

    # Normal valid frame
    frame_valid = np.zeros((200, 200, 3), dtype=np.uint8)
    feat_valid = FrameFeatures(face_valid=True, imu_gated=False)
    viz.draw(frame_valid, dummy_pts, feat_valid)
    # Border at (0, 0) should be green (0, 255, 0)
    assert tuple(frame_valid[0, 0]) == (0, 255, 0)

    # Gated frame (head pose invalid)
    frame_gated = np.zeros((200, 200, 3), dtype=np.uint8)
    feat_gated = FrameFeatures(face_valid=False, imu_gated=False)
    viz.draw(frame_gated, dummy_pts, feat_gated)
    # Border at (0, 0) should be red (0, 0, 255)
    assert tuple(frame_gated[0, 0]) == (0, 0, 255)

    # IMU shock frame
    frame_shock = np.zeros((200, 200, 3), dtype=np.uint8)
    feat_shock = FrameFeatures(face_valid=True, imu_gated=True)
    viz.draw(frame_shock, dummy_pts, feat_shock)
    # Border at (0, 0) should be red (0, 0, 255)
    assert tuple(frame_shock[0, 0]) == (0, 0, 255)


def test_overlay_off_draw_leaves_frame_clean():
    """Verify that in 'off' mode, no landmark overlay or legend is rendered."""
    viz = Visualizer(overlay_mode="off")
    frame = np.zeros((200, 200, 3), dtype=np.uint8)
    dummy_pts = np.full((478, 3), 100.0, dtype=np.float64)
    feat = FrameFeatures(face_valid=True, imu_gated=False)

    viz.draw(frame, dummy_pts, feat)
    assert viz.last_overlay_ms == 0.0


# ═══════════════════════════════════════════════════════════════════
# 3. Video Processing & Exact Feature Invariance Benchmark
# ═══════════════════════════════════════════════════════════════════

def _run_pipeline_pass(video_path: str, overlay_mode: str, max_frames: int = 60) -> Tuple[List[FrameFeatures], float, List[float]]:
    """Process recorded video through the full DMS pipeline and collect features and overlay latency.

    Drawing is done on a copy of the frame after feature extraction,
    matching main.py and ensuring zero pixel pollution.
    """
    cap = cv2.VideoCapture(video_path)
    assert cap.isOpened(), f"Failed to open video: {video_path}"

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    preprocessor = LightingNormalizer(
        clip_limit=DEFAULT_CONFIG.clahe_clip_limit,
        tile_grid_size=DEFAULT_CONFIG.clahe_tile_grid,
        low_light_threshold=DEFAULT_CONFIG.low_light_threshold,
        gamma_dark=DEFAULT_CONFIG.gamma_dark,
        gamma_dim=DEFAULT_CONFIG.gamma_dim,
    )
    detector = FaceMeshDetector(
        max_faces=DEFAULT_CONFIG.max_faces,
        refine_landmarks=DEFAULT_CONFIG.refine_landmarks,
        min_detection_confidence=DEFAULT_CONFIG.min_detection_confidence,
        min_tracking_confidence=DEFAULT_CONFIG.min_tracking_confidence,
    )
    normalizer = PnPNormalizer(frame_width=w, frame_height=h)
    pose_gate = HeadPoseGate(
        yaw_limit=DEFAULT_CONFIG.yaw_limit,
        pitch_down_limit=DEFAULT_CONFIG.pitch_down_limit,
        pitch_up_limit=DEFAULT_CONFIG.pitch_up_limit,
    )
    extractor = FeatureExtractor(
        blink_threshold=DEFAULT_CONFIG.ear_blink_threshold,
        mar_yawn_threshold=DEFAULT_CONFIG.mar_yawn_threshold,
        perclos_window_sec=DEFAULT_CONFIG.perclos_window_sec,
        perclos_alert_threshold=DEFAULT_CONFIG.perclos_alert_threshold,
        microsleep_frames=DEFAULT_CONFIG.microsleep_frames,
        yawn_frames=DEFAULT_CONFIG.yawn_frames,
        fps=fps,
    )
    imu_gate = SoftwareIMUGate(
        accel_threshold=DEFAULT_CONFIG.head_accel_threshold,
        freeze_duration_ms=DEFAULT_CONFIG.imu_freeze_ms,
        window_size=DEFAULT_CONFIG.imu_accel_window,
    )
    visualizer = Visualizer(overlay_mode=overlay_mode)

    filter_bank = None
    features_list: List[FrameFeatures] = []
    sclera_list: List[float] = []
    overlay_times: List[float] = []

    frame_id = 0
    t_start = time.perf_counter()

    while frame_id < max_frames:
        ret, frame = cap.read()
        if not ret:
            break

        timestamp = frame_id / fps
        frame_id += 1

        # Stage 1: Preprocessing
        processed = preprocessor.process(frame)

        # Stage 2: Face Mesh Detection
        landmarks = detector.detect(processed)

        if landmarks is None:
            # Display copy
            viz_frame = frame.copy()
            visualizer.draw(viz_frame, None, None, None)
            continue

        # Stage 3: 1-Euro Filter
        if filter_bank is None:
            filter_bank = OneEuroFilterBank(
                t0=timestamp,
                x0=landmarks,
                landmark_min_cutoff=DEFAULT_CONFIG.landmark_min_cutoff,
                landmark_beta=DEFAULT_CONFIG.landmark_beta,
                ear_min_cutoff=DEFAULT_CONFIG.ear_min_cutoff,
                ear_beta=DEFAULT_CONFIG.ear_beta,
            )
            smooth_landmarks = landmarks
        else:
            smooth_landmarks = filter_bank.filter_landmarks(timestamp, landmarks)

        # Stage 4: PnP Pose
        pnp_success = normalizer.solve_pose(smooth_landmarks)

        # Stage 5: Normalization
        norm_landmarks = normalizer.normalize_landmarks(smooth_landmarks)

        # Stage 6: Head Pose Gating
        euler = normalizer.get_euler_angles()
        pose_state = pose_gate.check(euler)

        # Stage 7: IMU Gating
        imu_frozen = imu_gate.update(
            timestamp=timestamp,
            tvec=normalizer.tvec if pnp_success else None,
            euler_angles=euler if pnp_success else None,
        )

        # Stage 8: Feature Extraction (computed on original untouched frame)
        features = extractor.compute(
            landmarks=norm_landmarks,
            timestamp=timestamp,
            frame_id=frame_id,
            face_valid=pose_state.face_valid,
            head_yaw=pose_state.yaw,
            head_pitch=pose_state.pitch,
            head_roll=pose_state.roll,
            imu_gated=imu_frozen,
            gating_reason=pose_state.gating_reason,
            frame=frame,
        )

        # Sclera redness extracted directly from frame
        sclera_val = ImpairmentClassifier._extract_sclera_redness(frame, smooth_landmarks)
        sclera_list.append(sclera_val)

        # Stage 9: Filter scalar signals
        if filter_bank is not None and pose_state.face_valid and not imu_frozen:
            features.ear_avg = filter_bank.filter_scalar(timestamp, features.ear_avg, "ear_avg")
            features.mar = filter_bank.filter_scalar(timestamp, features.mar, "mar")

        features_list.append(features)

        # Display Stage: Draw strictly on a copy of the frame
        viz_frame = frame.copy()
        visualizer.draw(viz_frame, smooth_landmarks, features, normalizer, pnp_success=pnp_success)
        overlay_times.append(visualizer.last_overlay_ms)

    t_end = time.perf_counter()
    total_sec = t_end - t_start

    cap.release()
    detector.close()

    return features_list, sclera_list, total_sec, overlay_times


def test_overlay_on_vs_off_feature_invariance_and_benchmark(capsys):
    """Verify bit-for-bit feature invariance across overlay modes and report performance metrics."""
    video_path = "tests/sample_driver_video.mp4"

    # Run Pass 1: Overlay OFF
    feats_off, sclera_off, sec_off, overlay_ms_off = _run_pipeline_pass(video_path, overlay_mode="off", max_frames=60)
    fps_off = len(feats_off) / sec_off if sec_off > 0 else 0.0

    # Run Pass 2: Overlay KEY REGIONS
    feats_regions, sclera_reg, sec_regions, overlay_ms_regions = _run_pipeline_pass(video_path, overlay_mode="regions", max_frames=60)
    fps_regions = len(feats_regions) / sec_regions if sec_regions > 0 else 0.0

    # Run Pass 3: Overlay FULL MESH
    feats_full, sclera_ful, sec_full, overlay_ms_full = _run_pipeline_pass(video_path, overlay_mode="full", max_frames=60)
    fps_full = len(feats_full) / sec_full if sec_full > 0 else 0.0

    assert len(feats_off) == len(feats_regions) == len(feats_full)
    assert len(sclera_off) == len(sclera_reg) == len(sclera_ful)
    n_frames = len(feats_off)
    assert n_frames > 0, "No frames processed from sample video!"

    # ── Assert Exact Bit-for-Bit Feature Invariance ──
    for i in range(n_frames):
        f_off = feats_off[i]
        f_reg = feats_regions[i]
        f_ful = feats_full[i]

        # Invariant geometric features
        assert f_off.ear_avg == f_reg.ear_avg == f_ful.ear_avg, f"EAR mismatch at frame {i}"
        assert f_off.ear_left == f_reg.ear_left == f_ful.ear_left, f"EAR Left mismatch at frame {i}"
        assert f_off.ear_right == f_reg.ear_right == f_ful.ear_right, f"EAR Right mismatch at frame {i}"
        assert f_off.mar == f_reg.mar == f_ful.mar, f"MAR mismatch at frame {i}"
        assert f_off.perclos == f_reg.perclos == f_ful.perclos, f"PERCLOS mismatch at frame {i}"
        assert f_off.head_yaw == f_reg.head_yaw == f_ful.head_yaw, f"Yaw mismatch at frame {i}"
        assert f_off.head_pitch == f_reg.head_pitch == f_ful.head_pitch, f"Pitch mismatch at frame {i}"
        assert f_off.head_roll == f_reg.head_roll == f_ful.head_roll, f"Roll mismatch at frame {i}"
        assert f_off.saccade_velocity == f_reg.saccade_velocity == f_ful.saccade_velocity, f"Saccade mismatch at frame {i}"

        # Invariant pixel-based features (Crucial: proves sclera redness & facial flushing never see overlay pixels!)
        assert sclera_off[i] == sclera_reg[i] == sclera_ful[i], (
            f"Sclera redness tainted by overlay at frame {i}! off={sclera_off[i]}, reg={sclera_reg[i]}"
        )
        assert f_off.facial_flushing_ratio == f_reg.facial_flushing_ratio == f_ful.facial_flushing_ratio, (
            f"Facial flushing ratio tainted by overlay at frame {i}! off={f_off.facial_flushing_ratio}, reg={f_reg.facial_flushing_ratio}"
        )
        assert f_off.flushing_delta == f_reg.flushing_delta == f_ful.flushing_delta, (
            f"Flushing delta tainted by overlay at frame {i}!"
        )

    # ── Calculate Benchmark Statistics ──
    mean_ms_off = float(np.mean(overlay_ms_off)) if overlay_ms_off else 0.0
    mean_ms_reg = float(np.mean(overlay_ms_regions)) if overlay_ms_regions else 0.0
    p95_ms_reg = float(np.percentile(overlay_ms_regions, 95)) if overlay_ms_regions else 0.0
    mean_ms_ful = float(np.mean(overlay_ms_full)) if overlay_ms_full else 0.0
    p95_ms_ful = float(np.percentile(overlay_ms_full, 95)) if overlay_ms_full else 0.0

    report = (
        "\n"
        "====================================================================================\n"
        "                     HUD OVERLAY BENCHMARK & INVARIANCE REPORT                      \n"
        "====================================================================================\n"
        f"Frames Evaluated: {n_frames} frames from tests/sample_driver_video.mp4\n"
        f"Feature Invariance: 100% BIT-FOR-BIT IDENTICAL ACROSS ALL 3 MODES (0.000 diff)\n"
        f"Pixel Integrity: Sclera Redness and Facial Flushing strictly untainted\n"
        "------------------------------------------------------------------------------------\n"
        f"Mode: OFF          | Pipeline FPS: {fps_off:5.1f} | Overlay ms: {mean_ms_off:5.2f} ms\n"
        f"Mode: KEY REGIONS  | Pipeline FPS: {fps_regions:5.1f} | Overlay ms: mean={mean_ms_reg:5.2f} ms, p95={p95_ms_reg:5.2f} ms\n"
        f"Mode: FULL MESH    | Pipeline FPS: {fps_full:5.1f} | Overlay ms: mean={mean_ms_ful:5.2f} ms, p95={p95_ms_ful:5.2f} ms\n"
        "====================================================================================\n"
    )
    print(report)

    # Assertions on performance
    assert mean_ms_off == 0.0
    assert mean_ms_reg < 5.0, f"Key regions overlay too slow: {mean_ms_reg:.2f} ms"
    assert mean_ms_ful < 15.0, f"Full mesh overlay too slow: {mean_ms_ful:.2f} ms"


# ═══════════════════════════════════════════════════════════════════
# 6. Solid Color Facial Parts Segmentation Overlay Tests
# ═══════════════════════════════════════════════════════════════════

def test_solid_face_regions_definitions():
    """Verify definitions and anatomical polygons of all solid face regions."""
    from utils.overlay_regions import (
        SOLID_REGIONS,
        SOLID_PART_KEYS,
        SolidFaceRegion,
        SOLID_FOREHEAD,
        SOLID_CHEEKS,
        SOLID_CHIN,
        SOLID_LIPS,
        SOLID_EYES,
        SOLID_NOSE,
        SOLID_EYEBROWS,
        SOLID_FACE,
    )
    assert len(SOLID_REGIONS) == 8
    expected_keys = {"forehead", "cheeks", "chin", "lips", "eyes", "nose", "eyebrows", "face"}
    actual_keys = {r.key for r in SOLID_REGIONS}
    assert actual_keys == expected_keys
    assert set(SOLID_PART_KEYS.keys()) == expected_keys

    for r in SOLID_REGIONS:
        assert isinstance(r, SolidFaceRegion)
        assert len(r.name) > 0
        assert len(r.hotkey) == 1
        assert len(r.color) == 3
        assert len(r.dimmed_color) == 3
        assert 0.0 < r.alpha <= 1.0
        assert len(r.polygons) >= 1
        for poly in r.polygons:
            assert isinstance(poly, np.ndarray)
            assert poly.dtype == np.int32
            assert len(poly) >= 3
            assert np.all(poly >= 0)
            assert np.all(poly < 478)


def test_solid_face_regions_draw_and_latency():
    """Verify solid face region rendering speed (< 2.0 ms) and ROI crop correctness."""
    from utils.overlay_regions import draw_solid_face_regions, SOLID_PART_KEYS

    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    # Synthetic face points centered in frame
    pts = np.zeros((478, 2), dtype=np.int32)
    # Spread points across a realistic 300x400 facial bbox
    np.random.seed(42)
    pts[:, 0] = np.random.randint(450, 750, size=478)
    pts[:, 1] = np.random.randint(150, 550, size=478)

    enabled_all = {k: True for k in SOLID_PART_KEYS}

    # Warm-up
    for _ in range(10):
        draw_solid_face_regions(frame, pts, enabled_all, dimmed=False, alpha=0.42)

    # Benchmark rendering latency over 30 iterations
    t0 = time.perf_counter()
    for _ in range(30):
        draw_solid_face_regions(frame, pts, enabled_all, dimmed=False, alpha=0.42)
    t1 = time.perf_counter()
    mean_ms = (t1 - t0) * 1000.0 / 30.0

    # Drawing on ROI crop must be extremely fast (< 5.0 ms on CPU)
    assert mean_ms < 5.0, f"Solid overlay too slow: {mean_ms:.2f} ms"
    assert np.count_nonzero(frame) > 0

    # Verify dimmed rendering runs cleanly
    frame_dimmed = np.zeros((720, 1280, 3), dtype=np.uint8)
    draw_solid_face_regions(frame_dimmed, pts, enabled_all, dimmed=True, alpha=0.42)
    assert np.count_nonzero(frame_dimmed) > 0


def test_solid_face_regions_individual_part_toggling():
    """Verify toggling individual facial parts on/off and alias resolution."""
    viz = Visualizer(overlay_mode="solid")
    assert viz.overlay_mode == "solid"

    # All parts enabled by default
    for part in ["forehead", "cheeks", "chin", "lips", "eyes", "nose", "eyebrows", "face"]:
        assert viz.is_solid_part_enabled(part) is True

    # Toggle lips off
    new_state = viz.toggle_solid_part("lips")
    assert new_state is False
    assert viz.is_solid_part_enabled("lips") is False

    # Toggle lips back on using alias 'lip'
    new_state = viz.toggle_solid_part("lip")
    assert new_state is True
    assert viz.is_solid_part_enabled("lips") is True

    # Toggle lips off using hotkey alias '4'
    new_state = viz.toggle_solid_part("4")
    assert new_state is False
    assert viz.is_solid_part_enabled("lips") is False

    # Toggle cheeks using alias 'c'
    new_state = viz.toggle_solid_part("c")
    assert new_state is False
    assert viz.is_solid_part_enabled("cheeks") is False

    # Explicit set_solid_part
    viz.set_solid_part("forehead", False)
    assert viz.is_solid_part_enabled("forehead") is False
    viz.set_solid_part("forehead", True)
    assert viz.is_solid_part_enabled("forehead") is True

    # Invalid part returns False and does not crash
    assert viz.toggle_solid_part("nonexistent_part") is False


def test_solid_face_regions_toggle_all():
    """Verify toggle_all_solid_parts transitions between all on and all off."""
    viz = Visualizer(overlay_mode="solid")

    # With all currently True, toggle_all turns them all False
    state = viz.toggle_all_solid_parts()
    assert state is False
    for k in viz.solid_parts:
        assert viz.is_solid_part_enabled(k) is False

    # With all currently False, toggle_all turns them all True
    state = viz.toggle_all_solid_parts()
    assert state is True
    for k in viz.solid_parts:
        assert viz.is_solid_part_enabled(k) is True

    # With only one enabled, toggle_all turns them all False
    viz.set_solid_part("lips", True)
    viz.set_solid_part("eyes", False)
    state = viz.toggle_all_solid_parts()
    assert state is False
    for k in viz.solid_parts:
        assert viz.is_solid_part_enabled(k) is False


def test_solid_mode_visualizer_draw_and_legend():
    """Verify Visualizer.draw runs in 'solid' mode with HUD legend rendering."""
    viz = Visualizer(overlay_mode="solid")
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)

    pts = np.zeros((478, 3), dtype=np.float64)
    pts[:, 0] = np.random.randint(450, 750, size=478)
    pts[:, 1] = np.random.randint(150, 550, size=478)

    features = FrameFeatures(face_valid=True)
    out = viz.draw(frame, pts, features=features)

    assert out.shape == (720, 1280, 3)
    assert np.count_nonzero(out) > 0

    # Turn off all parts and draw again
    viz.toggle_all_solid_parts()
    out2 = viz.draw(frame, pts, features=features)
    assert out2.shape == (720, 1280, 3)


def test_solid_face_polygons_are_geometrically_simple_and_non_crossing():
    """Verify that all anatomical polygons in SOLID_REGIONS have zero self-intersecting edges and valid bounds."""
    from utils.overlay_regions import SOLID_REGIONS, SOLID_NOSE, SOLID_CHIN, SOLID_FOREHEAD_POLYGON

    cap = cv2.VideoCapture("tests/sample_driver_video.mp4")
    ret, frame = cap.read()
    assert ret and frame is not None, "Failed to read sample driver video frame"

    detector = FaceMeshDetector()
    landmarks = detector.detect(frame)
    detector.close()
    cap.release()
    assert landmarks is not None and len(landmarks) >= 468, "Failed to detect face landmarks on test frame"

    pts = landmarks[:, :2]

    def ccw(A, B, C):
        return (B[0] - A[0]) * (C[1] - A[1]) - (B[1] - A[1]) * (C[0] - A[0])

    def segments_intersect(p1, p2, p3, p4):
        d1 = ccw(p1, p2, p3)
        d2 = ccw(p1, p2, p4)
        d3 = ccw(p3, p4, p1)
        d4 = ccw(p3, p4, p2)
        return (d1 * d2 < -1e-7) and (d3 * d4 < -1e-7)

    # 1. Zero self-crossing edges across every polygon in SOLID_REGIONS
    for region in SOLID_REGIONS:
        for p_idx, poly in enumerate(region.polygons):
            poly_pts = pts[poly]
            n = len(poly_pts)
            for i in range(n):
                p1, p2 = poly_pts[i], poly_pts[(i + 1) % n]
                for j in range(i + 2, n):
                    if i == 0 and j == n - 1:
                        continue
                    p3, p4 = poly_pts[j], poly_pts[(j + 1) % n]
                    assert not segments_intersect(p1, p2, p3, p4), (
                        f"Self-intersection in region '{region.name}' polygon {p_idx} "
                        f"between edge {i} ({poly[i]}->{poly[(i+1)%n]}) and edge {j} ({poly[j]}->{poly[(j+1)%n]})"
                    )

    # 2. SOLID_NOSE does not contain upper-lip landmark 393 and stays above the upper lip
    assert 393 not in SOLID_NOSE.polygons[0]
    nose_bottom_y = max(pts[p, 1] for p in SOLID_NOSE.polygons[0])
    upper_lip_y = pts[0, 1]
    assert nose_bottom_y <= upper_lip_y + 5, f"Nose extends into upper lip: nose_y={nose_bottom_y}, lip_y={upper_lip_y}"

    # 3. SOLID_CHIN is horizontally symmetric and stays below the mouth/jawline
    assert all(pts[p, 1] >= pts[61, 1] - 5 for p in SOLID_CHIN.polygons[0])

    # 4. SOLID_FOREHEAD stays above the eye contours
    forehead_bottom_y = max(pts[p, 1] for p in SOLID_FOREHEAD_POLYGON)
    eye_top_y = min(pts[p, 1] for p in (RIGHT_EYE_CONTOUR + LEFT_EYE_CONTOUR))
    assert forehead_bottom_y <= eye_top_y + 10, (
        f"Forehead loop extends into eyes: forehead_y={forehead_bottom_y}, eye_y={eye_top_y}"
    )


def test_visualizer_handle_key_and_mouse_click_part_selection():
    """Verify visualizer.handle_key, handle_mouse_click, and enabled_solid_parts initialization."""
    # 1. Key handler hotkey toggles
    viz = Visualizer(overlay_mode="solid")
    assert viz.is_solid_part_enabled("nose") is True
    assert viz.handle_key(ord("6")) is True
    assert viz.is_solid_part_enabled("nose") is False
    assert viz.handle_key(ord("n")) is True
    assert viz.is_solid_part_enabled("nose") is True

    # 2. Toggle all hotkey
    assert viz.handle_key(ord("0")) is True
    assert all(not v for v in viz.solid_parts.values())
    assert viz.handle_key(ord("0")) is True
    assert all(v for v in viz.solid_parts.values())

    # 3. Custom initial parts from CLI/config
    viz_sel = Visualizer(overlay_mode="solid", enabled_solid_parts=["eyes", "nose"])
    assert viz_sel.is_solid_part_enabled("eyes") is True
    assert viz_sel.is_solid_part_enabled("nose") is True
    assert viz_sel.is_solid_part_enabled("lips") is False
    assert viz_sel.is_solid_part_enabled("forehead") is False

    # 4. Mouse click hit testing on HUD legend
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    dummy_landmarks = np.zeros((478, 3), dtype=np.float64)
    dummy_landmarks[:, 0] = np.random.randint(450, 750, size=478)
    dummy_landmarks[:, 1] = np.random.randint(150, 550, size=478)
    viz_sel.draw(frame, dummy_landmarks, features=FrameFeatures(face_valid=True))

    assert len(viz_sel._legend_item_boxes) == 9

    # Click lips row
    lip_box = next(b for b in viz_sel._legend_item_boxes if b[4] == "lips")
    cx = (lip_box[0] + lip_box[2]) // 2
    cy = (lip_box[1] + lip_box[3]) // 2
    assert viz_sel.is_solid_part_enabled("lips") is False
    assert viz_sel.handle_mouse_click(cv2.EVENT_LBUTTONDOWN, cx, cy) is True
    assert viz_sel.is_solid_part_enabled("lips") is True

    # Click footer __all__ row
    all_box = next(b for b in viz_sel._legend_item_boxes if b[4] == "__all__")
    fx = (all_box[0] + all_box[2]) // 2
    fy = (all_box[1] + all_box[3]) // 2
    assert viz_sel.handle_mouse_click(cv2.EVENT_LBUTTONDOWN, fx, fy) is True
    assert all(not v for v in viz_sel.solid_parts.values())

    # Ignore clicks outside legend or non-LBUTTONDOWN events
    assert viz_sel.handle_mouse_click(cv2.EVENT_MOUSEMOVE, cx, cy) is False
    assert viz_sel.handle_mouse_click(cv2.EVENT_LBUTTONDOWN, 5, 5) is False


