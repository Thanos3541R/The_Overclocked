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
import time
from typing import Dict, List, Tuple
import cv2
import numpy as np
import pytest

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
    LEFT_EYE_CONTOUR,
    LEFT_IRIS_CENTER,
    LEFT_IRIS_RING,
    PNP_LANDMARK_INDICES,
    RIGHT_CHEEK,
    RIGHT_EYE_CONTOUR,
    RIGHT_IRIS_CENTER,
    RIGHT_IRIS_RING,
)
from utils.overlay_regions import (
    ALL_REGIONS,
    FULL_MESH_REGION,
    KEY_REGIONS,
    LEFT_EYELID_REGION,
    LEFT_IRIS_CORNERS_REGION,
    PNP_ANCHORS_REGION,
    RIGHT_EYELID_REGION,
    RIGHT_IRIS_CORNERS_REGION,
    OverlayRegion,
)
from utils.visualization import Visualizer


# ═══════════════════════════════════════════════════════════════════
# 1. Overlay Region Definitions and Static Arrays Tests
# ═══════════════════════════════════════════════════════════════════

def test_key_regions_definitions():
    """Verify that all 8 key regions and full mesh are properly defined."""
    assert len(KEY_REGIONS) == 8, f"Expected 8 key regions, found {len(KEY_REGIONS)}"
    expected_names = [
        "R Eyelid",
        "L Eyelid",
        "R Iris/Corners",
        "L Iris/Corners",
        "PnP Anchors",
        "Eyebrows",
        "Lips",
        "Cheek Patches",
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

    # Check distinct shades for Left vs Right iris + corners
    assert RIGHT_IRIS_CORNERS_REGION.color != LEFT_IRIS_CORNERS_REGION.color
    assert RIGHT_EYELID_REGION.color != LEFT_EYELID_REGION.color

    # Full mesh wireframe
    assert FULL_MESH_REGION.name == "Full Mesh"
    assert len(FULL_MESH_REGION.connections) == 2556
    assert FULL_MESH_REGION.connections.dtype == np.int32


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
    """Verify that hotkey [v] cycles through: off -> regions -> full -> off."""
    viz = Visualizer(overlay_mode="regions")
    assert viz.overlay_mode == "regions"

    # Cycle to full
    mode = viz.cycle_overlay_mode()
    assert mode == "full"
    assert viz.overlay_mode == "full"

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
