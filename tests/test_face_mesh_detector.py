"""Unit tests for AdaptiveROITracker, Multi-Scale Pyramid Acquisition, and FaceMeshDetector.

Verifies:
  - AdaptiveROITracker bounding box computation and EMA smoothing (alpha=0.35).
  - Coordinate round-trip mapping accuracy (error < 1e-4 px).
  - Fallback behavior when crop has no face (immediate same-frame full-frame fallback).
  - Multi-scale quadrant acquisition ([0.2*W, 0.1*H] to [0.8*W, 0.85*H]).
  - FrameCapture Windows DirectShow negotiation and logging.
  - Configuration threshold updates (0.35 confidence, enable_adaptive_roi, roi_margin).
"""
import sys
import os
import cv2
import numpy as np
import pytest
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import DEFAULT_CONFIG, PipelineConfig
from pipeline.face_mesh_detector import AdaptiveROITracker, FaceMeshDetector
from pipeline.frame_capture import FrameCapture


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def create_synthetic_landmarks(cx: float = 640.0, cy: float = 360.0,
                               width: float = 120.0, height: float = 160.0,
                               n_points: int = 478) -> np.ndarray:
    """Generate synthetic landmarks centered at (cx, cy) with given width/height."""
    np.random.seed(42)
    lms = np.zeros((n_points, 3), dtype=np.float64)
    # Generate points distributed inside the bounding box
    half_w = width / 2.0
    half_h = height / 2.0
    lms[:, 0] = np.random.uniform(cx - half_w, cx + half_w, n_points)
    lms[:, 1] = np.random.uniform(cy - half_h, cy + half_h, n_points)
    lms[:, 2] = np.random.uniform(-10.0, 10.0, n_points)
    # Guarantee bounding box extremities
    lms[0, 0] = cx - half_w
    lms[1, 0] = cx + half_w
    lms[2, 1] = cy - half_h
    lms[3, 1] = cy + half_h
    return lms


# ─────────────────────────────────────────────────────────────────────────────
# 1. AdaptiveROITracker Bounding Box and EMA Smoothing Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestAdaptiveROITrackerBBoxAndEMA:
    """Verify bounding box calculation and EMA smoothing behavior."""

    def test_compute_bbox(self):
        """Bounding box should correctly capture min and max landmark extents."""
        lms = np.array([
            [100.0, 150.0, 0.0],
            [250.0, 200.0, 5.0],
            [180.0, 350.0, -2.0],
            [120.0, 180.0, 1.0],
        ], dtype=np.float64)
        x_min, y_min, x_max, y_max = AdaptiveROITracker.compute_bbox(lms)
        assert x_min == 100.0
        assert x_max == 250.0
        assert y_min == 150.0
        assert y_max == 350.0

    def test_compute_center_scale(self):
        """Center should be midpoints and scale should be max(width, height)."""
        lms = np.array([
            [100.0, 150.0, 0.0],
            [200.0, 350.0, 0.0],  # width = 100, height = 200
        ], dtype=np.float64)
        cx, cy, scale = AdaptiveROITracker.compute_center_scale(lms)
        assert cx == 150.0
        assert cy == 250.0
        assert scale == 200.0

    def test_initial_update_initializes_state(self):
        """First frame should directly initialize tracker state without smoothing."""
        tracker = AdaptiveROITracker(alpha=0.35, margin=2.0)
        assert not tracker.is_tracking
        assert tracker.cx is None

        lms = create_synthetic_landmarks(cx=500.0, cy=300.0, width=100.0, height=120.0)
        tracker.update(lms)

        assert tracker.is_tracking
        assert pytest.approx(tracker.cx, abs=1e-5) == 500.0
        assert pytest.approx(tracker.cy, abs=1e-5) == 300.0
        assert pytest.approx(tracker.scale, abs=1e-5) == 120.0

    def test_ema_smoothing_step_response(self):
        """Subsequent updates should smooth center and scale with alpha=0.35."""
        tracker = AdaptiveROITracker(alpha=0.35, margin=2.0)
        # Frame 0
        lms0 = create_synthetic_landmarks(cx=100.0, cy=100.0, width=60.0, height=80.0)
        tracker.update(lms0)
        assert tracker.cx == 100.0
        assert tracker.cy == 100.0
        assert tracker.scale == 80.0

        # Frame 1: Step to (200, 200, scale=120)
        lms1 = create_synthetic_landmarks(cx=200.0, cy=200.0, width=100.0, height=120.0)
        tracker.update(lms1)

        expected_cx = 0.35 * 200.0 + (1.0 - 0.35) * 100.0  # 135.0
        expected_cy = 0.35 * 200.0 + (1.0 - 0.35) * 100.0  # 135.0
        expected_scale = 0.35 * 120.0 + (1.0 - 0.35) * 80.0  # 94.0

        assert pytest.approx(tracker.cx, abs=1e-5) == expected_cx
        assert pytest.approx(tracker.cy, abs=1e-5) == expected_cy
        assert pytest.approx(tracker.scale, abs=1e-5) == expected_scale

    def test_ema_convergence(self):
        """Repeated updates at the target value should converge exponentially."""
        tracker = AdaptiveROITracker(alpha=0.35, margin=2.0)
        tracker.update(create_synthetic_landmarks(cx=100.0, cy=100.0, width=50.0, height=50.0))

        target_lms = create_synthetic_landmarks(cx=300.0, cy=400.0, width=150.0, height=150.0)
        for _ in range(30):
            tracker.update(target_lms)

        assert pytest.approx(tracker.cx, abs=1e-3) == 300.0
        assert pytest.approx(tracker.cy, abs=1e-3) == 400.0
        assert pytest.approx(tracker.scale, abs=1e-3) == 150.0

    def test_reset_clears_state(self):
        """Reset should clear tracking flag and center/scale variables."""
        tracker = AdaptiveROITracker()
        tracker.update(create_synthetic_landmarks())
        assert tracker.is_tracking

        tracker.reset()
        assert not tracker.is_tracking
        assert tracker.cx is None
        assert tracker.cy is None
        assert tracker.scale is None

    def test_crop_bounds_and_boundary_shifts(self):
        """Crop bounds should expand by margin=2.0 and shift smoothly at frame boundaries."""
        tracker = AdaptiveROITracker(alpha=0.35, margin=2.0)
        frame_shape = (720, 1280)

        # 1. Centered face (cx=640, cy=360, scale=100) -> crop_size = 200
        tracker.update(create_synthetic_landmarks(cx=640.0, cy=360.0, width=100.0, height=100.0))
        x1, y1, x2, y2 = tracker.get_crop_bounds(frame_shape)
        assert (x2 - x1) == 200
        assert (y2 - y1) == 200
        assert x1 == 540 and x2 == 740
        assert y1 == 260 and y2 == 460

        # 2. Left border face (cx=50, cy=360, scale=100) -> shifts right to [0, 200]
        tracker.reset()
        tracker.update(create_synthetic_landmarks(cx=50.0, cy=360.0, width=100.0, height=100.0))
        x1, y1, x2, y2 = tracker.get_crop_bounds(frame_shape)
        assert x1 == 0
        assert x2 == 200
        assert (x2 - x1) == 200

        # 3. Right border face (cx=1250, cy=360, scale=100) -> shifts left to [1080, 1280]
        tracker.reset()
        tracker.update(create_synthetic_landmarks(cx=1250.0, cy=360.0, width=100.0, height=100.0))
        x1, y1, x2, y2 = tracker.get_crop_bounds(frame_shape)
        assert x1 == 1080
        assert x2 == 1280
        assert (x2 - x1) == 200

    def test_crop_frame_extraction(self):
        """crop_frame should slice image matching the computed bounding box."""
        tracker = AdaptiveROITracker(alpha=0.35, margin=2.0)
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        tracker.update(create_synthetic_landmarks(cx=400.0, cy=300.0, width=80.0, height=80.0))

        crop, bounds = tracker.crop_frame(frame)
        x1, y1, x2, y2 = bounds
        assert crop.shape == (y2 - y1, x2 - x1, 3)
        assert (x2 - x1) == 160
        assert (y2 - y1) == 160

    def test_crop_bounds_strict_square_aspect_ratio_all_scales(self):
        """Crop bounds must maintain an EXACT square aspect ratio (width == height) for all scales."""
        tracker = AdaptiveROITracker(alpha=0.35, margin=2.0)
        frame_shape = (720, 1280)

        # Test various scales: small, medium, subpixel, large (exceeding frame height)
        test_cases = [
            (640.0, 360.0, 50.0),    # crop_size = 100
            (640.0, 360.2, 100.5),   # crop_size = 201 with fractional center
            (640.0, 360.0, 200.0),   # crop_size = 400
            (640.0, 360.0, 450.0),   # scale*margin = 900 > 720 (must clamp to 720x720)
            (640.0, 360.0, 800.0),   # scale*margin = 1600 > 720 (must clamp to 720x720)
            (10.0, 10.0, 500.0),     # corner with large scale
            (1270.0, 710.0, 500.0),  # opposite corner with large scale
        ]

        for cx, cy, scale in test_cases:
            tracker.reset()
            tracker.cx, tracker.cy, tracker.scale = cx, cy, scale
            tracker.is_tracking = True
            x1, y1, x2, y2 = tracker.get_crop_bounds(frame_shape)

            w_crop = x2 - x1
            h_crop = y2 - y1
            assert w_crop == h_crop, f"Failed square aspect ratio at scale {scale}: w={w_crop}, h={h_crop}"
            assert 0 <= x1 < x2 <= frame_shape[1], f"Crop x bounds [{x1}, {x2}] out of frame width {frame_shape[1]}"
            assert 0 <= y1 < y2 <= frame_shape[0], f"Crop y bounds [{y1}, {y2}] out of frame height {frame_shape[0]}"

    def test_crop_bounds_portrait_and_extreme_aspect_ratios(self):
        """Crop bounds must remain square in portrait (H > W) and square frame orientations."""
        tracker = AdaptiveROITracker(alpha=0.35, margin=2.0)

        # Portrait orientation: 1280x720 (height 1280, width 720)
        tracker.reset()
        tracker.cx, tracker.cy, tracker.scale = 360.0, 640.0, 450.0  # margin*scale = 900 > 720
        tracker.is_tracking = True
        x1, y1, x2, y2 = tracker.get_crop_bounds((1280, 720))
        assert (x2 - x1) == (y2 - y1) == 720
        assert 0 <= x1 < x2 <= 720
        assert 0 <= y1 < y2 <= 1280

        # Square frame: 512x512
        tracker.reset()
        tracker.cx, tracker.cy, tracker.scale = 256.0, 256.0, 300.0  # margin*scale = 600 > 512
        tracker.is_tracking = True
        x1, y1, x2, y2 = tracker.get_crop_bounds((512, 512))
        assert (x2 - x1) == (y2 - y1) == 512



# ─────────────────────────────────────────────────────────────────────────────
# 2. Coordinate Round-Trip Mapping Accuracy Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestCoordinateRoundTripMapping:
    """Verify coordinate remapping precision (error < 1e-4 px)."""

    def test_round_trip_accuracy_exact(self):
        """Mapping from global -> normalized -> global should have error < 1e-4 px."""
        lms_global = create_synthetic_landmarks(cx=600.0, cy=350.0, width=150.0, height=180.0)
        crop_box = (400, 200, 800, 600)  # W_crop = 400, H_crop = 400

        # Global -> Crop Normalized
        norm = AdaptiveROITracker.global_to_normalized(lms_global, crop_box)

        # Crop Normalized -> Global Remapped
        rec = AdaptiveROITracker.remap_to_global(norm, crop_box)

        max_err = np.max(np.abs(rec - lms_global))
        assert max_err < 1e-4, f"Round-trip mapping error too high: {max_err} px"
        assert max_err < 1e-12, f"Expected machine precision round-trip, got {max_err}"

    def test_single_point_round_trip(self):
        """1D single coordinate array should round-trip cleanly."""
        pt_global = np.array([456.789, 234.567, 12.345], dtype=np.float64)
        crop_box = (300, 100, 700, 500)

        norm = AdaptiveROITracker.global_to_normalized(pt_global, crop_box)
        rec = AdaptiveROITracker.remap_to_global(norm, crop_box)

        assert rec.shape == (3,)
        assert np.max(np.abs(rec - pt_global)) < 1e-4

    def test_remap_formula_explicit(self):
        """Verify the exact formulas:
        x_global = x_crop_min + x_norm * W_crop
        y_global = y_crop_min + y_norm * H_crop
        z_global = z_norm * W_crop
        """
        crop_box = (100, 50, 500, 450)  # x_crop_min=100, y_crop_min=50, W_crop=400, H_crop=400
        norm_lms = np.array([
            [0.25, 0.50, 0.10],
            [0.75, 0.20, -0.05],
        ], dtype=np.float64)

        remapped = AdaptiveROITracker.remap_to_global(norm_lms, crop_box)

        # Point 0:
        # x = 100 + 0.25 * 400 = 200.0
        # y = 50 + 0.50 * 400 = 250.0
        # z = 0.10 * 400 = 40.0
        assert pytest.approx(remapped[0, 0], abs=1e-5) == 200.0
        assert pytest.approx(remapped[0, 1], abs=1e-5) == 250.0
        assert pytest.approx(remapped[0, 2], abs=1e-5) == 40.0

        # Point 1:
        # x = 100 + 0.75 * 400 = 400.0
        # y = 50 + 0.20 * 400 = 130.0
        # z = -0.05 * 400 = -20.0
        assert pytest.approx(remapped[1, 0], abs=1e-5) == 400.0
        assert pytest.approx(remapped[1, 1], abs=1e-5) == 130.0
        assert pytest.approx(remapped[1, 2], abs=1e-5) == -20.0

    def test_non_square_crop_box(self):
        """Non-square crop boxes (e.g. boundary clamped) should scale width and height accurately."""
        crop_box = (50, 10, 350, 510)  # W_crop = 300, H_crop = 500
        lms = create_synthetic_landmarks(cx=200.0, cy=250.0, width=80.0, height=120.0)

        norm = AdaptiveROITracker.global_to_normalized(lms, crop_box)
        rec = AdaptiveROITracker.remap_to_global(norm, crop_box)

        assert np.max(np.abs(rec - lms)) < 1e-4


# ─────────────────────────────────────────────────────────────────────────────
# 3. Fallback Behavior Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestFallbackBehavior:
    """Verify detector fallback logic when crop detection fails."""

    def test_crop_failure_triggers_immediate_same_frame_full_frame_fallback(self):
        """If crop detection fails, detector must immediately fall back to full frame in same frame."""
        detector = FaceMeshDetector(enable_adaptive_roi=True)
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)

        # Seed tracking
        initial_lms = create_synthetic_landmarks(cx=640.0, cy=360.0, width=100.0, height=100.0)
        detector.roi_tracker.update(initial_lms)
        assert detector.roi_tracker.is_tracking

        calls = []

        def mock_detect_raw(img):
            calls.append(img.shape)
            # Fail on crop (shape != (720, 1280, 3)), succeed on full frame
            if img.shape[:2] == (720, 1280):
                # Return normalized landmarks for full frame
                norm = np.zeros((478, 3), dtype=np.float64)
                norm[:, 0] = 0.5
                norm[:, 1] = 0.5
                return norm
            return None

        with patch.object(detector, "_detect_raw", side_effect=mock_detect_raw):
            result = detector.detect(frame)

        # Both crop AND full-frame should have been evaluated in the SAME frame
        assert len(calls) == 2, f"Expected crop attempt then full frame fallback, got {len(calls)} calls"
        assert calls[0][:2] != (720, 1280), "First call should be crop"
        assert calls[1][:2] == (720, 1280), "Second call should be full frame fallback"

        # Result should be valid landmarks from full-frame
        assert result is not None
        assert result.shape == (478, 3)
        assert pytest.approx(result[0, 0], abs=1e-4) == 640.0
        assert pytest.approx(result[0, 1], abs=1e-4) == 360.0

        # Tracker should be re-initialized from full-frame landmarks
        assert detector.roi_tracker.is_tracking
        assert pytest.approx(detector.roi_tracker.cx, abs=1e-4) == 640.0

    def test_crop_and_full_frame_failure_falls_back_to_quadrant(self):
        """If crop and full frame both fail, detector falls back to quadrant crop."""
        detector = FaceMeshDetector(enable_adaptive_roi=True)
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)

        # Seed tracking
        detector.roi_tracker.update(create_synthetic_landmarks(cx=640.0, cy=360.0, width=100.0, height=100.0))

        calls = []
        q_box = detector.roi_tracker.get_driver_quadrant_bounds(frame.shape)
        q_h = q_box[3] - q_box[1]
        q_w = q_box[2] - q_box[0]

        def mock_detect_raw(img):
            calls.append(img.shape[:2])
            # Only succeed on quadrant crop
            if img.shape[:2] == (q_h, q_w):
                norm = np.zeros((478, 3), dtype=np.float64)
                norm[:, 0] = 0.5  # middle of quadrant
                norm[:, 1] = 0.5
                return norm
            return None

        with patch.object(detector, "_detect_raw", side_effect=mock_detect_raw):
            result = detector.detect(frame)

        assert len(calls) == 3  # crop, then full frame, then quadrant
        assert result is not None
        assert detector.roi_tracker.is_tracking

    def test_all_stages_failure_returns_none(self):
        """When all detection attempts fail, detect() returns None and tracker is not tracking."""
        detector = FaceMeshDetector(enable_adaptive_roi=True)
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        detector.roi_tracker.update(create_synthetic_landmarks())

        with patch.object(detector, "_detect_raw", return_value=None):
            result = detector.detect(frame)

        assert result is None
        assert not detector.roi_tracker.is_tracking


# ─────────────────────────────────────────────────────────────────────────────
# 4. Multi-Scale Pyramid Acquisition Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestMultiScaleQuadrantAcquisition:
    """Verify pyramid quadrant acquisition for distant drivers on frame 0 or when lost."""

    def test_driver_quadrant_bounds_dimensions(self):
        """Central driver quadrant must be [0.2*W, 0.1*H] to [0.8*W, 0.85*H]."""
        # 1280x720 frame
        x1, y1, x2, y2 = AdaptiveROITracker.get_driver_quadrant_bounds((720, 1280))
        assert x1 == int(round(0.2 * 1280)) == 256
        assert y1 == int(round(0.1 * 720)) == 72
        assert x2 == int(round(0.8 * 1280)) == 1024
        assert y2 == int(round(0.85 * 720)) == 612

        # 1920x1080 frame
        x1, y1, x2, y2 = AdaptiveROITracker.get_driver_quadrant_bounds((1080, 1920))
        assert x1 == 384
        assert y1 == 108
        assert x2 == 1536
        assert y2 == 918

    def test_frame_0_acquisition_via_quadrant_crop(self):
        """On frame 0, distant face failing full-frame detection is acquired via quadrant crop."""
        detector = FaceMeshDetector(enable_adaptive_roi=True)
        assert not detector.roi_tracker.is_tracking

        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        q_box = AdaptiveROITracker.get_driver_quadrant_bounds(frame.shape)
        qx1, qy1, qx2, qy2 = q_box
        qw = qx2 - qx1
        qh = qy2 - qy1

        calls = []

        def mock_detect_raw(img):
            calls.append(img.shape[:2])
            # Full frame fails (simulating distant face below 32px stride)
            if img.shape[:2] == (720, 1280):
                return None
            # Quadrant crop succeeds
            if img.shape[:2] == (qh, qw):
                norm = np.zeros((478, 3), dtype=np.float64)
                norm[:, 0] = 0.5  # centered in quadrant
                norm[:, 1] = 0.5
                return norm
            return None

        with patch.object(detector, "_detect_raw", side_effect=mock_detect_raw):
            result = detector.detect(frame)

        # Call sequence on frame 0: full frame -> quadrant crop
        assert calls == [(720, 1280), (qh, qw)]
        assert result is not None
        assert result.shape == (478, 3)

        # Expected global center: qx1 + 0.5 * qw, qy1 + 0.5 * qh
        expected_x = qx1 + 0.5 * qw
        expected_y = qy1 + 0.5 * qh
        assert pytest.approx(result[0, 0], abs=1e-4) == expected_x
        assert pytest.approx(result[0, 1], abs=1e-4) == expected_y

        # Tracker is now active for frame 1
        assert detector.roi_tracker.is_tracking
        assert pytest.approx(detector.roi_tracker.cx, abs=1e-4) == expected_x


# ─────────────────────────────────────────────────────────────────────────────
# 5. Configuration & Public API Contract Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestFaceMeshDetectorContract:
    """Verify configuration thresholds and public API contract."""

    def test_default_confidence_and_roi_config(self):
        """Config defaults must specify 0.35 confidence, enable_adaptive_roi=True, margin=2.0."""
        assert DEFAULT_CONFIG.min_detection_confidence == 0.35
        assert DEFAULT_CONFIG.min_tracking_confidence == 0.35
        assert DEFAULT_CONFIG.enable_adaptive_roi is True
        assert DEFAULT_CONFIG.roi_margin == 2.0

    def test_detector_default_parameters(self):
        """Detector initializes with updated defaults and ROI tracker enabled."""
        detector = FaceMeshDetector()
        assert detector.enable_adaptive_roi is True
        assert detector.roi_margin == 2.0
        assert detector.roi_tracker.alpha == 0.35
        assert detector.roi_tracker.margin == 2.0
        assert detector.n_landmarks == 478
        detector.close()

    def test_detect_returns_correct_shape_on_real_image(self):
        """detect() on real image must return (478, 3) numpy array with float64 dtype."""
        test_img_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "inspection_result.jpg"
        )
        if not os.path.exists(test_img_path):
            pytest.skip("inspection_result.jpg not present in repo root")

        img = cv2.imread(test_img_path)
        detector = FaceMeshDetector()
        landmarks = detector.detect(img)

        assert landmarks is not None
        assert isinstance(landmarks, np.ndarray)
        assert landmarks.shape == (478, 3)
        assert landmarks.dtype == np.float64

        # Subsequent call should use adaptive ROI tracking successfully
        assert detector.roi_tracker.is_tracking
        landmarks2 = detector.detect(img)
        assert landmarks2 is not None
        assert landmarks2.shape == (478, 3)
        detector.close()

    def test_detect_grayscale_frame(self):
        """Grayscale frames must be processed without error."""
        test_img_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "inspection_result.jpg"
        )
        if not os.path.exists(test_img_path):
            pytest.skip("inspection_result.jpg not present in repo root")

        bgr = cv2.imread(test_img_path)
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        detector = FaceMeshDetector()
        landmarks = detector.detect(gray)

        assert landmarks is not None
        assert landmarks.shape == (478, 3)
        detector.close()

    def test_detect_blank_image_returns_none(self):
        """Blank image should return None cleanly without error."""
        blank = np.zeros((480, 640, 3), dtype=np.uint8)
        detector = FaceMeshDetector()
        res = detector.detect(blank)
        assert res is None
        detector.close()

    def test_detect_invalid_or_empty_input(self):
        """Invalid or empty frames should return None safely."""
        detector = FaceMeshDetector()
        assert detector.detect(None) is None
        assert detector.detect(np.array([])) is None
        assert detector.detect(np.zeros((5, 5, 3), dtype=np.uint8)) is None
        detector.close()


# ─────────────────────────────────────────────────────────────────────────────
# 6. FrameCapture DirectShow & Logging Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestFrameCaptureEnhancements:
    """Verify FrameCapture camera instantiation, logging, and video playback."""

    def test_frame_capture_logging_and_properties(self, capsys):
        """FrameCapture logs [DMS Capture] Camera opened: {W}x{H} @ {fps} fps."""
        video_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "sample_driver_video.mp4"
        )
        if not os.path.exists(video_path):
            pytest.skip("sample_driver_video.mp4 not present")

        with FrameCapture(source=video_path) as cap:
            captured = capsys.readouterr()
            assert "[DMS Capture] Camera opened:" in captured.out
            assert f"{cap.frame_width}x{cap.frame_height}" in captured.out
            assert f"{cap.fps} fps" in captured.out

            frame = cap.read()
            assert frame is not None
            assert frame.shape[1] == cap.frame_width
            assert frame.shape[0] == cap.frame_height
            assert cap.frame_count == 1

    @patch("cv2.VideoCapture")
    @patch("sys.platform", "win32")
    def test_windows_dshow_and_fourcc_negotiation(self, mock_cv2_cap):
        """On Windows with camera source, VideoCapture must use cv2.CAP_DSHOW and MJPG fourcc."""
        mock_instance = MagicMock()
        mock_instance.isOpened.return_value = True
        mock_instance.read.return_value = (True, np.zeros((720, 1280, 3), dtype=np.uint8))
        mock_instance.get.return_value = 30.0
        mock_cv2_cap.return_value = mock_instance

        with FrameCapture(source=0, width=1280, height=720):
            pass

        # Verify cv2.VideoCapture was called with cv2.CAP_DSHOW
        mock_cv2_cap.assert_called_with(0, cv2.CAP_DSHOW)

        # Verify fourcc was negotiated before frame width/height
        expected_fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        mock_instance.set.assert_any_call(cv2.CAP_PROP_FOURCC, expected_fourcc)
        mock_instance.set.assert_any_call(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        mock_instance.set.assert_any_call(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    @patch("cv2.VideoCapture")
    @patch("sys.platform", "win32")
    def test_camera_string_source_does_not_call_pos_frames(self, mock_cv2_cap):
        """String camera index (e.g. '0') must not trigger CAP_PROP_POS_FRAMES on live capture."""
        mock_instance = MagicMock()
        mock_instance.isOpened.return_value = True
        mock_instance.read.return_value = (True, np.zeros((720, 1280, 3), dtype=np.uint8))
        mock_instance.get.return_value = 30.0
        mock_cv2_cap.return_value = mock_instance

        with FrameCapture(source="0", width=1280, height=720) as cap:
            assert cap.source == 0
            # Ensure CAP_PROP_POS_FRAMES was never set
            calls = [c.args for c in mock_instance.set.mock_calls]
            assert (cv2.CAP_PROP_POS_FRAMES, 0) not in calls

    @patch("cv2.VideoCapture")
    @patch("sys.platform", "win32")
    def test_windows_dshow_fallback_when_unsupported(self, mock_cv2_cap):
        """If cv2.CAP_DSHOW fails to open camera, fallback to default VideoCapture."""
        mock_dshow = MagicMock()
        mock_dshow.isOpened.return_value = False

        mock_default = MagicMock()
        mock_default.isOpened.return_value = True
        mock_default.read.return_value = (True, np.zeros((720, 1280, 3), dtype=np.uint8))
        mock_default.get.return_value = 30.0

        mock_cv2_cap.side_effect = [mock_dshow, mock_default]

        with FrameCapture(source=0, width=1280, height=720):
            pass

        assert mock_cv2_cap.call_count == 2
        mock_cv2_cap.assert_any_call(0, cv2.CAP_DSHOW)
        mock_cv2_cap.assert_any_call(0)


# ─────────────────────────────────────────────────────────────────────────────
# 7. Robustness & Extended Edge Case Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestTrackerRobustnessAndEdgeCases:
    """Verify tracker resilience against invalid inputs, NaNs, and multidimensional variations."""

    def test_tracker_parameter_validation(self):
        """AdaptiveROITracker must validate alpha in (0, 1] and margin >= 1.0."""
        with pytest.raises(ValueError, match="alpha must be in"):
            AdaptiveROITracker(alpha=0.0)
        with pytest.raises(ValueError, match="alpha must be in"):
            AdaptiveROITracker(alpha=-0.5)
        with pytest.raises(ValueError, match="alpha must be in"):
            AdaptiveROITracker(alpha=1.5)
        with pytest.raises(ValueError, match="margin must be >= 1.0"):
            AdaptiveROITracker(margin=0.5)

    def test_empty_or_small_landmarks_in_compute_bbox(self):
        """compute_bbox must handle empty or 1-column array safely without raising."""
        assert AdaptiveROITracker.compute_bbox(np.empty((0, 3))) == (0.0, 0.0, 0.0, 0.0)
        assert AdaptiveROITracker.compute_bbox(np.zeros((10, 1))) == (0.0, 0.0, 0.0, 0.0)
        assert AdaptiveROITracker.compute_bbox(None) == (0.0, 0.0, 0.0, 0.0)

    def test_nan_and_inf_landmarks_in_update_do_not_poison_tracker(self):
        """NaN or Inf coordinates must safely reset tracking without crashing subsequent crop calls."""
        tracker = AdaptiveROITracker(alpha=0.35, margin=2.0)
        tracker.update(create_synthetic_landmarks(cx=640, cy=360, width=100, height=100))
        assert tracker.is_tracking

        # Update with NaN coordinates
        bad_lms = np.array([[np.nan, 300.0, 0.0], [600.0, np.nan, 0.0]])
        tracker.update(bad_lms)

        assert not tracker.is_tracking
        assert tracker.cx is None
        # get_crop_bounds must return full frame bounds instead of raising ValueError: cannot convert float NaN to integer
        x1, y1, x2, y2 = tracker.get_crop_bounds((720, 1280))
        assert (x1, y1, x2, y2) == (0, 0, 1280, 720)

        # Update with Inf coordinates
        inf_lms = np.array([[np.inf, 300.0, 0.0], [600.0, -np.inf, 0.0]])
        tracker.update(inf_lms)
        assert not tracker.is_tracking

    def test_2d_landmarks_remapping_and_normalization(self):
        """Remap and normalization should support 2D landmarks without IndexError."""
        crop_box = (100, 50, 500, 450)  # W=400, H=400
        lms_2d = np.array([
            [200.0, 250.0],
            [400.0, 130.0],
        ], dtype=np.float64)

        norm = AdaptiveROITracker.global_to_normalized(lms_2d, crop_box)
        assert norm.shape == (2, 2)
        rec = AdaptiveROITracker.remap_to_global(norm, crop_box)
        assert rec.shape == (2, 2)
        assert np.max(np.abs(rec - lms_2d)) < 1e-4

    def test_4d_landmarks_preserves_auxiliary_attributes(self):
        """Remapping and normalization of (N, 4) landmarks must preserve auxiliary columns."""
        crop_box = (100, 50, 500, 450)
        lms_4d = np.array([
            [200.0, 250.0, 40.0, 0.95],
            [400.0, 130.0, -20.0, 0.88],
        ], dtype=np.float64)

        norm = AdaptiveROITracker.global_to_normalized(lms_4d, crop_box)
        assert norm.shape == (2, 4)
        assert pytest.approx(norm[0, 3]) == 0.95
        assert pytest.approx(norm[1, 3]) == 0.88

        rec = AdaptiveROITracker.remap_to_global(norm, crop_box)
        assert rec.shape == (2, 4)
        assert np.max(np.abs(rec[:, :3] - lms_4d[:, :3])) < 1e-4
        assert pytest.approx(rec[0, 3]) == 0.95
        assert pytest.approx(rec[1, 3]) == 0.88

    def test_empty_landmarks_remapping(self):
        """Empty landmarks array should return empty array safely."""
        crop_box = (0, 0, 100, 100)
        assert len(AdaptiveROITracker.remap_to_global(np.empty((0, 3)), crop_box)) == 0
        assert len(AdaptiveROITracker.global_to_normalized(np.empty((0, 3)), crop_box)) == 0

    def test_detect_1d_array_length_20(self):
        """1D array of length >= 10 should return None safely without IndexError."""
        detector = FaceMeshDetector()
        assert detector.detect(np.ones((20,))) is None
        assert detector.detect(np.zeros((50,))) is None
        detector.close()

    def test_refine_landmarks_false_returns_468_shape(self):
        """When refine_landmarks=False, detector returns (468, 3) matching n_landmarks."""
        test_img_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "inspection_result.jpg"
        )
        if not os.path.exists(test_img_path):
            pytest.skip("inspection_result.jpg not present")

        img = cv2.imread(test_img_path)
        detector = FaceMeshDetector(refine_landmarks=False)
        assert detector.n_landmarks == 468
        landmarks = detector.detect(img)
        assert landmarks is not None
        assert landmarks.shape == (468, 3)
        detector.close()

    def test_custom_roi_alpha_initialization(self):
        """Detector passes roi_alpha parameter to internal tracker."""
        detector = FaceMeshDetector(roi_alpha=0.6)
        assert detector.roi_tracker.alpha == 0.6
        detector.close()

    def test_detect_with_batch_or_4d_array_returns_none_safely(self):
        """4D batch array inputs must return None safely without OpenCV assertion crash."""
        detector = FaceMeshDetector()
        assert detector.detect(np.zeros((16, 720, 1280, 3), dtype=np.uint8)) is None
        detector.close()

    def test_detect_with_float_and_uint16_images(self):
        """Float32/uint16 images should be processed or return None safely without MediaPipe crash."""
        detector = FaceMeshDetector()
        assert detector.detect(np.zeros((100, 100, 3), dtype=np.float32)) is None
        assert detector.detect(np.zeros((100, 100, 3), dtype=np.uint16)) is None
        detector.close()

    def test_compute_bbox_all_nan_array(self):
        """All-NaN landmarks array returns (0, 0, 0, 0) without raising RuntimeWarning."""
        all_nan = np.array([[np.nan, np.nan], [np.nan, np.nan]], dtype=np.float64)
        assert AdaptiveROITracker.compute_bbox(all_nan) == (0.0, 0.0, 0.0, 0.0)

    def test_remap_none_and_global_to_normalized_none(self):
        """Passing None to remap_to_global and global_to_normalized returns None."""
        assert AdaptiveROITracker.remap_to_global(None, (0, 0, 100, 100)) is None
        assert AdaptiveROITracker.global_to_normalized(None, (0, 0, 100, 100)) is None

    def test_tracker_update_accepts_list_and_tuple(self):
        """update() must accept python sequences (lists/tuples) of coordinates without error."""
        tracker = AdaptiveROITracker(alpha=0.35, margin=2.0)
        tracker.update([[100.0, 200.0, 0.0], [200.0, 300.0, 0.0]])
        assert tracker.is_tracking
        assert tracker.cx == 150.0
        assert tracker.cy == 250.0

    @patch("cv2.VideoCapture")
    @patch("sys.platform", "win32")
    def test_camera_whitespace_string_source(self, mock_cv2_cap):
        """String camera index with whitespace (' 0 ') should be stripped and parsed as camera index."""
        mock_instance = MagicMock()
        mock_instance.isOpened.return_value = True
        mock_instance.read.return_value = (True, np.zeros((720, 1280, 3), dtype=np.uint8))
        mock_instance.get.return_value = 30.0
        mock_cv2_cap.return_value = mock_instance

        with FrameCapture(source=" 0 ", width=1280, height=720) as cap:
            assert cap.source == 0
            calls = [c.args for c in mock_instance.set.mock_calls]
            assert (cv2.CAP_PROP_POS_FRAMES, 0) not in calls


