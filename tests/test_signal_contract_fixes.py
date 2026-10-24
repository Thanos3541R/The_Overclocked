"""Unit tests verifying signal processing, cross-module contracts, and training mask upgrades."""
import numpy as np
import pytest

from pipeline.feature_extractor import FeatureExtractor
from utils.landmarks import (
    LEFT_IRIS_CENTER,
    LEFT_IRIS_RING,
    RIGHT_IRIS_CENTER,
    RIGHT_IRIS_RING,
)


def test_downward_iris_shift_negative_y_and_depth_invariance():
    """(a) Downward iris shift (+y pixels) returns negative iris_y and is invariant to z depth."""
    ext = FeatureExtractor()

    # Landmarks array with 478 points
    lms_base = np.zeros((478, 3), dtype=np.float64)
    # Eye corners for right eye: outer 33, inner 133
    lms_base[33] = [100.0, 100.0, 0.0]
    lms_base[133] = [140.0, 100.0, 0.0]
    # Eye center at (120, 100)
    lms_base[RIGHT_IRIS_CENTER] = [120.0, 100.0, 0.0]

    rx0, ry0 = ext.compute_iris_position(lms_base, RIGHT_IRIS_CENTER, 33, 133)
    assert pytest.approx(ry0, abs=1e-5) == 0.0

    # Downward shift in image pixels (+Y in screen coords)
    lms_down = lms_base.copy()
    lms_down[RIGHT_IRIS_CENTER] = [120.0, 110.0, 0.0]
    rx_down, ry_down = ext.compute_iris_position(lms_down, RIGHT_IRIS_CENTER, 33, 133)
    assert ry_down < 0.0, f"Expected negative iris_y for downward gaze, got {ry_down}"

    # Upward shift in image pixels (-Y in screen coords)
    lms_up = lms_base.copy()
    lms_up[RIGHT_IRIS_CENTER] = [120.0, 90.0, 0.0]
    rx_up, ry_up = ext.compute_iris_position(lms_up, RIGHT_IRIS_CENTER, 33, 133)
    assert ry_up > 0.0, f"Expected positive iris_y for upward gaze, got {ry_up}"

    # Test depth (z-coordinate) invariance
    lms_z = lms_down.copy()
    lms_z[33, 2] = 50.0
    lms_z[133, 2] = -30.0
    lms_z[RIGHT_IRIS_CENTER, 2] = 100.0
    rx_z, ry_z = ext.compute_iris_position(lms_z, RIGHT_IRIS_CENTER, 33, 133)
    assert rx_z == pytest.approx(rx_down, abs=1e-6)
    assert ry_z == pytest.approx(ry_down, abs=1e-6)


def test_anatomical_iris_indices():
    """(b) RIGHT_IRIS_CENTER == 468 and LEFT_IRIS_CENTER == 473."""
    assert RIGHT_IRIS_CENTER == 468
    assert LEFT_IRIS_CENTER == 473
    assert RIGHT_IRIS_RING == [469, 470, 471, 472]
    assert LEFT_IRIS_RING == [474, 475, 476, 477]


def test_vergence_nasal_convergence_and_temporal_divergence():
    """(c) Nasal convergence returns positive vergence_angle_deg and temporal divergence returns negative (< -3.5 deg, lack_of_convergence=True)."""
    ext = FeatureExtractor(fps=30.0)

    # Base landmarks: viewer's left eye (subject right) at x=200, viewer's right eye (subject left) at x=400
    lms = np.zeros((478, 3), dtype=np.float64)
    # Subject RIGHT eye (viewer left): outer=33, inner=133, center=468
    lms[33] = [180.0, 200.0, 0.0]
    lms[133] = [220.0, 200.0, 0.0]
    lms[468] = [200.0, 200.0, 0.0]

    # Subject LEFT eye (viewer right): outer=263, inner=362, center=473
    lms[263] = [420.0, 200.0, 0.0]
    lms[362] = [380.0, 200.0, 0.0]
    lms[473] = [400.0, 200.0, 0.0]

    # Forward gaze: vergence ~ 0
    feat_forward = ext.compute(lms, timestamp=0.0, frame_id=0, face_valid=True)
    assert abs(feat_forward.vergence_angle_deg) < 2.0
    assert feat_forward.lack_of_convergence is False

    # Nasal convergence: eyes turn inward towards nose
    # Subject right eye (468) moves nasal/right (+x in pixels)
    # Subject left eye (473) moves nasal/left (-x in pixels)
    ext.reset()
    lms_converge = lms.copy()
    lms_converge[468, 0] += 8.0  # nasal inward
    lms_converge[473, 0] -= 8.0  # nasal inward
    feat_conv = ext.compute(lms_converge, timestamp=0.1, frame_id=1, face_valid=True)
    assert feat_conv.vergence_angle_deg > 0.0, f"Expected positive vergence for convergence, got {feat_conv.vergence_angle_deg}"

    # Temporal divergence: eyes turn outward away from nose
    # Subject right eye (468) moves temporal/left (-x in pixels)
    # Subject left eye (473) moves temporal/right (+x in pixels)
    ext.reset()
    lms_diverge = lms.copy()
    lms_diverge[468, 0] -= 10.0  # temporal outward
    lms_diverge[473, 0] += 10.0  # temporal outward
    feat_div = ext.compute(lms_diverge, timestamp=0.2, frame_id=2, face_valid=True)
    assert feat_div.vergence_angle_deg < -3.5, f"Expected < -3.5 deg for divergence, got {feat_div.vergence_angle_deg}"
    assert feat_div.lack_of_convergence is True


def test_tzone_coordinates_pixel_x1_not_misclassified_as_normalized():
    """(d) T-zone coordinates at pixel x=1 are not misclassified as normalized."""
    ext = FeatureExtractor()

    # Image of 640x480 (non-monochrome RGB)
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    frame[:, :, 0] = 50   # B
    frame[:, :, 1] = 100  # G
    frame[:, :, 2] = 150  # R

    # Pixel landmarks: one landmark at pixel (1.0, 1.0), but face spans realistic pixel coordinates
    lms_pixel = np.zeros((478, 3), dtype=np.float64)
    lms_pixel[0] = [1.0, 1.0, 0.0]  # x=1.0, y=1.0
    for i in range(1, 478):
        lms_pixel[i] = [320.0 + (i % 20), 240.0 + (i // 20), 0.0]

    # Sample T-zone RGB
    res = ext._sample_tzone_rgb(frame, lms_pixel)
    # Because landmarks are in pixel coordinates and not normalized (range 0..1),
    # the function must treat them as pixel coords (finite_pts not all <= 1.0)
    # and not multiply pixel coords by 640 / 480
    assert res is not None
    r, g, b = res
    assert r > 0 or g > 0 or b > 0


def test_invalid_rgb_frames_return_flushing_and_rppg_invalid():
    """(e) Invalid RGB frames return flushing_valid=False and rppg_valid=False."""
    ext = FeatureExtractor(fps=30.0)
    lms = np.zeros((478, 3), dtype=np.float64)
    lms[33] = [180.0, 200.0, 0.0]
    lms[133] = [220.0, 200.0, 0.0]
    lms[263] = [420.0, 200.0, 0.0]
    lms[362] = [380.0, 200.0, 0.0]

    # Frame with frame = None
    feat_none = ext.compute(lms, timestamp=0.1, frame_id=1, face_valid=True, frame=None)
    assert feat_none.flushing_valid is False
    assert feat_none.rppg_valid is False

    # Frame with 0-size frame
    empty_frame = np.zeros((0, 0, 3), dtype=np.uint8)
    feat_empty = ext.compute(lms, timestamp=0.2, frame_id=2, face_valid=True, frame=empty_frame)
    assert feat_empty.flushing_valid is False
    assert feat_empty.rppg_valid is False


def test_count_adjacent_valid_reversals_ignores_non_adjacent_flips():
    """(f) _count_adjacent_valid_reversals ignores non-adjacent sign flips."""
    # Strict adjacent test with >= 6 valid samples
    # 7 alternating diffs
    diffs = np.array([1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0])

    # Case 1: All frames valid -> 6 adjacent reversals
    mask_all = np.ones(7, dtype=bool)
    rev_all = FeatureExtractor._count_adjacent_valid_reversals(diffs, mask_all)
    assert rev_all == 6

    # Case 2: Mask has a gap breaking an adjacent reversal
    # diffs[0]=1.0, diffs[1]=-1.0 (sign flip). But if mask[1] is False,
    # the reversal between diffs[0] and diffs[1] must NOT be counted!
    mask_gap = np.array([True, False, True, True, True, True, True])
    rev_gap = FeatureExtractor._count_adjacent_valid_reversals(diffs, mask_gap)
    # adjacent_valid for (0,1) is False, for (1,2) is False.
    # The remaining adjacent pairs are (2,3), (3,4), (4,5), (5,6) -> 4 reversals
    assert rev_gap == 4
