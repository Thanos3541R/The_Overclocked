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
    from utils.landmarks import VASODILATION_LANDMARKS
    ext = FeatureExtractor()

    # Spatially varying frame: left-edge columns frame[:, :50] have BGR (40, 80, 200) (R=200, B=40)
    # and right-side columns frame[:, 50:] have BGR (200, 80, 40) (R=40, B=200)
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    frame[:, :50] = (40, 80, 200)
    frame[:, 50:] = (200, 80, 40)

    # Pixel landmarks: place all VASODILATION_LANDMARKS at [1.0, 100.0, 0.0]
    # except one landmark at [20.0, 100.0, 0.0] so the array is in pixel space.
    lms_pixel = np.zeros((478, 3), dtype=np.float64)
    for idx in VASODILATION_LANDMARKS:
        if idx < 478:
            lms_pixel[idx] = [1.0, 100.0, 0.0]
    if VASODILATION_LANDMARKS:
        lms_pixel[VASODILATION_LANDMARKS[0]] = [20.0, 100.0, 0.0]

    # Sample T-zone RGB
    res = ext._sample_tzone_rgb(frame, lms_pixel)
    assert res is not None
    r, g, b = res

    # If correctly treated as pixel coordinates, it samples the red left edge (r > 180, b < 60).
    # If misclassified as normalized, x=1.0 would scale to 640, sampling the blue right edge (b=200, r=40).
    assert r > 180, f"Expected red left edge (r > 180), got r={r:.1f}"
    assert b < 60, f"Expected low blue on left edge (b < 60), got b={b:.1f}"


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


def test_bayesian_skip_unobserved_preserves_posterior():
    """Verify update() with elevated assessment followed by INSUFFICIENT_DATA preserves posterior & log_odds."""
    from pipeline.bayesian_filter import BayesianEvidenceAccumulator
    from pipeline.classifier import ImpairmentAssessment

    filter_bank = BayesianEvidenceAccumulator()

    # Step 1: Provide elevated impairment assessment across a few frames
    elevated_assessment = ImpairmentAssessment(
        risk_score=75.0,
        classification="HIGH_RISK_INTOXICATED",
        confidence=0.88,
        involuntary_risk_score=75.0,
        total_risk_score=75.0,
    )
    for step in range(5):
        s_elevated = filter_bank.update(elevated_assessment, timestamp=1.0 + step * 0.1)
    p_initial = s_elevated.posterior_probability
    lo_initial = s_elevated.log_odds
    assert p_initial > filter_bank.prior_prob  # Has accumulated evidence

    # Step 2: 5 consecutive INSUFFICIENT_DATA (gated/unobserved) assessments
    gated_assessment = ImpairmentAssessment(
        risk_score=0.0,
        classification="INSUFFICIENT_DATA",
        confidence=0.0,
    )
    for i in range(1, 6):
        t = 2.0 + i * 0.1
        s_gated = filter_bank.update(gated_assessment, timestamp=t)
        assert s_gated.posterior_probability == pytest.approx(p_initial, abs=1e-6)
        assert s_gated.log_odds == pytest.approx(lo_initial, abs=1e-6)


def test_classifier_populates_alias_fields_on_valid_frames():
    """Verify evaluate() and evaluate_temporal_stream() populate all alias fields on valid frames."""
    from pipeline.classifier import ImpairmentClassifier, StarterLockout, ThrottleState
    from pipeline.feature_extractor import FrameFeatures

    clf = ImpairmentClassifier()

    # Test static evaluate()
    dummy_frame = np.full((480, 640, 3), 180, dtype=np.uint8)
    landmarks = np.zeros((478, 3), dtype=np.float64)
    assess_static = clf.evaluate(
        frame=dummy_frame,
        landmarks=landmarks,
        ear_left=0.25,
        ear_right=0.25,
        mar=0.10,
        head_pitch=0.0,
        head_roll=0.0,
    )
    assert assess_static.involuntary_score == assess_static.involuntary_risk_score
    assert assess_static.total_risk_score == assess_static.risk_score
    assert assess_static.starter_lockout in (StarterLockout.ALLOW, StarterLockout.LOCK)
    assert assess_static.throttle_state in (ThrottleState.NORMAL, ThrottleState.LIMITED)
    assert isinstance(assess_static.reason, str) and len(assess_static.reason) > 0

    # Also test evaluate with FrameFeatures
    feat_static = FrameFeatures(
        perclos=0.25,
        blink_opening_velocity=1.2,
        saccade_velocity=45.0,
    )
    assess_feat = clf.evaluate(feat_static)
    assert assess_feat.involuntary_score == assess_feat.involuntary_risk_score
    assert assess_feat.total_risk_score == assess_feat.risk_score

    # Test temporal stream evaluate_temporal_stream()
    temp_features = FrameFeatures(
        face_valid=True,
        imu_gated=False,
        perclos=0.22,
        blink_opening_velocity=1.4,
        saccade_velocity=50.0,
        gaze_yaw_dispersion=0.18,
    )
    assess_temp = clf.evaluate_temporal_stream(temp_features)
    assert assess_temp.involuntary_score == assess_temp.involuntary_risk_score
    assert assess_temp.total_risk_score == assess_temp.risk_score
    assert assess_temp.voluntary_score == assess_temp.voluntary_masking_score
    assert assess_temp.starter_lockout in (StarterLockout.ALLOW, StarterLockout.LOCK)
    assert assess_temp.throttle_state in (ThrottleState.NORMAL, ThrottleState.LIMITED)
    assert isinstance(assess_temp.reason, str) and len(assess_temp.reason) > 0


def test_main_module_imports_cleanly():
    """Verify import main succeeds without NameError."""
    import importlib
    import main
    importlib.reload(main)
    assert hasattr(main, "main")


def test_resting_vergence_auto_zeroing_on_real_frames():
    """Verify resting vergence auto-zeroing adapts on real frames and is bypassed on synthetic frames."""
    from utils.landmarks import (
        RIGHT_EYE_OUTER, RIGHT_EYE_INNER, LEFT_EYE_OUTER, LEFT_EYE_INNER,
        RIGHT_IRIS_CENTER, LEFT_IRIS_CENTER, RIGHT_EYE_EAR_FLAT, LEFT_EYE_EAR_FLAT
    )
    ext = FeatureExtractor()

    # Build realistic eye geometry
    lms = np.zeros((478, 3), dtype=np.float64)
    # Right eye (subject's right, viewer's left: outer=33, inner=133)
    lms[RIGHT_EYE_OUTER] = [180.0, 200.0, 0.0]
    lms[RIGHT_EYE_INNER] = [220.0, 200.0, 0.0]
    # Small offset on right iris (+1.0 px normalized offset)
    lms[RIGHT_IRIS_CENTER] = [201.0, 200.0, 0.0]
    for idx in RIGHT_EYE_EAR_FLAT[1:3]:
        lms[idx] = [200.0, 190.0, 0.0]
    for idx in RIGHT_EYE_EAR_FLAT[4:]:
        lms[idx] = [200.0, 210.0, 0.0]

    # Left eye (subject's left, viewer's right: inner=362, outer=263)
    lms[LEFT_EYE_INNER] = [380.0, 200.0, 0.0]
    lms[LEFT_EYE_OUTER] = [420.0, 200.0, 0.0]
    lms[LEFT_IRIS_CENTER] = [400.0, 200.0, 0.0]
    for idx in LEFT_EYE_EAR_FLAT[1:3]:
        lms[idx] = [400.0, 190.0, 0.0]
    for idx in LEFT_EYE_EAR_FLAT[4:]:
        lms[idx] = [400.0, 210.0, 0.0]

    # Synthetic frame (image_landmarks is None) -> _baseline_vergence_deg stays 0.0
    f_synth = ext.compute(lms, timestamp=1.0, frame_id=1, face_valid=True, image_landmarks=None)
    assert ext._baseline_vergence_deg == 0.0

    # Real frame (image_landmarks is not None) with constant offset
    # Simulate 15 frames with forward gaze and static vergence offset
    for i in range(15):
        t = 2.0 + i * 0.033
        f_real = ext.compute(lms, timestamp=t, frame_id=10 + i, face_valid=True, image_landmarks=lms)

    assert len(ext._vergence_cal_samples) >= 10
    # Baseline was learned and subtracted from real frames
    assert f_real.vergence_angle_deg == pytest.approx(0.0, abs=0.5)
    assert f_real.lack_of_convergence is False
    assert ext._baseline_vergence_deg != 0.0

    # Simulate closed-eye blink sequence; held vergence must NOT accumulate -baseline
    vergence_before_blink = f_real.vergence_angle_deg
    lms_closed = lms.copy()
    # Flatten eye landmarks so EAR < 0.15 (closed eye)
    for idx in RIGHT_EYE_EAR_FLAT[1:]:
        lms_closed[idx] = lms[RIGHT_EYE_EAR_FLAT[0]]
    for idx in LEFT_EYE_EAR_FLAT[1:]:
        lms_closed[idx] = lms[LEFT_EYE_EAR_FLAT[0]]

    for j in range(10):
        t_closed = 2.462 + (j + 1) * 0.033
        f_closed = ext.compute(lms_closed, timestamp=t_closed, frame_id=30 + j, face_valid=True, image_landmarks=lms_closed)
        assert f_closed.blink_phase in ("closing", "closed")
        assert f_closed.vergence_angle_deg == pytest.approx(vergence_before_blink, abs=1e-5)


def test_low_fps_blink_upstroke_chord_floor():
    """Verify low-FPS (dt >= 0.042) blink upstroke computes physiological chord velocity."""
    ext = FeatureExtractor(fps=20.0)

    # Simulate blink sequence at ~17 FPS (dt = 0.06s)
    # Frame 1: Open
    ext._compute_blink_velocity(0.28, timestamp=1.00)
    assert ext._blink_phase == "open"

    # Frame 2: Closing
    ext._compute_blink_velocity(0.18, timestamp=1.06)
    assert ext._blink_phase in ("closing", "closed")

    # Frame 3: Closed at nadir (0.08)
    ext._compute_blink_velocity(0.08, timestamp=1.12)
    assert ext._blink_phase == "closed"
    assert ext._min_closed_ear == pytest.approx(0.08, abs=1e-4)

    # Frame 4: Partial reopening
    ext._compute_blink_velocity(0.16, timestamp=1.18)
    assert ext._blink_phase == "opening"

    # Frame 5: Reopening slow step
    ext._compute_blink_velocity(0.20, timestamp=1.24)
    assert ext._blink_phase == "opening"

    # Frame 6: Reaches full open baseline (0.25)
    # At ~17 FPS, dt = 0.06s >= 0.042
    ext._compute_blink_velocity(0.25, timestamp=1.30)
    assert ext._blink_phase == "open"
    # Chord upstroke velocity: (0.25 - 0.08) / 0.085 = 2.00 EAR/s
    expected_chord = (0.25 - 0.08) / 0.085
    assert ext._last_blink_opening_vel == pytest.approx(expected_chord, abs=0.15)
    assert ext._last_blink_opening_vel >= 1.8
