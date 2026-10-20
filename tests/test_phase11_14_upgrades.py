"""Unit and integration tests for Phases 11-14 DMS upgrades.

Verifies:
1. Eyeball oscillation frequency (eyeball_tremor_hz) detection, blink/saccade gating, and vibration discounting.
2. Single-pass T-zone RGB sampling and strided rPPG FFT evaluation consistency.
3. ImpairmentClassifier evaluation of eyeball_tremor_hz and corroborating involuntary risk bonus.
4. Visualizer top-center status banner, micromotion HUD rows, context guard badges, and [r] ROI hotkey.
5. inspect_driver.py --video CLI end-to-end forensic report generation.
"""
import json
import os
import sys
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline.feature_extractor import FeatureExtractor, FrameFeatures
from pipeline.classifier import ImpairmentClassifier, ImpairmentAssessment
from pipeline.bayesian_filter import BayesianState
from utils.visualization import Visualizer
from utils.landmarks import VASODILATION_LANDMARKS
from tests.test_alcohol_signals import make_full_landmarks
import inspect_driver


def test_eyeball_tremor_hz_detection_and_guards():
    """Verify horizontal eyeball micro-oscillation frequency (Hz) and guards."""
    fps = 60.0
    ext = FeatureExtractor(fps=fps)

    # 1. Feed 60 frames (1.0s) of 8.0 Hz sub-saccadic iris oscillation (amplitude 0.25 px -> norm 0.0125)
    feat = None
    for i in range(60):
        t = i / fps
        iris_offset = 0.25 * np.sin(2.0 * np.pi * 8.0 * t)
        lm = make_full_landmarks(ear_val=0.30, iris_x_offset=iris_offset)
        feat = ext.compute(lm, timestamp=t, frame_id=i, face_valid=True, vibration_score=0.0)

    assert feat is not None
    assert feat.saccade_detected is False
    assert feat.eyeball_tremor_hz > 5.0, f"Expected eyeball_tremor_hz > 5.0 Hz, got {feat.eyeball_tremor_hz}"

    # 2. Verify blink phase and saccade guards zero out eyeball tremor
    assert ext._compute_eyeball_tremor(0.0, 1.1, "closed", False, vibration_score=0.0) == 0.0
    assert ext._compute_eyeball_tremor(0.0, 1.1, "open", True, vibration_score=0.0) == 0.0

    # 3. Verify cabin vibration_score > 0.35 discounts eyeball_tremor_hz
    t_base = ext._compute_eyeball_tremor(0.01, 1.02, "open", False, vibration_score=0.0)
    t_vib = ext._compute_eyeball_tremor(-0.01, 1.04, "open", False, vibration_score=0.60)
    assert t_vib <= t_base * 0.45 + 0.1


def test_single_pass_tzone_rgb_and_strided_rppg_consistency():
    """Verify _sample_tzone_rgb single-pass helper and strided rPPG FFT accuracy."""
    ext = FeatureExtractor(fps=30.0)
    lm = make_full_landmarks(ear_val=0.30)
    for idx in VASODILATION_LANDMARKS:
        lm[idx] = [300.0, 300.0, 0.0]

    # 1. Monochrome frame returns None
    gray = np.full((400, 400), 120, dtype=np.uint8)
    assert ext._sample_tzone_rgb(gray, lm) is None

    # 2. RGB frame returns exact (R, G, B) tuple
    rgb_frame = np.zeros((400, 400, 3), dtype=np.uint8)
    rgb_frame[290:315, 290:315] = [100, 130, 170]  # BGR
    sampled = ext._sample_tzone_rgb(rgb_frame, lm)
    assert sampled == pytest.approx((170.0, 130.0, 100.0), abs=1e-3)

    # 3. 150 frames (5.0s) of 1.2 Hz (72 BPM) green oscillation with strided FFT
    feats = []
    for i in range(150):
        t = i / 30.0
        pulse = 1.0 + 0.03 * np.sin(2.0 * np.pi * 1.2 * t)
        g_val = int(np.clip(120.0 * pulse, 0, 255))
        f_img = np.zeros((400, 400, 3), dtype=np.uint8)
        f_img[290:315, 290:315] = [100, g_val, 150]
        feat = ext.compute(lm, timestamp=t, frame_id=i, face_valid=True, frame=f_img)
        feats.append(feat)

    # After 4.0s warmup (frame 121+), rppg_valid must stay True on every frame (including strided frames)
    for f in feats[122:]:
        assert f.rppg_valid is True
        assert f.rppg_pulse_bpm == pytest.approx(72.0, abs=1.5)


def test_classifier_evaluates_eyeball_tremor():
    """Verify ImpairmentClassifier scores eyeball_tremor_hz and adds corroborating risk."""
    clf = ImpairmentClassifier()
    feats = FrameFeatures(
        face_valid=True,
        eyeball_tremor_hz=5.8,
        gen_detected=True,  # Base inv_risk = 90 * 0.24 = 21.6 >= 20.0
    )
    assessment = clf.evaluate_temporal_stream(feats)
    assert assessment.marker_scores.get("eyeball_tremor") == 65.0
    assert assessment.involuntary_risk_score == pytest.approx(21.6 + 4.0)
    assert any("Ocular eyeball micro-tremor" in ind for ind in assessment.primary_indicators)


def test_visualizer_status_banner_context_badges_and_roi_hotkey():
    """Verify Visualizer renders status banner, ROI box [r], and context guard badges."""
    viz = Visualizer(overlay_mode="regions")
    assert viz.show_roi_box is False
    assert viz.handle_key(ord("r")) is True
    assert viz.show_roi_box is True

    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    lm = make_full_landmarks(ear_val=0.28)
    feats = FrameFeatures(
        face_valid=True,
        eyelid_tremor_hz=5.2,
        eyeball_tremor_hz=4.8,
        brow_elevation_delta=0.15,
        rppg_pulse_bpm=74.0,
        rppg_hrv_rmssd_ms=32.0,
        vibration_score=0.22,
        masking_divergence_score=18.5,
        eyeglass_glare_detected=True,
        sunglasses_occluded=True,
        speech_active=True,
        highway_fixation_exempt=True,
    )
    assessment = ImpairmentAssessment(
        risk_score=32.5,
        classification="MILD_IMPAIRMENT",
        involuntary_risk_score=28.0,
    )
    b_state = BayesianState(
        posterior_probability=0.45,
        sustained_high_confidence_s=2.1,
        lockout_confirmed=False,
    )

    out = viz.draw(
        frame,
        lm,
        features=feats,
        assessment=assessment,
        bayesian_state=b_state,
        calibration_progress=0.75,
        roi_bounds=(400, 180, 800, 580),
        shift_vector=(2.5, -4.0),
    )
    assert out.shape == (720, 1280, 3)
    assert np.count_nonzero(out) > 0


def test_inspect_driver_video_cli_integration(monkeypatch, tmp_path):
    """Verify inspect_driver.py --video runs extract_video_features and outputs forensic files."""
    out_csv = tmp_path / "forensic_out.csv"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "inspect_driver.py",
            "--video",
            "tests/sample_driver_video.mp4",
            "--output",
            str(out_csv),
            "--headless",
        ],
    )
    inspect_driver.main()

    summary_json = tmp_path / "video_features_summary.json"
    report_txt = tmp_path / "video_features_report.txt"
    assert out_csv.exists()
    assert summary_json.exists()
    assert report_txt.exists()

    with open(summary_json, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert "mean_eyeball_tremor_hz" in data["section_a_involuntary_biomarkers"]


def test_infer_subject_id_groups_youtube_clips():
    """Verify infer_subject_id correctly strips suffixes and groups YouTube clips."""
    from ingest_and_audit_clips import infer_subject_id

    id1 = infer_subject_id("Drunk/ drunk_driver_abc_clip01.mp4", "drunk")
    id2 = infer_subject_id("Drunk/drunk_driver_abc_clip02.mp4", "drunk")
    assert id1 == id2
    assert id1 == "drunk_driver_abc"

    # Verify part, segment, and nested folder variations
    assert infer_subject_id("Sober/videoA_part1.mp4", "sober") == infer_subject_id(
        "Sober/videoA_part2.mp4", "sober"
    )
    assert infer_subject_id("clips/yt_abc123_001.mp4", "drunk") == infer_subject_id(
        "clips/yt_abc123_002.mp4", "drunk"
    )
    assert infer_subject_id("Sober/person_1/clip01.mp4", "sober") == infer_subject_id(
        "Sober/person_1/clip02.mp4", "sober"
    )


def test_in_fold_minority_augmentation_balances_classes():
    """Verify augment_minority_windows_in_fold balances a 20-sober / 5-drunk matrix to 20/20 without NaNs."""
    from train_baseline import augment_minority_windows_in_fold

    rng = np.random.RandomState(42)
    # 20 sober (0), 5 drunk (1) with 10 features (8 continuous, 2 binary)
    X_neg = rng.randn(20, 10).astype(np.float32)
    X_pos = rng.randn(5, 10).astype(np.float32) + 2.0
    # Set columns 8 and 9 as binary indicators (0 or 1)
    X_neg[:, 8] = 0.0
    X_neg[:, 9] = 1.0
    X_pos[:, 8] = 1.0
    X_pos[:, 9] = 0.0

    X_train = np.vstack([X_neg, X_pos])
    y_train = np.array([0] * 20 + [1] * 5, dtype=np.int32)

    X_aug, y_aug = augment_minority_windows_in_fold(X_train, y_train, random_state=42)

    assert len(y_aug) == 40
    assert np.sum(y_aug == 0) == 20
    assert np.sum(y_aug == 1) == 20
    assert not np.isnan(X_aug).any()
    assert X_aug.shape == (40, 10)
    # Binary columns must strictly retain their 0.0 / 1.0 values without Gaussian jitter
    assert np.all(np.isin(X_aug[:, 8], [0.0, 1.0]))
    assert np.all(np.isin(X_aug[:, 9], [0.0, 1.0]))


def test_stratified_group_kfold_and_nonzero_feature_importance():
    """Verify StratifiedGroupKFold fallback to StratifiedKFold when class representation is unbalanced across subjects,
    and verify non-zero feature importance via stratified sampling / Cohen's d fallback."""
    from train_baseline import train_and_evaluate

    rng = np.random.RandomState(42)
    # 1 Sober subject (40 windows) and 4 Drunk subjects (15 windows each = 60 windows)
    # Total windows: 100
    n_features = 15
    feature_names = [f"feat_{i}" for i in range(n_features)]

    # Feat 0 has strong signal between sober and drunk
    X_sober = rng.randn(40, n_features).astype(np.float32)
    X_sober[:, 0] -= 2.0  # sober has lower feat 0

    X_drunk = rng.randn(60, n_features).astype(np.float32)
    X_drunk[:, 0] += 2.0  # drunk has higher feat 0

    X = np.vstack([X_sober, X_drunk])
    y = np.array([0] * 40 + [1] * 60, dtype=np.int32)
    subjects = np.array(
        ["sober_sub_01"] * 40
        + ["drunk_sub_01"] * 15
        + ["drunk_sub_02"] * 15
        + ["drunk_sub_03"] * 15
        + ["drunk_sub_04"] * 15
    )

    results, model = train_and_evaluate(
        X=X,
        y=y,
        subjects=subjects,
        feature_names=feature_names,
        n_splits=5,
        use_xgboost=False,
        random_state=42,
    )

    assert results["oof_metrics"]["roc_auc"] > 0.65
    assert len(results["top_features"]) > 0
    assert results["top_features"][0]["importance"] > 0.0
    assert results["top_features"][0]["feature"] == "feat_0"


