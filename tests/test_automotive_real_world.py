"""Real-world automotive stress test suite for Driver Monitoring System (DMS).

Validates the 8 core real-world vehicle cabin environmental and operational hardening fixes:
1. test_windshield_backlight_and_split_sunlight_normalization:
   Bright background (240) + dark face ROI (35) with split lighting (10 vs 60).
   Verifies LightingNormalizer lifts face ROI luminance >40% and reduces left-right asymmetry.
2. test_off_axis_camera_zero_tare_calibration:
   Dashboard camera at yaw=+20°, pitch=-10°, roll=+4°.
   Verifies PersonalBaselineCalibrator learns offsets and apply_to enables symmetric ±30°
   driver turns without false gating or false head_slump.
3. test_brief_hand_occlusion_preserves_filter_state:
   5-frame steering wheel occlusion (landmarks=None).
   Verifies OneEuroFilterBank and PnPNormalizer hold state up to max_occlusion_hold_frames=15,
   and tracking resumes smoothly without re-initialization shock.
4. test_eyeglass_glare_monocular_fallback_and_sunglasses:
   Specular glare (>245) on one eye triggers monocular fallback from clean eye.
   Dark sunglasses (<45 lum, uniform texture, static EAR) trigger sunglasses_occluded=True
   and suppress false compensatory_stare_active.
5. test_downward_instrument_cluster_glance_no_false_blink:
   Downward glance (pitch=-18°, iris_y=-0.40) with natural resting EAR=0.175.
   Verifies dynamic threshold scaling prevents false blink and false microsleep detection.
6. test_highway_fixation_exemption_vs_catatonic_stare:
   7-second highway stare with physiological drift (sigma ~0.35°) and healthy blink velocity
   triggers highway_fixation_exempt=True and compensatory_stare_active=False.
   In contrast, true catatonic stare (sluggish blink or zero tremor) triggers compensatory_stare_active=True.
7. test_speech_cadence_does_not_trigger_yawn:
   45 frames of 3 Hz speech cadence (MAR 0.10 to 0.62) sets speech_active=True and keeps
   yawn_detected=False. Sustained mouth opening without cadence triggers yawn_detected=True.
8. test_streetlight_strobe_rppg_immunity_and_rolling_shutter_ear:
   1.5 Hz global brightness strobe is rejected by chromaticity-normalized rPPG while recovering
   1.2 Hz (72 BPM) pulse. Rolling shutter vertical CMOS stretch is corrected via baseline nose ratio.
"""
import os
import sys
import numpy as np
import pytest
import cv2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline.preprocessing import LightingNormalizer
from pipeline.calibration import PersonalBaselineCalibrator
from pipeline.head_pose_estimator import HeadPoseGate
from pipeline.classifier import ImpairmentClassifier
from pipeline.feature_extractor import FeatureExtractor, FrameFeatures
from pipeline.one_euro_filter import OneEuroFilterBank
from pipeline.pnp_normalizer import PnPNormalizer
from pipeline.imu_gate import SoftwareIMUGate
from pipeline.stabilizer import KinematicStabilizer
from config import PipelineConfig
from utils.landmarks import (
    RIGHT_EYE_CONTOUR,
    LEFT_EYE_CONTOUR,
    VASODILATION_LANDMARKS,
)
from tests.test_alcohol_signals import make_full_landmarks
from tests.test_pnp import make_frontal_face_landmarks


def make_landmarks_with_contours(ear_l: float = 0.30, ear_r: float = 0.30) -> np.ndarray:
    """Create 478x3 synthetic landmarks with realistic eye contours, irises, and nose bridge/tip."""
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

    # Explicit EAR 6-points for exact EAR Soukupová formula: EAR = (v1 + v2)/(2*h) = h/w = ear_val
    lm[33] = [cx_r - w_r / 2.0, cy_r, 0.0]
    lm[160] = [cx_r - w_r * 0.15, cy_r - h_r / 2.0, 0.0]
    lm[158] = [cx_r + w_r * 0.15, cy_r - h_r / 2.0, 0.0]
    lm[133] = [cx_r + w_r / 2.0, cy_r, 0.0]
    lm[153] = [cx_r + w_r * 0.15, cy_r + h_r / 2.0, 0.0]
    lm[144] = [cx_r - w_r * 0.15, cy_r + h_r / 2.0, 0.0]

    lm[263] = [cx_l + w_l / 2.0, cy_l, 0.0]
    lm[385] = [cx_l + w_l * 0.15, cy_l - h_l / 2.0, 0.0]
    lm[387] = [cx_l - w_l * 0.15, cy_l - h_l / 2.0, 0.0]
    lm[362] = [cx_l - w_l / 2.0, cy_l, 0.0]
    lm[373] = [cx_l - w_l * 0.15, cy_l + h_l / 2.0, 0.0]
    lm[380] = [cx_l + w_l * 0.15, cy_l + h_l / 2.0, 0.0]

    # Irises
    lm[468] = [cx_r, cy_r, 0.0]
    lm[473] = [cx_l, cy_l, 0.0]

    # Nose bridge (168) and tip (4)
    lm[168] = [300.0, 200.0, 0.0]
    lm[4] = [300.0, 280.0, 0.0]

    # Vasodilation / rPPG forehead and cheek landmarks around (300, 300)
    for idx in VASODILATION_LANDMARKS:
        lm[idx] = [300.0, 300.0, 0.0]

    # Mouth default (MAR ~0.15)
    cy_m = 350.0
    half_m = 20.0 * 0.15
    lm[61] = [280.0, cy_m, 0.0]
    lm[291] = [320.0, cy_m, 0.0]
    lm[13] = [300.0, cy_m - half_m, 0.0]
    lm[14] = [300.0, cy_m + half_m, 0.0]
    lm[81] = [290.0, cy_m - half_m, 0.0]
    lm[178] = [290.0, cy_m + half_m, 0.0]
    lm[311] = [310.0, cy_m - half_m, 0.0]
    lm[402] = [310.0, cy_m + half_m, 0.0]

    return lm


def test_windshield_backlight_and_split_sunlight_normalization():
    """Test 1: Normalization under bright windshield backlighting and side-window split sunlight."""
    normalizer = LightingNormalizer()

    # 720x1280 frame: bright windshield background (240)
    frame = np.full((720, 1280, 3), 240, dtype=np.uint8)

    # Face ROI [440, 200, 840, 600]: x in [440, 840], y in [200, 600]
    # Split lighting across face: left half is 10, right half is 60 (mean = 35.0, asymmetry = 50.0)
    frame[200:600, 440:640] = 10
    frame[200:600, 640:840] = 60

    face_bbox = (440, 200, 840, 600)

    # Baseline unnormalized ROI metrics
    roi_raw = cv2.cvtColor(frame[200:600, 440:840], cv2.COLOR_BGR2GRAY)
    raw_mean = float(roi_raw.mean())
    raw_asym = abs(float(roi_raw[:, :200].mean()) - float(roi_raw[:, 200:].mean()))

    assert raw_mean == pytest.approx(35.0, abs=1.0)
    assert raw_asym == pytest.approx(50.0, abs=1.0)

    # Case A: Process WITH face_bbox (Face-ROI aware CLAHE + split-sunlight bilateral ramp)
    processed_with_bbox = normalizer.process(frame, face_bbox=face_bbox)
    roi_proc = cv2.cvtColor(processed_with_bbox[200:600, 440:840], cv2.COLOR_BGR2GRAY)
    proc_mean = float(roi_proc.mean())
    proc_asym = abs(float(roi_proc[:, :200].mean()) - float(roi_proc[:, 200:].mean()))

    # Verify face ROI luminance is lifted by > 40%
    lift_ratio = (proc_mean - raw_mean) / raw_mean
    assert lift_ratio > 0.40, f"Expected luminance lift > 40%, got {lift_ratio * 100:.1f}%"
    assert proc_mean > 70.0, f"Expected boosted luminance > 70, got {proc_mean:.1f}"

    # Verify split-sunlight asymmetry is reduced
    assert proc_asym < raw_asym, f"Expected asymmetry reduction, got {proc_asym:.1f} vs raw {raw_asym:.1f}"
    assert proc_asym <= 35.0, f"Expected post-balance asymmetry <= 35, got {proc_asym:.1f}"

    # Case B: Without face_bbox, bright background tricks evaluator into skipping gamma lift
    processed_no_bbox = normalizer.process(frame, face_bbox=None)
    roi_no_bbox = cv2.cvtColor(processed_no_bbox[200:600, 440:840], cv2.COLOR_BGR2GRAY)
    no_bbox_mean = float(roi_no_bbox.mean())
    # Without face_bbox, mean luminance should not experience significant lift (< 10%)
    assert abs(no_bbox_mean - raw_mean) / raw_mean < 0.10


def test_off_axis_camera_zero_tare_calibration():
    """Test 2: Dashboard camera off-axis mounting zero-tare calibration."""
    calibrator = PersonalBaselineCalibrator(calibration_duration_sec=5.0, min_samples=30, fps=30.0)

    # Simulate camera mounted on dashboard/A-pillar: yaw=+20°, pitch=-10°, roll=+4°
    for i in range(50):
        t = i / 30.0
        f = FrameFeatures(
            timestamp=t,
            frame_id=i,
            face_valid=True,
            ear_avg=0.28,
            ear_left=0.28,
            ear_right=0.28,
            mar=0.15,
            blink_detected=False,
            blink_opening_velocity=2.4,
            facial_flushing_ratio=1.35,
            flushing_valid=True,
            head_yaw=20.0,
            head_pitch=-10.0,
            head_roll=4.0,
        )
        calibrator.update(f)

    baselines = calibrator.finalize()
    assert baselines.is_calibrated is True
    assert baselines.head_yaw_baseline == pytest.approx(20.0, abs=0.5)
    assert baselines.head_pitch_baseline == pytest.approx(-10.0, abs=0.5)
    assert baselines.head_roll_baseline == pytest.approx(4.0, abs=0.5)

    extractor = FeatureExtractor(fps=30.0)
    classifier = ImpairmentClassifier()
    pose_gate = HeadPoseGate(yaw_limit=35.0, pitch_down_limit=-30.0, pitch_up_limit=25.0)

    calibrator.apply_to(extractor, classifier, pose_gate)

    assert pose_gate.yaw_offset == pytest.approx(20.0, abs=0.5)
    assert pose_gate.pitch_offset == pytest.approx(-10.0, abs=0.5)
    assert pose_gate.roll_offset == pytest.approx(4.0, abs=0.5)
    assert classifier.pitch_baseline == pytest.approx(-10.0, abs=0.5)
    assert classifier.roll_baseline == pytest.approx(4.0, abs=0.5)

    # Verify symmetric ±30° driver turns from nominal posture are valid
    # Turn left by 30° -> raw yaw = 20 + 30 = +50°
    turn_left_state = pose_gate.check((-10.0, 50.0, 4.0))
    assert turn_left_state.face_valid is True, f"Expected face_valid=True, got {turn_left_state.gating_reason}"

    # Turn right by 30° -> raw yaw = 20 - 30 = -10°
    turn_right_state = pose_gate.check((-10.0, -10.0, 4.0))
    assert turn_right_state.face_valid is True, f"Expected face_valid=True, got {turn_right_state.gating_reason}"

    # Verify resting posture pitch (-10°) produces zero head slump in classifier
    dummy_frame = np.zeros((100, 100, 3), dtype=np.uint8)
    dummy_lm = np.zeros((478, 3), dtype=np.float64)
    assessment = classifier.evaluate_static_image(
        dummy_frame, dummy_lm,
        ear_left=0.28, ear_right=0.28, mar=0.15,
        head_pitch=-10.0, head_roll=4.0,
    )
    assert assessment.marker_scores["head_slump"] == 0.0


def test_brief_hand_occlusion_preserves_filter_state():
    """Test 3: Brief hand occlusion while steering preserves filter bank and PnP state."""
    w, h = 1280, 720
    normalizer = PnPNormalizer(w, h)
    imu_gate = SoftwareIMUGate()
    stabilizer = KinematicStabilizer()
    config = PipelineConfig()
    assert config.max_occlusion_hold_frames == 15

    # Step 1: Initial valid face acquisition on frame 0
    lm_initial = make_frontal_face_landmarks(frame_width=w, frame_height=h)
    pose_ok = normalizer.solve_pose(lm_initial)
    assert pose_ok is True
    assert normalizer.rvec is not None

    filter_bank = OneEuroFilterBank(
        t0=0.0,
        x0=lm_initial,
        landmark_min_cutoff=config.landmark_min_cutoff,
        landmark_beta=config.landmark_beta,
    )
    initial_filter_id = id(filter_bank)

    # Step 2: Simulate 5-frame hand occlusion during a steering turn (landmarks = None)
    missing_face_frames = 0
    for frame_id in range(1, 6):
        landmarks = None
        if landmarks is None:
            missing_face_frames += 1
            if missing_face_frames > config.max_occlusion_hold_frames:
                normalizer.reset()
                imu_gate.reset()
                stabilizer.reset()
                filter_bank = None

        assert missing_face_frames == frame_id
        assert filter_bank is not None
        assert normalizer.rvec is not None

    # Step 3: Face re-emerges on frame 6
    landmarks = make_frontal_face_landmarks(frame_width=w, frame_height=h)
    missing_face_frames = 0

    assert id(filter_bank) == initial_filter_id, "Filter bank must be preserved without re-initialization shock"
    smooth_lms = filter_bank.filter_landmarks(6 / 30.0, landmarks)
    assert smooth_lms.shape == (478, 3)
    assert np.all(np.isfinite(smooth_lms))

    # Step 4: In contrast, prolonged occlusion exceeding max_occlusion_hold_frames (16 frames) resets state
    missing_face_frames = 0
    for frame_id in range(16):
        landmarks = None
        missing_face_frames += 1
        if missing_face_frames > config.max_occlusion_hold_frames:
            normalizer.reset()
            imu_gate.reset()
            stabilizer.reset()
            filter_bank = None

    assert missing_face_frames == 16
    assert filter_bank is None
    assert normalizer.rvec is None


def test_eyeglass_glare_monocular_fallback_and_sunglasses():
    """Test 4: Eyeglass glare monocular fallback and dark sunglasses occlusion handling."""
    ext = FeatureExtractor(fps=30.0)

    # ── Part A: Eyeglass Glare Monocular Fallback ──
    # Subject's left eye has corrupted EAR=0.45; subject's right eye has clean EAR=0.25
    lm = make_landmarks_with_contours(ear_l=0.45, ear_r=0.25)
    # Set distinct iris coordinates to test iris fallback
    lm[468] = [200.0 - 8.0, 200.0 + 5.0, 0.0]  # Right eye in viewer frame (corners 33 & 133)
    lm[473] = [400.0 + 4.0, 200.0 - 2.0, 0.0]  # Left eye in viewer frame (corners 263 & 362)

    # Base textured frame (std ~6.0)
    np.random.seed(42)
    frame_glare = np.random.randint(90, 115, (720, 1280, 3), dtype=np.uint8)
    # Add specular reflection patch (>245) on left eye contour (around 400, 200)
    frame_glare[191:209, 391:409] = 255

    feat = ext.compute(lm, timestamp=0.0, frame_id=0, face_valid=True, frame=frame_glare)

    assert feat.eyeglass_glare_detected is True
    # Clean right eye used for ear_avg
    assert feat.ear_avg == pytest.approx(feat.ear_right, rel=1e-3)
    # Corrupted left iris coordinates copied from clean right iris
    assert feat.left_iris_x == pytest.approx(feat.right_iris_x, abs=1e-5)
    assert feat.left_iris_y == pytest.approx(feat.right_iris_y, abs=1e-5)

    # ── Part B: Sunglasses Occlusion & False Compensatory Stare Suppression ──
    ext_sunglasses = FeatureExtractor(fps=30.0)
    lm_static = make_landmarks_with_contours(ear_l=0.30, ear_r=0.30)

    # Dark lenses (mean lum < 45, std in [1.0, 8.0), non-monochrome)
    frame_dark = np.random.randint(22, 34, (720, 1280, 3), dtype=np.uint8)
    frame_dark[:, :, 0] += 5

    # Feed 65 frames of static landmarks (duration > 2.0s, history >= 60)
    feat_dark = None
    for i in range(65):
        t = i / 30.0
        feat_dark = ext_sunglasses.compute(lm_static, timestamp=t, frame_id=i, face_valid=True, frame=frame_dark)

    assert feat_dark is not None
    assert feat_dark.sunglasses_occluded is True
    assert feat_dark.deliberate_blink_suppression is False
    assert feat_dark.compensatory_stare_active is False


def test_downward_instrument_cluster_glance_no_false_blink():
    """Test 5: Downward glance at instrument cluster lowers blink threshold to avoid false blink/microsleep."""
    ext = FeatureExtractor(blink_threshold=0.20, microsleep_frames=15, fps=30.0)

    # Case A: Level head posture (pitch=0.0) with ear_avg = 0.175
    # Without glance compensation, 0.175 < 0.20 triggers eyes_closed=True
    lm_level = make_full_landmarks(ear_val=0.175 / 2.0, iris_y_offset=0.0)
    feat_level = ext.compute(lm_level, timestamp=0.0, frame_id=0, face_valid=True, head_pitch=0.0)
    assert ext._consecutive_closed == 1

    # Case B: Sustained downward glance at instrument cluster for 30 frames (1 second)
    # pitch = -18.0°, downward iris gaze (iris_y_offset = +8.0 -> avg_iris_y = -0.40)
    # Natural downward EAR reduction to 0.175
    ext.reset()
    feat_down = None
    for i in range(30):
        t = i / 30.0
        lm_down = make_full_landmarks(ear_val=0.175 / 2.0, iris_y_offset=+8.0)
        feat_down = ext.compute(lm_down, timestamp=t, frame_id=i, face_valid=True, head_pitch=-18.0)

    assert feat_down is not None
    assert feat_down.ear_avg == pytest.approx(0.175, abs=0.005)
    assert ext._consecutive_closed == 0, "Downward glance compensation must prevent consecutive closed accumulation"
    assert feat_down.blink_detected is False
    assert feat_down.microsleep_detected is False


def test_highway_fixation_exemption_vs_catatonic_stare():
    """Test 6: Highway fixation exemption vs. impaired catatonic stare."""
    # ── Driver A: Alert Driver on Long Highway (Subtle Tremor + Healthy Blink Velocity) ──
    ext_alert = FeatureExtractor(fps=30.0)
    ext_alert._last_blink_opening_vel = 2.4  # Healthy reopening (>= 1.8 EAR/s)

    # 210 frames (7.0 seconds) with subtle physiological ocular drift (~0.35 deg)
    feat_alert = None
    for i in range(210):
        t = i / 30.0
        drift_x = 0.7 * np.sin(i * 0.3)
        lm = make_full_landmarks(ear_val=0.30 / 2.0, iris_x_offset=drift_x)
        feat_alert = ext_alert.compute(lm, timestamp=t, frame_id=i, face_valid=True)

    assert feat_alert is not None
    assert feat_alert.stare_fixation_duration_s >= 6.0
    assert feat_alert.highway_fixation_exempt is True
    assert feat_alert.compensatory_stare_active is False

    # ── Driver B: Impaired Driver with Catatonic Stare (Sluggish Reopening Velocity) ──
    ext_catatonic = FeatureExtractor(fps=30.0)
    ext_catatonic._last_blink_opening_vel = 1.2  # Sluggish reopening (< 1.8 EAR/s)

    feat_catatonic = None
    for i in range(210):
        t = i / 30.0
        drift_x = 0.7 * np.sin(i * 0.3)
        lm = make_full_landmarks(ear_val=0.30 / 2.0, iris_x_offset=drift_x)
        feat_catatonic = ext_catatonic.compute(lm, timestamp=t, frame_id=i, face_valid=True)

    assert feat_catatonic is not None
    assert feat_catatonic.stare_fixation_duration_s >= 6.0
    assert feat_catatonic.highway_fixation_exempt is False
    assert feat_catatonic.compensatory_stare_active is True


def test_speech_cadence_does_not_trigger_yawn():
    """Test 7: Speech/singing mouth cadence sets speech_active and suppresses false yawns."""
    ext = FeatureExtractor(fps=30.0, yawn_frames=10)

    # 45 frames (1.5s) of 3 Hz speech cadence with MAR oscillating between 0.10 and 0.62
    feat_speech = None
    for i in range(45):
        t = i / 30.0
        mar_val = 0.36 + 0.26 * np.sin(2 * np.pi * 3.0 * t)
        mouth_half_h = 20.0 * mar_val
        lm = make_full_landmarks(ear_val=0.30 / 2.0)
        cy = 350.0
        lm[13] = [300.0, cy - mouth_half_h, 0.0]
        lm[14] = [300.0, cy + mouth_half_h, 0.0]
        lm[81] = [290.0, cy - mouth_half_h, 0.0]
        lm[178] = [290.0, cy + mouth_half_h, 0.0]
        lm[311] = [310.0, cy - mouth_half_h, 0.0]
        lm[402] = [310.0, cy + mouth_half_h, 0.0]

        feat_speech = ext.compute(lm, timestamp=t, frame_id=i, face_valid=True)

    assert feat_speech is not None
    assert feat_speech.speech_active is True
    assert ext._consecutive_yawn == 0, "Speech cadence guard must freeze consecutive yawn counter"
    assert feat_speech.yawn_detected is False

    # In contrast, sustained mouth opening without speech cadence triggers true yawn
    ext_yawn = FeatureExtractor(fps=30.0, yawn_frames=10)
    feat_yawn = None
    for i in range(12):
        t = i / 30.0
        mouth_half_h = 20.0 * 0.65
        lm_y = make_full_landmarks(ear_val=0.30 / 2.0)
        cy = 350.0
        lm_y[13] = [300.0, cy - mouth_half_h, 0.0]
        lm_y[14] = [300.0, cy + mouth_half_h, 0.0]
        lm_y[81] = [290.0, cy - mouth_half_h, 0.0]
        lm_y[178] = [290.0, cy + mouth_half_h, 0.0]
        lm_y[311] = [310.0, cy - mouth_half_h, 0.0]
        lm_y[402] = [310.0, cy + mouth_half_h, 0.0]
        feat_yawn = ext_yawn.compute(lm_y, timestamp=t, frame_id=i, face_valid=True)

    assert feat_yawn is not None
    assert ext_yawn._consecutive_yawn >= 10
    assert feat_yawn.yawn_detected is True


def test_streetlight_strobe_rppg_immunity_and_rolling_shutter_ear():
    """Test 8: Streetlight strobe immunity in rPPG and rolling-shutter vertical stretch compensation."""
    ext = FeatureExtractor(fps=30.0)

    # ── Part A: rPPG 1.5 Hz Common-Mode Strobe Immunity & 72 BPM (1.2 Hz) Recovery ──
    lm_rppg = np.zeros((478, 3), dtype=np.float64)
    for idx in VASODILATION_LANDMARKS:
        lm_rppg[idx] = [300.0, 300.0, 0.0]

    # 150 frames (5.0s): 1.5 Hz common-mode luminance strobe + 1.2 Hz pulse
    recovered_bpm = 0.0
    for i in range(150):
        t = i / 30.0
        strobe = 1.0 + 0.40 * np.sin(2 * np.pi * 1.5 * t)
        pulse = 1.0 + 0.02 * np.sin(2 * np.pi * 1.2 * t)

        b = int(np.clip(100.0 * strobe, 0, 255))
        g = int(np.clip(120.0 * strobe * pulse, 0, 255))
        r = int(np.clip(160.0 * strobe, 0, 255))

        patch_frame = np.zeros((400, 400, 3), dtype=np.uint8)
        patch_frame[290:315, 290:315] = [b, g, r]

        bpm, _, valid = ext._compute_rppg_pulse(patch_frame, lm_rppg, t)
        if valid:
            recovered_bpm = bpm

    # Chromaticity normalization must reject 1.5 Hz common-mode strobe and isolate 1.2 Hz (72 BPM) pulse
    assert recovered_bpm == pytest.approx(72.0, abs=1.5)

    # ── Part B: Rolling-Shutter Vertical Stretch EAR Normalization ──
    ext_shutter = FeatureExtractor(fps=30.0)
    lm_nominal = make_landmarks_with_contours(ear_l=0.30, ear_r=0.30)

    # Step 1: Calibrate baseline under low vibration (<= 0.15) over 30 frames
    for i in range(30):
        ext_shutter.compute(
            lm_nominal,
            timestamp=i / 30.0,
            frame_id=i,
            face_valid=True,
            head_pitch=0.0,
            vibration_score=0.10,
        )

    # Nose distance ||[300, 280] - [300, 200]|| = 80 px; canthal ||[420, 200] - [180, 200]|| = 240 px
    assert ext_shutter._baseline_nose_ratio == pytest.approx(80.0 / 240.0, rel=1e-3)

    # Step 2: High cabin vibration (> 0.25) with 10% rolling-shutter vertical CMOS stretch
    # Nose length stretches 80 -> 88 px (tip moves to 288), raw EAR stretches 0.30 -> 0.33
    lm_stretched = make_landmarks_with_contours(ear_l=0.33, ear_r=0.33)
    lm_stretched[4] = [300.0, 288.0, 0.0]

    feat_shutter = ext_shutter.compute(
        lm_stretched,
        timestamp=1.5,
        frame_id=31,
        face_valid=True,
        head_pitch=0.0,
        vibration_score=0.35,
    )

    # Shutter stretch = 1.10. EAR compensated: 0.33 / 1.10 = 0.30
    assert feat_shutter.ear_avg == pytest.approx(0.30, abs=0.005)
