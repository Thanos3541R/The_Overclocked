"""Export Driver Monitoring System feature extraction output to text and JSON formats."""
import cv2
import json
import numpy as np
from datetime import datetime

from pipeline.preprocessing import LightingNormalizer
from pipeline.face_mesh_detector import FaceMeshDetector
from pipeline.pnp_normalizer import PnPNormalizer
from pipeline.feature_extractor import FeatureExtractor
from pipeline.classifier import ImpairmentClassifier

def main():
    image_path = 'captured_snapshot.jpg'
    frame = cv2.imread(image_path)
    if frame is None:
        print(f"Error: {image_path} not found.")
        return

    h, w = frame.shape[:2]

    # Initialize full pipeline
    prep = LightingNormalizer()
    detector = FaceMeshDetector()
    pnp = PnPNormalizer(w, h)
    classifier = ImpairmentClassifier()
    extractor = FeatureExtractor()

    # Process frame
    processed = prep.process(frame)
    landmarks = detector.detect(processed)
    if landmarks is None:
        print("Error: No face detected.")
        return

    pnp.solve_pose(landmarks)
    euler = pnp.get_euler_angles()
    pitch, yaw, roll = euler if euler is not None else (0.0, 0.0, 0.0)

    # Compute complete FrameFeatures
    features = extractor.compute(
        landmarks=landmarks,
        timestamp=0.0,
        frame_id=1,
        face_valid=True,
        head_yaw=yaw,
        head_pitch=pitch,
        head_roll=roll,
        imu_gated=False,
        gating_reason='',
        frame=frame
    )

    # Diagnostic evaluation
    assessment = classifier.evaluate_static_image(
        frame=frame,
        landmarks=landmarks,
        ear_left=features.ear_left,
        ear_right=features.ear_right,
        mar=features.mar,
        head_pitch=pitch,
        head_roll=roll
    )
    sclera_redness = classifier._extract_sclera_redness(frame, landmarks)
    raw_asym = abs(features.ear_left - features.ear_right)

    # Build human-readable formatted report
    lines = []
    lines.append("=" * 80)
    lines.append("       DRIVER MONITORING SYSTEM (DMS) — FEATURE EXTRACTION REPORT")
    lines.append("=" * 80)
    lines.append(f"Generated At          : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"Input Source File     : {image_path} ({w}x{h} px)")
    lines.append(f"Face Mesh Status      : DETECTED (478 3D landmarks with iris refinement)")
    lines.append(f"Overall Classification: {assessment.classification}")
    lines.append(f"Impairment Risk Score : {assessment.risk_score:.1f}%")
    lines.append(f"Diagnostic Confidence : {assessment.confidence * 100:.1f}%")
    lines.append("-" * 80)
    lines.append("")

    lines.append("1. EYE & DROWSINESS BIOMARKERS (EAR & CLOSURE)")
    lines.append(f"  * Left Eye Aspect Ratio (EAR)       : {features.ear_left:.4f}      [Normal Awake: 0.28 - 0.35]")
    lines.append(f"  * Right Eye Aspect Ratio (EAR)      : {features.ear_right:.4f}      [Normal Awake: 0.28 - 0.35]")
    lines.append(f"  * Average Eye Aspect Ratio (EAR)    : {features.ear_avg:.4f}      [Droop Alert: < 0.22, Closed: < 0.12]")
    lines.append(f"  * Eyelid Asymmetry (|dEAR|)         : {raw_asym:.4f}      [Normal: < 0.035, Baseline Subtracted: 0.000]")
    lines.append(f"  * Instantaneous Blink Detected      : {features.blink_detected}")
    lines.append(f"  * Microsleep Episode Detected       : {features.microsleep_detected}     [Threshold: >= 15 frames / 500ms]")
    lines.append(f"  * PERCLOS (% Eyes Closed Window)    : {features.perclos * 100:.2f}%      [Drowsiness Threshold: > 15.0%]")
    lines.append("")

    lines.append("2. MOUTH & YAWN BIOMARKERS (MAR)")
    lines.append(f"  * Mouth Aspect Ratio (MAR)          : {features.mar:.4f}      [Closed Lip: < 0.20, Yawn Alert: > 0.60]")
    lines.append(f"  * Yawn Episode Detected             : {features.yawn_detected}     [Threshold: MAR > 0.60 for >= 30 frames]")
    lines.append("")

    lines.append("3. 3D HEAD POSE & CERVICAL TONE (PnP)")
    lines.append(f"  * Head Pitch (Nodding / Slump)      : {features.head_pitch:+.2f} deg   [Slump Alert: < -14.0 deg]")
    lines.append(f"  * Head Yaw (Horizontal Turn)        : {features.head_yaw:+.2f} deg   [FOV Gating Limit: +/- 35.0 deg]")
    lines.append(f"  * Head Roll (Lateral Tilt)          : {features.head_roll:+.2f} deg   [Tilt Alert: > 14.0 deg]")
    lines.append("")

    lines.append("4. IRIS CENTROID & GAZE TRACKING (NORMALIZED [-1.0, 1.0])")
    lines.append(f"  * Left Iris Offset (X, Y)           : ({features.left_iris_x:+.4f}, {features.left_iris_y:+.4f})")
    lines.append(f"  * Right Iris Offset (X, Y)          : ({features.right_iris_x:+.4f}, {features.right_iris_y:+.4f})")
    lines.append("")

    lines.append("5. ALCOHOL-DISCRIMINATING OCULOMOTOR KINEMATICS (Fransson et al. 2010)")
    lines.append(f"  * Saccadic Angular Velocity         : {features.saccade_velocity:.2f} deg/s   [Normal Peak: 200 - 900 deg/s]")
    lines.append(f"  * Saccade In-Progress Trigger       : {features.saccade_detected}     [Hysteresis: Onset > 30, Offset < 15]")
    lines.append(f"  * Saccade Fixation Latency          : {features.saccade_latency_ms:.1f} ms      [Inter-saccadic reaction time]")
    lines.append(f"  * Blink Downstroke Velocity         : {features.blink_closing_velocity:.3f} EAR/s   [Involuntary eyelid downstroke]")
    lines.append(f"  * Blink Upstroke Reopening Velocity : {features.blink_opening_velocity:.3f} EAR/s   [Involuntary levator upstroke]")
    lines.append(f"  * Gaze Yaw Dispersion (sigma_yaw)   : {features.gaze_yaw_dispersion:.4f}        [Road Tunneling Alert: < 0.030]")
    lines.append("")

    lines.append("6. SECTION A — INVOLUNTARY BIOMARKERS (UNFAKEABLE REFLEXES)")
    lines.append(f"  * Gaze-Evoked Nystagmus (GEN)       : {features.gen_detected}     [Romano et al. 2017]")
    lines.append(f"  * GEN Corrective Fast Beat Count    : {features.gen_beat_count}")
    lines.append(f"  * GEN Centripetal Drift Speed       : {features.gen_slow_phase_vel:.2f} deg/s   [Diagnostic Slip: 2.0 - 18.0 deg/s]")
    lines.append(f"  * Smooth Pursuit Fragmentation      : {features.pursuit_fragmentation_ratio * 100:.1f}%      [Normal: ~0%, Impaired: > 20%]")
    lines.append(f"  * Binocular Vergence Angle          : {features.vergence_angle_deg:+.2f} deg  [Positive = Convergence]")
    lines.append(f"  * Lack of Convergence (LOC - DRE)   : {features.lack_of_convergence}     [Strabismus Alert: < -3.5 deg]")
    lines.append(f"  * VOR Micro-Compensation Gain       : {features.vor_gain:.3f}        [0% lockout weight - Tier 3 experimental]")
    lines.append(f"  * Head Postural Sway (Micro-Tremor) : {features.head_postural_sway:.2f} deg       [0.5-2.0 Hz road de-trended sway]")
    lines.append(f"  * Facial Flushing / Vasodilation    : {features.facial_flushing_ratio:.3f}        [Valid: {features.flushing_valid}, Delta: {features.flushing_delta*100:+.1f}%]")
    lines.append("")

    lines.append("7. SECTION B — VOLUNTARY & MASKABLE BEHAVIORS (GAMING INDICATORS)")
    lines.append(f"  * Deliberate Blink Suppression      : {features.deliberate_blink_suppression}     [Inter-blink interval > 6.0s]")
    lines.append(f"  * Voluntary Eye-Widening Spike      : {features.voluntary_eye_widening}     [Frontalis/levator contraction]")
    lines.append(f"  * Compensatory Stare Duration       : {features.stare_fixation_duration_s:.1f} s       [Active: {features.compensatory_stare_active} (>6.0s, sigma<0.8 deg)]")
    lines.append(f"  * Rigid Head Stabilization Effort   : {features.rigid_head_stabilization}     [Conscious neck muscle locking]")
    lines.append("")

    lines.append("8. ANTI-MASKING DIVERGENCE ANALYSIS (M_mask)")
    lines.append(f"  * Masking Divergence Score (M_mask) : {features.masking_divergence_score:.1f}%      [Biophysical kinetic discordance]")
    lines.append(f"  * Active Masking Attempt Detected   : {features.masking_detected}     [Threshold: M_mask >= 50.0%]")
    lines.append("")

    lines.append("9. SCLERA VASCULAR INJECTION (BLOODSHOT INDEX)")
    lines.append(f"  * Sclera Redness Ratio (R/(G+B))    : {sclera_redness:.3f}        [Normal: ~1.00, Bloodshot: > 1.20, Neutral on NIR]")
    lines.append("")

    lines.append("10. NOISE MITIGATION & VEHICLE GATING STATE")
    lines.append(f"  * Head Pose Valid (Within Limits)   : {features.face_valid}")
    lines.append(f"  * Software IMU Pothole Gated        : {features.imu_gated}     [Linear Accel > 2.5 m/s2 shock]")
    lines.append(f"  * Active Gating Reason              : {features.gating_reason if features.gating_reason else 'None (Operating Normally)'}")
    lines.append("=" * 80)
    lines.append("")

    # Machine-learning tensor dictionary
    raw_dict = {
        'timestamp': float(features.timestamp),
        'frame_id': int(features.frame_id),
        'ear_left': float(features.ear_left),
        'ear_right': float(features.ear_right),
        'ear_avg': float(features.ear_avg),
        'mar': float(features.mar),
        'perclos': float(features.perclos),
        'left_iris_x': float(features.left_iris_x),
        'left_iris_y': float(features.left_iris_y),
        'right_iris_x': float(features.right_iris_x),
        'right_iris_y': float(features.right_iris_y),
        'head_yaw': float(features.head_yaw),
        'head_pitch': float(features.head_pitch),
        'head_roll': float(features.head_roll),
        'saccade_velocity': float(features.saccade_velocity),
        'saccade_latency_ms': float(features.saccade_latency_ms),
        'blink_closing_velocity': float(features.blink_closing_velocity),
        'blink_opening_velocity': float(features.blink_opening_velocity),
        'gaze_yaw_dispersion': float(features.gaze_yaw_dispersion),
        'gen_slow_phase_vel': float(features.gen_slow_phase_vel),
        'gen_beat_count': int(features.gen_beat_count),
        'pursuit_fragmentation_ratio': float(features.pursuit_fragmentation_ratio),
        'vergence_angle_deg': float(features.vergence_angle_deg),
        'vor_gain': float(features.vor_gain),
        'head_postural_sway': float(features.head_postural_sway),
        'facial_flushing_ratio': float(features.facial_flushing_ratio),
        'flushing_valid': bool(features.flushing_valid),
        'flushing_delta': float(features.flushing_delta),
        'deliberate_blink_suppression': bool(features.deliberate_blink_suppression),
        'voluntary_eye_widening': bool(features.voluntary_eye_widening),
        'stare_fixation_duration_s': float(features.stare_fixation_duration_s),
        'compensatory_stare_active': bool(features.compensatory_stare_active),
        'rigid_head_stabilization': bool(features.rigid_head_stabilization),
        'masking_divergence_score': float(features.masking_divergence_score),
        'masking_detected': bool(features.masking_detected),
        'blink_detected': bool(features.blink_detected),
        'microsleep_detected': bool(features.microsleep_detected),
        'yawn_detected': bool(features.yawn_detected),
        'saccade_detected': bool(features.saccade_detected),
        'gen_detected': bool(features.gen_detected),
        'lack_of_convergence': bool(features.lack_of_convergence),
        'imu_gated': bool(features.imu_gated),
        'risk_score': float(assessment.risk_score),
        'classification': assessment.classification,
        'confidence': float(assessment.confidence)
    }

    lines.append("11. RAW MACHINE-LEARNING TENSOR RECORD (JSON FORMAT)")
    lines.append(json.dumps(raw_dict, indent=2))
    lines.append("")
    lines.append("=" * 80)

    # Save to text file
    txt_path = 'feature_extraction_output.txt'
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    print(f"Generated text report: {txt_path}")

    # Save to JSON file
    json_path = 'feature_extraction_output.json'
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(raw_dict, f, indent=2)
    print(f"Generated JSON record: {json_path}")

    # Save to CSV row
    csv_path = 'feature_extraction_output.csv'
    keys = list(raw_dict.keys())
    values = [str(v) for v in raw_dict.values()]
    with open(csv_path, 'w', encoding='utf-8') as f:
        f.write(','.join(keys) + '\n')
        f.write(','.join(values) + '\n')
    print(f"Generated CSV table : {csv_path}")

if __name__ == '__main__':
    main()
