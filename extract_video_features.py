"""Automated Video Feature Extraction & Anti-Masking Analysis Engine.

Processes video files (MP4, AVI, MOV, MKV) or camera streams through the complete
Driver Monitoring System (DMS) pipeline to extract:
  1. Section A — Involuntary Changes (GEN, VOR gain, pursuit fragmentation, LOC,
     blink upstroke velocity, low-frequency postural sway, facial flushing).
  2. Section B — Voluntary & Semi-Voluntary Changes (deliberate blink suppression,
     voluntary eye-widening spikes, compensatory staring fixation, rigid head stabilization).
  3. Anti-Masking Divergence Metric (M_mask) & Lockout Recommendation.

Outputs generated:
  - video_features_report.txt : Comprehensive formatted forensic & clinical report
  - video_features.csv        : Time-series table of all frame-by-frame features
  - video_features_summary.json : Machine-readable aggregate metrics & ML tensor payload
"""
import argparse
import os
import sys
import time
import json
from datetime import datetime
from typing import Dict, List, Optional, Any
import cv2
import numpy as np

from pipeline.preprocessing import LightingNormalizer
from pipeline.face_mesh_detector import FaceMeshDetector
from pipeline.pnp_normalizer import PnPNormalizer
from pipeline.feature_extractor import FeatureExtractor, FrameFeatures
from pipeline.classifier import ImpairmentClassifier, ImpairmentAssessment
from pipeline.calibration import PersonalBaselineCalibrator
from pipeline.bayesian_filter import BayesianEvidenceAccumulator


def extract_video_features(video_path: str,
                           output_dir: str = ".",
                           max_frames: Optional[int] = None,
                           headless: bool = True) -> Dict[str, Any]:
    """Process a video file and extract voluntary, involuntary, and anti-masking features.

    Args:
        video_path: Path to video file (or camera integer string '0').
        output_dir: Directory to save generated output files.
        max_frames: Optional maximum frames to process.
        headless: If False, displays OpenCV visualization window during processing.

    Returns:
        Summary dictionary of video analysis.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Open video capture
    is_cam = video_path.isdigit()
    cap = cv2.VideoCapture(int(video_path) if is_cam else video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video source: {video_path}")

    # Video properties
    total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if not is_cam else -1
    source_fps = cap.get(cv2.CAP_PROP_FPS)
    fps = source_fps if source_fps > 5.0 else 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration_sec = total_video_frames / fps if total_video_frames > 0 else 0.0

    print(f"\n{'='*75}")
    print(f"  VIDEO FEATURE EXTRACTION & ANTI-MASKING ANALYSIS")
    print(f"{'='*75}")
    print(f"  Source Video       : {video_path}")
    print(f"  Resolution         : {width} x {height}")
    print(f"  Native Frame Rate  : {fps:.2f} FPS")
    print(f"  Total Frames       : {total_video_frames if total_video_frames > 0 else 'Live Stream'}")
    print(f"  Estimated Duration : {duration_sec:.1f} seconds" if duration_sec > 0 else "")
    print(f"  Output Directory   : {os.path.abspath(output_dir)}")
    print(f"{'-'*75}")

    # Initialize DMS pipeline components
    prep = LightingNormalizer()
    detector = FaceMeshDetector()
    pnp = PnPNormalizer(width, height)
    extractor = FeatureExtractor(fps=fps)
    classifier = ImpairmentClassifier()
    calibrator = PersonalBaselineCalibrator(calibration_duration_sec=30.0, fps=fps)
    bayesian_filter = BayesianEvidenceAccumulator(lockout_sustained_sec=15.0)

    frame_records: List[Dict[str, Any]] = []
    classifications: List[str] = []
    lockout_triggers = 0
    bayesian_lockout_triggers = 0
    masking_triggers = 0
    face_detected_frames = 0
    processed_count = 0
    start_time = time.time()

    prev_pts = 0.0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            processed_count += 1
            if max_frames and processed_count > max_frames:
                break

            timestamp = (processed_count - 1) / fps

            # 1. Illumination preprocessing
            processed_frame = prep.process(frame)

            # 2. 478 3D facial landmark detection with iris refinement
            landmarks = detector.detect(processed_frame)
            face_valid = landmarks is not None

            if face_valid:
                face_detected_frames += 1
                # 3. PnP 3D head pose estimation
                pnp.solve_pose(landmarks)
                euler = pnp.get_euler_angles()
                pitch, yaw, roll = euler if euler is not None else (0.0, 0.0, 0.0)

                # 4. Feature extraction (including Section A involuntary, Section B voluntary, and M_mask)
                features = extractor.compute(
                    landmarks=landmarks,
                    timestamp=timestamp,
                    frame_id=processed_count,
                    face_valid=True,
                    head_yaw=yaw,
                    head_pitch=pitch,
                    head_roll=roll,
                    imu_gated=False,
                    gating_reason='',
                    frame=frame
                )

                # Online Personal Baseline Auto-Calibration
                calibrator.update(features)
                if calibrator.is_calibrated:
                    calibrator.apply_to(extractor, classifier)
            else:
                features = extractor.compute(
                    landmarks=np.zeros((478, 3)),
                    timestamp=timestamp,
                    frame_id=processed_count,
                    face_valid=False,
                    gating_reason='No face detected',
                    frame=frame
                )

            # 5. Temporal classification and anti-masking assessment
            assessment = classifier.evaluate_temporal_stream(features)
            classifications.append(assessment.classification)

            # 6. Sequential Bayesian Evidence Accumulation (ISO 26262 ASIL-B/D)
            b_state = bayesian_filter.update(assessment, timestamp)

            if assessment.lockout_recommended:
                lockout_triggers += 1
            if b_state.lockout_confirmed:
                bayesian_lockout_triggers += 1
            if features.masking_detected:
                masking_triggers += 1

            # Build record
            rec = {
                'frame_id': processed_count,
                'timestamp': round(timestamp, 3),
                'face_valid': face_valid,
                # Ocular & Facial
                'ear_avg': round(features.ear_avg, 4),
                'ear_left': round(features.ear_left, 4),
                'ear_right': round(features.ear_right, 4),
                'mar': round(features.mar, 4),
                'perclos': round(features.perclos, 4),
                # 3D Head Pose
                'head_yaw': round(features.head_yaw, 2),
                'head_pitch': round(features.head_pitch, 2),
                'head_roll': round(features.head_roll, 2),
                # Section A — Involuntary Biomarkers
                'gen_detected': bool(features.gen_detected),
                'gen_beat_count': int(features.gen_beat_count),
                'gen_slow_phase_vel': round(features.gen_slow_phase_vel, 2),
                'pursuit_fragmentation_ratio': round(features.pursuit_fragmentation_ratio, 4),
                'vor_gain': round(features.vor_gain, 3),
                'vergence_angle_deg': round(features.vergence_angle_deg, 2),
                'lack_of_convergence': bool(features.lack_of_convergence),
                'blink_closing_velocity': round(features.blink_closing_velocity, 3),
                'blink_opening_velocity': round(features.blink_opening_velocity, 3),
                'head_postural_sway': round(features.head_postural_sway, 3),
                'facial_flushing_ratio': round(features.facial_flushing_ratio, 3),
                'flushing_valid': bool(features.flushing_valid),
                'flushing_delta': round(features.flushing_delta, 3),
                'gaze_yaw_dispersion': round(features.gaze_yaw_dispersion, 4),
                # Section B — Voluntary & Masking Behaviors
                'deliberate_blink_suppression': bool(features.deliberate_blink_suppression),
                'voluntary_eye_widening': bool(features.voluntary_eye_widening),
                'stare_fixation_duration_s': round(features.stare_fixation_duration_s, 2),
                'compensatory_stare_active': bool(features.compensatory_stare_active),
                'rigid_head_stabilization': bool(features.rigid_head_stabilization),
                # Anti-Masking Divergence (M_mask)
                'masking_divergence_score': round(features.masking_divergence_score, 1),
                'masking_detected': bool(features.masking_detected),
                # Diagnosis & Audit
                'risk_score': round(assessment.risk_score, 1),
                'involuntary_risk_score': round(assessment.involuntary_risk_score, 1),
                'voluntary_masking_score': round(assessment.voluntary_masking_score, 1),
                'classification': assessment.classification,
                'lockout_recommended': bool(assessment.lockout_recommended),
                # ISO 26262 Bayesian Filter
                'bayesian_posterior': round(b_state.posterior_probability, 4),
                'bayesian_sustained_s': round(b_state.sustained_high_confidence_s, 2),
                'bayesian_lockout_confirmed': bool(b_state.lockout_confirmed),
            }
            frame_records.append(rec)

            if processed_count % 30 == 0 or processed_count == total_video_frames:
                pct = (processed_count / total_video_frames * 100) if total_video_frames > 0 else 0
                print(f"  Frame {processed_count:5d}/{total_video_frames if total_video_frames > 0 else '?':5} "
                      f"({pct:5.1f}%) | Risk: {assessment.risk_score:5.1f}% | "
                      f"Invol Risk: {assessment.involuntary_risk_score:5.1f}% | "
                      f"M_mask: {features.masking_divergence_score:4.1f}% | "
                      f"State: {assessment.classification:<22} | "
                      f"Lockout: {assessment.lockout_recommended} | "
                      f"Bayes Confirmed: {b_state.lockout_confirmed}")

    finally:
        cap.release()
        cv2.destroyAllWindows()

    elapsed = time.time() - start_time
    proc_fps = processed_count / elapsed if elapsed > 0 else 0.0
    print(f"{'-'*75}")
    print(f"  Processing Complete: {processed_count} frames in {elapsed:.2f}s ({proc_fps:.1f} FPS)")

    # Aggregate Statistics
    sober_count = classifications.count("SOBER")
    mild_count = classifications.count("MILD_IMPAIRMENT")
    high_risk_count = classifications.count("HIGH_RISK_INTOXICATED")

    valid_records = [r for r in frame_records if r['face_valid']]
    n_valid = max(len(valid_records), 1)

    mean_ear = float(np.mean([r['ear_avg'] for r in valid_records])) if valid_records else 0.0
    mean_mar = float(np.mean([r['mar'] for r in valid_records])) if valid_records else 0.0
    mean_vor = float(np.mean([r['vor_gain'] for r in valid_records])) if valid_records else 1.0
    mean_pursuit = float(np.mean([r['pursuit_fragmentation_ratio'] for r in valid_records])) if valid_records else 0.0
    mean_sway = float(np.mean([r['head_postural_sway'] for r in valid_records])) if valid_records else 0.0
    mean_flushing = float(np.mean([r['facial_flushing_ratio'] for r in valid_records])) if valid_records else 1.0
    max_m_mask = float(np.max([r['masking_divergence_score'] for r in valid_records])) if valid_records else 0.0
    mean_m_mask = float(np.mean([r['masking_divergence_score'] for r in valid_records])) if valid_records else 0.0
    max_invol_risk = float(np.max([r['involuntary_risk_score'] for r in valid_records])) if valid_records else 0.0
    mean_invol_risk = float(np.mean([r['involuntary_risk_score'] for r in valid_records])) if valid_records else 0.0

    gen_frames = sum(1 for r in valid_records if r['gen_detected'])
    loc_frames = sum(1 for r in valid_records if r['lack_of_convergence'])
    blink_suppress_frames = sum(1 for r in valid_records if r['deliberate_blink_suppression'])
    eye_widening_frames = sum(1 for r in valid_records if r['voluntary_eye_widening'])
    rigid_head_frames = sum(1 for r in valid_records if r['rigid_head_stabilization'])
    max_stare_duration = float(np.max([r['stare_fixation_duration_s'] for r in valid_records])) if valid_records else 0.0

    # Primary Lockout Determination
    lockout_triggered = (lockout_triggers >= 5) or (high_risk_count / n_valid >= 0.10)
    final_classification = "HIGH_RISK_INTOXICATED" if lockout_triggered else (
        "MILD_IMPAIRMENT" if (mild_count / n_valid >= 0.25 or mean_invol_risk >= 25.0) else "SOBER"
    )

    summary_data = {
        'metadata': {
            'video_path': os.path.abspath(video_path),
            'analysis_timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'resolution': f"{width}x{height}",
            'source_fps': round(fps, 2),
            'total_processed_frames': processed_count,
            'face_detected_frames': face_detected_frames,
            'face_detection_rate_pct': round((face_detected_frames / processed_count * 100), 1) if processed_count > 0 else 0.0,
            'processing_time_sec': round(elapsed, 2),
            'processing_speed_fps': round(proc_fps, 1),
        },
        'overall_verdict': {
            'final_classification': final_classification,
            'lockout_triggered': lockout_triggered,
            'lockout_trigger_frames': lockout_triggers,
            'lockout_trigger_pct': round((lockout_triggers / n_valid * 100), 1),
            'bayesian_lockout_confirmed': bool(bayesian_lockout_triggers >= 1),
            'bayesian_lockout_frames': bayesian_lockout_triggers,
            'personal_baseline_calibrated': bool(calibrator.is_calibrated),
            'mean_involuntary_risk_pct': round(mean_invol_risk, 1),
            'max_involuntary_risk_pct': round(max_invol_risk, 1),
            'mean_masking_divergence_pct': round(mean_m_mask, 1),
            'max_masking_divergence_pct': round(max_m_mask, 1),
            'masking_detected_frames': masking_triggers,
        },
        'section_a_involuntary_biomarkers': {
            'mean_vor_gain': round(mean_vor, 3),
            'mean_pursuit_fragmentation_ratio': round(mean_pursuit, 4),
            'gaze_evoked_nystagmus_frames': gen_frames,
            'lack_of_convergence_frames': loc_frames,
            'mean_head_postural_sway_deg': round(mean_sway, 3),
            'mean_facial_flushing_ratio': round(mean_flushing, 3),
            'mean_ear': round(mean_ear, 4),
        },
        'section_b_voluntary_masking_biomarkers': {
            'deliberate_blink_suppression_frames': blink_suppress_frames,
            'voluntary_eye_widening_spikes': eye_widening_frames,
            'max_compensatory_stare_duration_s': round(max_stare_duration, 2),
            'rigid_head_stabilization_frames': rigid_head_frames,
        },
        'classification_breakdown': {
            'sober_frames': sober_count,
            'mild_impairment_frames': mild_count,
            'high_risk_intoxicated_frames': high_risk_count,
        }
    }

    # ── Write CSV Time-Series File ──
    csv_path = os.path.join(output_dir, "video_features.csv")
    if frame_records:
        keys = list(frame_records[0].keys())
        with open(csv_path, 'w', encoding='utf-8') as f:
            f.write(','.join(keys) + '\n')
            for r in frame_records:
                f.write(','.join(str(r[k]) for k in keys) + '\n')
    print(f"  [Output] CSV Table Saved        : {csv_path}")

    # ── Write JSON Summary File ──
    json_path = os.path.join(output_dir, "video_features_summary.json")
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(summary_data, f, indent=2)
    print(f"  [Output] JSON Summary Saved     : {json_path}")

    # ── Write Formatted Text Report ──
    txt_path = os.path.join(output_dir, "video_features_report.txt")
    report_lines = _generate_text_report(summary_data, valid_records)
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report_lines))
    print(f"  [Output] Formatted Report Saved : {txt_path}")
    print(f"{'='*75}\n")

    return summary_data


def _generate_text_report(summary: Dict[str, Any], records: List[Dict[str, Any]]) -> List[str]:
    """Generate structured human-readable text report separating Section A vs Section B."""
    m = summary['metadata']
    v = summary['overall_verdict']
    sa = summary['section_a_involuntary_biomarkers']
    sb = summary['section_b_voluntary_masking_biomarkers']
    cb = summary['classification_breakdown']

    lines = []
    lines.append("=" * 85)
    lines.append("       DRIVER MONITORING SYSTEM (DMS) — VIDEO FEATURE EXTRACTION REPORT")
    lines.append("     Passive Sensing & Anti-Masking Involuntary/Voluntary Analysis Engine")
    lines.append("=" * 85)
    lines.append(f"Generated At           : {m['analysis_timestamp']}")
    lines.append(f"Source Video File      : {m['video_path']}")
    lines.append(f"Resolution & FPS       : {m['resolution']} @ {m['source_fps']} FPS")
    lines.append(f"Total Frames Analyzed  : {m['total_processed_frames']} frames ({m['processing_time_sec']}s wall time)")
    lines.append(f"Face Tracking Health   : {m['face_detected_frames']}/{m['total_processed_frames']} frames tracked ({m['face_detection_rate_pct']}%)")
    lines.append(f"Baseline Auto-Calib    : {'Calibrated (Driver-Specific Baselines Active)' if v.get('personal_baseline_calibrated') else 'Uncalibrated (Population Defaults)'}")
    lines.append("-" * 85)
    lines.append(f"OVERALL CLASSIFICATION : {v['final_classification']}")
    lines.append(f"INSTANTANEOUS LOCKOUT  : {'TRIGGERED' if v['lockout_triggered'] else 'PERMISSIVE (NO LOCKOUT)'} ({v['lockout_trigger_frames']} frames, {v['lockout_trigger_pct']}%)")
    lines.append(f"ISO 26262 BAYESIAN     : {'>>> SUSTAINED VEHICLE INTERLOCK LOCKOUT ARMED <<<' if v.get('bayesian_lockout_confirmed') else 'PERMISSIVE (Under Confirmation Horizon)'}")
    lines.append(f"Mean Involuntary Risk  : {v['mean_involuntary_risk_pct']}% (Peak: {v['max_involuntary_risk_pct']}%)")
    lines.append(f"Anti-Masking M_mask    : Mean {v['mean_masking_divergence_pct']}%, Peak {v['max_masking_divergence_pct']}% ({v['masking_detected_frames']} masking events)")
    lines.append("=" * 85)
    lines.append("")

    lines.append("SECTION A — INVOLUNTARY BIOMARKERS (HIGH EVIDENTIAL WEIGHT FOR LOCKOUT)")
    lines.append("Physiological, autonomic, or brainstem reflexes that a driver cannot consciously fake or suppress.")
    lines.append("-" * 85)
    lines.append(f"  1. Gaze-Evoked Nystagmus (GEN) : {sa['gaze_evoked_nystagmus_frames']} frames detected")
    lines.append(f"     * Diagnostic Significance   : Brainstem/cerebellar neural integrator failure at lateral gaze.")
    lines.append(f"  2. Vestibulo-Ocular Reflex     : Mean VOR Micro-Gain = {sa['mean_vor_gain']:.3f}")
    lines.append(f"     * Functional Safety Status  : Demoted to 0% weight in primary lockout (Tier 3 experimental proxy).")
    lines.append(f"  3. Smooth Pursuit Breakdown    : Mean Fragmentation Ratio = {sa['mean_pursuit_fragmentation_ratio']*100:.1f}%  [Normal: < 10%]")
    lines.append(f"     * Diagnostic Significance   : Intrusion of catch-up saccades disrupting smooth visual pursuit.")
    lines.append(f"  4. Lack of Convergence (LOC)   : {sa['lack_of_convergence_frames']} divergent strabismus frames")
    lines.append(f"     * Diagnostic Significance   : Standard DRE ocular sign of central nervous system depression.")
    lines.append(f"  5. Involuntary Postural Sway   : Mean Sway = {sa['mean_head_postural_sway_deg']:.2f} deg  [Alert: > 1.40 deg]")
    lines.append(f"     * Diagnostic Significance   : Bandpass de-trended (0.5 - 2.0 Hz) micro-tremor rejecting road terrain.")
    lines.append(f"  6. Facial Flushing / Perfusion : Mean Chromaticity Ratio = {sa['mean_facial_flushing_ratio']:.3f}  [Normal: ~1.00]")
    lines.append(f"     * Diagnostic Significance   : Evaluates relative cheek delta >= +20% on RGB; cleanly disabled on NIR.")
    lines.append(f"  7. Palpebral Aperture & EAR    : Mean EAR = {sa['mean_ear']:.4f}  [Normal awake: 0.28 - 0.35]")
    lines.append(f"     * Diagnostic Significance   : Ptosis and levator motor neuron inhibition.")
    lines.append("")

    lines.append("SECTION B — VOLUNTARY & SEMI-VOLUNTARY BEHAVIORS (CORROBORATING / GAMING)")
    lines.append("Consciously alterable behaviors susceptible to gaming, used for corroboration and mismatch detection.")
    lines.append("-" * 85)
    lines.append(f"  1. Deliberate Blink Suppression: {sb['deliberate_blink_suppression_frames']} frames (>6s inter-blink interval)")
    lines.append(f"     * Mechanism                 : Driver consciously fighting eyelid droop by holding eyes open.")
    lines.append(f"  2. Voluntary Eye-Widening Spike: {sb['voluntary_eye_widening_spikes']} frontalis/levator contraction surges")
    lines.append(f"     * Mechanism                 : Transient eyebrow and lid raise to temporarily counter ptosis.")
    lines.append(f"  3. Compensatory Stare Duration : Peak unbroken gaze = {sb['max_compensatory_stare_duration_s']:.1f} seconds")
    lines.append(f"     * Mechanism                 : Rigid unbroken gaze (>6.0s with low spatial variance sigma<0.8 deg).")
    lines.append(f"  4. Rigid Head Stabilization    : {sb['rigid_head_stabilization_frames']} frames unnatural neck tension")
    lines.append(f"     * Mechanism                 : Tensing cervical muscles to artificially suppress natural head sway.")
    lines.append("")

    lines.append("ANTI-MASKING DIVERGENCE METRIC (M_mask) ANALYSIS")
    lines.append("Models biophysical kinetic discordance: (f_blink / f_baseline) x (v_baseline_up / v_up).")
    lines.append("-" * 85)
    lines.append(f"  * Biophysical Principle        : Normal blink rate cannot mask pharmacologically slowed levator upstroke.")
    lines.append(f"  * Average M_mask Divergence    : {v['mean_masking_divergence_pct']}%")
    lines.append(f"  * Peak M_mask Divergence       : {v['max_masking_divergence_pct']}%")
    lines.append(f"  * Discordant Masking Alerts    : {v['masking_detected_frames']} frames with M_mask >= 50.0%")
    lines.append(f"  * Anti-Masking Verdict         : " + (
        "CONFIRMED GAMING ATTEMPT (Driver actively fighting observable impairment)"
        if v['masking_detected_frames'] >= 5 else "NO SYSTEMATIC GAMING DETECTED"
    ))
    lines.append("")

    lines.append("TIME-SERIES CLASSIFICATION DISTRIBUTION")
    lines.append("-" * 85)
    lines.append(f"  * SOBER Frames                 : {cb['sober_frames']:5d} ({cb['sober_frames']/m['total_processed_frames']*100:.1f}%)")
    lines.append(f"  * MILD_IMPAIRMENT Frames       : {cb['mild_impairment_frames']:5d} ({cb['mild_impairment_frames']/m['total_processed_frames']*100:.1f}%)")
    lines.append(f"  * HIGH_RISK_INTOXICATED Frames : {cb['high_risk_intoxicated_frames']:5d} ({cb['high_risk_intoxicated_frames']/m['total_processed_frames']*100:.1f}%)")
    lines.append("=" * 85)
    lines.append("END OF REPORT — Generated by Driver Monitoring System Passive Sensing Engine")
    return lines


def main():
    parser = argparse.ArgumentParser(
        description="Extract Voluntary, Involuntary, and Anti-Masking Features from Driver Video"
    )
    parser.add_argument("--video", type=str, required=True,
                        help="Path to input video file (e.g. driver_test.mp4) or webcam ID ('0')")
    parser.add_argument("--output-dir", type=str, default=".",
                        help="Directory to write output files (default: current directory)")
    parser.add_argument("--max-frames", type=int, default=None,
                        help="Maximum frames to process (optional)")
    parser.add_argument("--show", action="store_true", default=False,
                        help="Show OpenCV visualization window during extraction")

    args = parser.parse_args()

    extract_video_features(
        video_path=args.video,
        output_dir=args.output_dir,
        max_frames=args.max_frames,
        headless=not args.show
    )


if __name__ == '__main__':
    main()
