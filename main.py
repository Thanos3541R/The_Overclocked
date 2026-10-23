"""Driver Monitoring System — Main Entry Point.

Orchestrates the full face landmark detection and micromotion extraction
pipeline with real-time visualization:

  Frame Capture → Lighting Normalize → Face Mesh (478 landmarks)
  → PnP Normalization → 1-Euro Smoothing → Head Pose Gating
  → Feature Extraction (EAR/MAR/Gaze/PERCLOS) → Output

Usage:
    python main.py                          # webcam, live display
    python main.py --source video.mp4       # video file
    python main.py --headless --save-csv    # no display, CSV output
    python main.py --source 0 --headless    # camera, no display
"""
import argparse
import csv
import logging
import os
import sys
import time

logger = logging.getLogger(__name__)

# Fast pre-flight Python version sanity check
if sys.version_info < (3, 9):
    sys.exit(
        f"[FATAL] The Overclocked DMS requires Python 3.9+ (running {sys.version.split()[0]}).\n"
        "Python 3.10 or 3.11 is strongly recommended for seamless pre-compiled binary wheels."
    )

from dataclasses import asdict, fields
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from config import PipelineConfig, DEFAULT_CONFIG
from pipeline.frame_capture import FrameCapture
from pipeline.preprocessing import LightingNormalizer
from pipeline.face_mesh_detector import FaceMeshDetector
from pipeline.pnp_normalizer import PnPNormalizer
from pipeline.one_euro_filter import OneEuroFilterBank
from pipeline.head_pose_estimator import HeadPoseGate
from pipeline.feature_extractor import FeatureExtractor, FrameFeatures
from pipeline.calibration import PersonalBaselineCalibrator
from pipeline.imu_gate import SoftwareIMUGate
from pipeline.stabilizer import KinematicStabilizer
from pipeline.classifier import ImpairmentClassifier
from pipeline.bayesian_filter import BayesianEvidenceAccumulator
from utils.visualization import Visualizer


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Driver Monitoring System — Face Landmark & Micromotion Pipeline"
    )
    parser.add_argument(
        "--source", type=str, default="0",
        help="Camera index (int) or video file path. Default: 0 (webcam)")
    parser.add_argument(
        "--width", type=int, default=1280,
        help="Requested frame width (default: 1280)")
    parser.add_argument(
        "--height", type=int, default=720,
        help="Requested frame height (default: 720)")
    parser.add_argument(
        "--headless", action="store_true",
        help="Run without OpenCV display window")
    parser.add_argument(
        "--save-csv", action="store_true", dest="save_csv",
        help="Save features to CSV file")
    parser.add_argument(
        "--csv-path", type=str, default="output_features.csv",
        help="CSV output file path (default: output_features.csv)")
    parser.add_argument(
        "--overlay-mode", type=str, choices=["off", "regions", "full", "solid"], default="regions",
        help="HUD landmark overlay mode: 'regions' (default key regions), 'full' (full mesh), 'solid' (color-filled facial parts), 'off'")
    parser.add_argument(
        "--mesh-style", type=str, choices=["regions", "full", "solid", "contours", "dots", "off"], default=None,
        help="Legacy alias for --overlay-mode")
    parser.add_argument(
        "--solid-parts", type=str, default=None,
        help="Comma-separated list of solid facial parts to enable initially (e.g. 'eyes,nose,lips,cheeks'). Default: all")
    parser.add_argument(
        "--no-mesh", action="store_true",
        help="Hide face mesh wireframe in visualization (alias for --overlay-mode off)")
    parser.add_argument(
        "--no-iris", action="store_true",
        help="Hide iris crosshairs in visualization")
    parser.add_argument(
        "--no-axes", action="store_true",
        help="Hide head pose axes in visualization")
    parser.add_argument(
        "--config", type=str, default=None,
        help="Optional JSON configuration file path")
    parser.add_argument(
        "--model-path", type=str, default=None,
        help="Path to trained .joblib impairment model (default auto-discover)")
    parser.add_argument(
        "--stabilize-display", action="store_true",
        help="Enable real-time visual video stabilization on the display window (toggle with 's')")
    parser.add_argument(
        "--fast-capture", action="store_true",
        help="Fast camera capture mode (640x480 @ 60 FPS)")
    return parser.parse_args()


class CSVLogger:
    """Thread-safe CSV logger for FrameFeatures."""

    def __init__(self, filepath: str):
        self.filepath = filepath
        self._file = open(filepath, "w", newline="", buffering=1, encoding="utf-8")
        self._field_names = [f.name for f in fields(FrameFeatures)]
        self._writer = csv.DictWriter(self._file, fieldnames=self._field_names)
        self._writer.writeheader()

    def log(self, features: FrameFeatures):
        """Write a single frame's features to CSV."""
        self._writer.writerow(asdict(features))

    def close(self):
        """Flush and close the file."""
        self._file.close()


def run_pipeline(config: PipelineConfig, args: argparse.Namespace):
    """Run the full driver monitoring pipeline."""

    # ── Parse source ──
    try:
        source = int(args.source)
    except ValueError:
        source = args.source  # video file path

    # ── Initialize all components ──
    req_width = 640 if getattr(args, "fast_capture", False) else args.width
    req_height = 480 if getattr(args, "fast_capture", False) else args.height
    req_fps = 60.0 if getattr(args, "fast_capture", False) else config.target_fps
    capture = FrameCapture(
        source=source,
        width=req_width,
        height=req_height,
        target_fps=req_fps,
        anti_motion_blur=config.anti_motion_blur,
    )
    preprocessor = LightingNormalizer(
        clip_limit=config.clahe_clip_limit,
        tile_grid_size=config.clahe_tile_grid,
        low_light_threshold=config.low_light_threshold,
        gamma_dark=config.gamma_dark,
        gamma_dim=config.gamma_dim,
    )
    detector = FaceMeshDetector(
        max_faces=config.max_faces,
        refine_landmarks=config.refine_landmarks,
        min_detection_confidence=config.min_detection_confidence,
        min_tracking_confidence=config.min_tracking_confidence,
        enable_adaptive_roi=config.enable_adaptive_roi,
        roi_margin=config.roi_margin,
        roi_alpha=config.roi_alpha,
    )
    normalizer = PnPNormalizer(
        frame_width=capture.frame_width,
        frame_height=capture.frame_height,
        focal_length=config.focal_length_px,
        calibration_path=config.camera_calibration_path,
    )
    pose_gate = HeadPoseGate(
        yaw_limit=config.yaw_limit,
        pitch_down_limit=config.pitch_down_limit,
        pitch_up_limit=config.pitch_up_limit,
    )
    # Scale temporal frame counts dynamically to actual FPS (e.g. 500ms microsleep, 1.0s yawn)
    scaled_microsleep_frames = max(5, int(round(0.50 * capture.fps)))
    scaled_yawn_frames = max(10, int(round(1.0 * capture.fps)))

    extractor = FeatureExtractor(
        blink_threshold=config.ear_blink_threshold,
        mar_yawn_threshold=config.mar_yawn_threshold,
        perclos_window_sec=config.perclos_window_sec,
        perclos_alert_threshold=config.perclos_alert_threshold,
        microsleep_frames=scaled_microsleep_frames,
        yawn_frames=scaled_yawn_frames,
        gaze_dispersion_window_sec=config.gaze_dispersion_window_sec,
        saccade_onset_velocity=config.saccade_onset_velocity,
        saccade_offset_velocity=config.saccade_offset_velocity,
        eccentric_gaze_threshold=config.eccentric_gaze_threshold,
        gen_drift_min_vel=config.gen_drift_min_vel,
        gen_drift_max_vel=config.gen_drift_max_vel,
        gen_reset_min_vel=config.gen_reset_min_vel,
        loc_vergence_threshold_deg=config.loc_vergence_threshold_deg,
        vor_min_head_speed=config.vor_min_head_speed,
        vor_max_head_speed=config.vor_max_head_speed,
        pursuit_window_sec=config.pursuit_window_sec,
        perclos_min_warmup_sec=5.0,
        fps=capture.fps,
        deg_per_iris_unit=config.deg_per_iris_unit,
        saccade_deg_per_unit=config.saccade_deg_per_unit,
        saccade_subpixel_deadzone_dx=config.saccade_subpixel_deadzone_dx,
        saccade_subpixel_deadzone_hypot=config.saccade_subpixel_deadzone_hypot,
        saccade_head_motion_dx=config.saccade_head_motion_dx,
        eye_socket_vel_threshold=config.eye_socket_vel_threshold,
        blink_phase_timeout_frames=config.blink_phase_timeout_frames,
        sway_window_sec=config.sway_window_sec,
        sway_yaw_gate_deg_s=config.sway_yaw_gate_deg_s,
        sway_nod_reset_deg_s=config.sway_nod_reset_deg_s,
        rigid_head_sway_threshold=config.rigid_head_sway_threshold,
    )
    calibrator = PersonalBaselineCalibrator(calibration_duration_sec=30.0, fps=capture.fps)
    imu_gate = SoftwareIMUGate(
        accel_threshold=config.head_accel_threshold,
        freeze_duration_ms=config.imu_freeze_ms,
        window_size=config.imu_accel_window,
        continuous_accel_min=config.continuous_vibration_accel_min,
        continuous_duration_sec=config.continuous_vibration_duration_s,
        max_freeze_duty_cycle=config.max_freeze_duty_cycle,
    )
    stabilizer = KinematicStabilizer()
    model_path = getattr(args, "model_path", None)
    if model_path is None:
        candidates = [
            "models/involuntary_model.joblib",
            "models/baseline_impairment_model.joblib",
            "models/all_features_model.joblib",
        ]
        for c in candidates:
            if os.path.exists(c):
                model_path = c
                break

    if model_path and os.path.exists(model_path):
        print(f"[DMS] Loaded trained ML model: {model_path}")
        classifier = ImpairmentClassifier(model_path=model_path)
    else:
        classifier = ImpairmentClassifier()
    bayesian_filter = BayesianEvidenceAccumulator(lockout_sustained_sec=15.0)

    # Filter bank (lazy-initialized on first valid frame)
    filter_bank: Optional[OneEuroFilterBank] = None

    # Visualization
    if args.no_mesh:
        overlay_mode = "off"
    elif getattr(args, "mesh_style", None) is not None:
        overlay_mode = "regions" if args.mesh_style in ("contours", "dots") else args.mesh_style
    else:
        overlay_mode = getattr(args, "overlay_mode", "regions")
    enabled_parts = (
        [p.strip() for p in args.solid_parts.split(",") if p.strip()]
        if getattr(args, "solid_parts", None)
        else None
    )
    visualizer = Visualizer(
        overlay_mode=overlay_mode,
        show_iris=not args.no_iris,
        show_pose_axes=not args.no_axes,
        enabled_solid_parts=enabled_parts,
        stabilize_display=getattr(args, "stabilize_display", False),
    )

    if not args.headless:
        try:
            cv2.namedWindow("Driver Monitor")
            cv2.setMouseCallback("Driver Monitor", visualizer.handle_mouse_click)
        except cv2.error:
            pass

    # CSV logger
    csv_logger: Optional[CSVLogger] = None
    if args.save_csv:
        csv_logger = CSVLogger(args.csv_path)
        print(f"[DMS] CSV logging to: {args.csv_path}")

    print(f"[DMS] Pipeline ready — {capture.frame_width}x{capture.frame_height} "
          f"@ {capture.fps:.1f} fps (est.)")
    print(f"[DMS] Input type: {'Grayscale (NIR)' if capture.is_grayscale else 'BGR (RGB camera)'}")
    if not args.headless:
        print("[DMS] Controls: 'q' quit | 'p' pause | 'v' overlay | 'r' ROI box | 's' video stabilizer | 'u' unlatch/reset")
        print("[DMS] Solid mode: [1] Forehead [2] Cheeks [3] Chin [4] Lips [5] Eyes [6] Nose [7] Brows [8] Face | [0] Toggle all")

    # ── Main loop ──
    paused = False
    frame_id = 0
    t0 = time.perf_counter()
    last_telemetry_time = t0

    last_raw_frame = None
    last_smooth_landmarks = None
    last_features = None
    last_normalizer = None
    last_pnp_success = None
    last_assessment = None
    last_bayesian_state = None
    last_roi_bounds = None
    last_shift_vector = None
    missing_face_frames = 0

    try:
        while capture.is_open():
            if paused:
                if not args.headless:
                    try:
                        key = cv2.waitKey(50) & 0xFF
                        if key in (ord('p'), ord('P')):
                            paused = False
                        elif key in (ord('q'), ord('Q')):
                            break
                        elif key in (ord('u'), ord('U')):
                            bayesian_filter.unlatch(park_gear_confirmed=True)
                            classifier.reset()
                            print("[DMS] Simulating PARK gear: Bayesian lockout unlatched & classifier reset.")
                        elif visualizer.handle_key(key) or getattr(visualizer, "_mouse_clicked", False):
                            visualizer._mouse_clicked = False
                            if last_raw_frame is not None:
                                viz_frame = last_raw_frame.copy() if len(last_raw_frame.shape) == 3 else cv2.cvtColor(last_raw_frame, cv2.COLOR_GRAY2BGR)
                                visualizer.draw(
                                    viz_frame,
                                    last_smooth_landmarks,
                                    last_features,
                                    last_normalizer,
                                    pnp_success=last_pnp_success,
                                    assessment=last_assessment,
                                    bayesian_state=last_bayesian_state,
                                    calibration_progress=calibrator.progress,
                                    roi_bounds=last_roi_bounds,
                                    shift_vector=last_shift_vector,
                                )
                                cv2.imshow("Driver Monitor", viz_frame)
                    except cv2.error as exc:
                        print(f"[DMS] OpenCV GUI error: {exc}. Switching to headless mode.")
                        args.headless = True
                        paused = False
                else:
                    time.sleep(0.05)
                continue

            frame = capture.read()
            if frame is None:
                break

            frame_id += 1
            if capture.is_camera:
                timestamp = time.perf_counter() - t0
            else:
                timestamp = (frame_id - 1) / max(capture.fps, 1.0)

            # ── Stage 1: Preprocessing ──
            face_bbox = (
                detector.roi_tracker.get_crop_bounds(frame.shape)
                if detector.roi_tracker.is_tracking
                else None
            )
            processed = preprocessor.process(frame, face_bbox=face_bbox)

            # ── Stage 2: Face Mesh Detection ──
            face_center = (detector.roi_tracker.cx, detector.roi_tracker.cy) if (detector.roi_tracker.is_tracking and detector.roi_tracker.cx is not None and detector.roi_tracker.cy is not None) else None
            face_scale = detector.roi_tracker.scale if detector.roi_tracker.is_tracking else None
            dx_cam, dy_cam, is_bumpy = stabilizer.estimate_frame_shift(
                frame,
                face_center=face_center,
                face_scale=face_scale,
            )
            landmarks = detector.detect(processed, shift_hint=(dx_cam, dy_cam))

            if landmarks is None:
                missing_face_frames += 1
                if missing_face_frames > config.max_occlusion_hold_frames:
                    normalizer.reset()
                    imu_gate.reset()
                    stabilizer.reset()
                    filter_bank = None
                features = extractor.compute(
                    landmarks=None,
                    timestamp=timestamp,
                    frame_id=frame_id,
                    face_valid=False,
                    imu_gated=False,
                    gating_reason="no_face",
                    frame=frame,
                    vibration_score=stabilizer.vibration_score,
                )
                assessment = classifier.evaluate_temporal_stream(features)
                bayesian_state = bayesian_filter.update(assessment, timestamp)
                if csv_logger is not None:
                    csv_logger.log(features)

                last_raw_frame = frame
                last_smooth_landmarks = None
                last_features = features
                last_normalizer = None
                last_pnp_success = False
                last_assessment = assessment
                last_bayesian_state = bayesian_state
                last_roi_bounds = face_bbox
                last_shift_vector = (dx_cam, dy_cam)

                if not args.headless:
                    try:
                        viz_frame = frame.copy() if len(frame.shape) == 3 else cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
                        visualizer.draw(
                            viz_frame,
                            None,
                            features,
                            None,
                            pnp_success=False,
                            assessment=assessment,
                            bayesian_state=bayesian_state,
                            calibration_progress=calibrator.progress,
                            roi_bounds=face_bbox,
                            shift_vector=(dx_cam, dy_cam),
                        )
                        cv2.imshow("Driver Monitor", viz_frame)
                        key = cv2.waitKey(1) & 0xFF
                        if key in (ord('q'), ord('Q')):
                            break
                        elif key in (ord('p'), ord('P')):
                            paused = True
                        elif key in (ord('u'), ord('U')):
                            bayesian_filter.unlatch(park_gear_confirmed=True)
                            classifier.reset()
                            print("[DMS] Simulating PARK gear: Bayesian lockout unlatched & classifier reset.")
                        else:
                            visualizer.handle_key(key)
                    except cv2.error as exc:
                        print(f"[DMS] OpenCV GUI error: {exc}. Switching to headless mode.")
                        args.headless = True
                continue

            missing_face_frames = 0

            # ── Stage 3: Split-Bandwidth 1-Euro Temporal Filtering ──
            # Filter raw 2D/3D landmarks FIRST to eliminate detector sub-pixel jitter
            # before non-linear PnP optimization, preventing pose chatter.
            eff_vibration = max(stabilizer.vibration_score, 1.0 if imu_gate.continuous_vibration_active else 0.0)
            if filter_bank is None:
                filter_bank = OneEuroFilterBank(
                    t0=timestamp,
                    x0=landmarks,
                    landmark_min_cutoff=config.landmark_min_cutoff,
                    landmark_beta=config.landmark_beta,
                    landmark_d_cutoff=config.landmark_d_cutoff,
                    eyelid_min_cutoff=config.eyelid_min_cutoff,
                    eyelid_beta=config.eyelid_beta,
                    eyelid_d_cutoff=config.eyelid_d_cutoff,
                    iris_min_cutoff=config.iris_min_cutoff,
                    iris_beta=config.iris_beta,
                    iris_d_cutoff=config.iris_d_cutoff,
                    corner_min_cutoff=config.corner_min_cutoff,
                    corner_beta=config.corner_beta,
                    ear_min_cutoff=config.ear_min_cutoff,
                    ear_beta=config.ear_beta,
                )
                smooth_landmarks = landmarks
            else:
                smooth_landmarks = filter_bank.filter_landmarks(
                    timestamp, landmarks, vibration_score=eff_vibration
                )

            # ── Stage 4: PnP Pose Estimation on Smoothed Landmarks ──
            pnp_success = normalizer.solve_pose(smooth_landmarks)

            # ── Stage 5: Face Normalization ──
            norm_landmarks = normalizer.normalize_landmarks(smooth_landmarks)

            # ── Stage 6: Head Pose Gating ──
            euler = normalizer.get_euler_angles()
            pose_state = pose_gate.check(euler)

            # ── Stage 7: IMU (Software) Gating with Rotational Decoupling ──
            imu_frozen = imu_gate.update(
                timestamp=timestamp,
                tvec=normalizer.tvec if pnp_success else None,
                euler_angles=euler if pnp_success else None,
            )

            # Recompute eff_vibration in case imu_gate updated continuous_vibration_active
            eff_vibration = max(stabilizer.vibration_score, 1.0 if imu_gate.continuous_vibration_active else 0.0)

            # ── Stage 8: Feature Extraction (Pass raw frame + un-rotated image_landmarks) ──
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
                image_landmarks=smooth_landmarks,
                vibration_score=eff_vibration,
            )

            # ── Online Personal Baseline Auto-Calibration ──
            if features.face_valid and not features.imu_gated:
                calibrator.update(features)
                if calibrator.is_calibrated:
                    calibrator.apply_to(extractor, classifier, pose_gate=pose_gate)

            assessment = classifier.evaluate_temporal_stream(features)
            bayesian_state = bayesian_filter.update(assessment, timestamp)

            # ── Output ──
            if csv_logger is not None:
                csv_logger.log(features)

            # ── Visualization ──
            current_roi_bounds = (
                detector.roi_tracker.get_crop_bounds(frame.shape)
                if detector.roi_tracker.is_tracking
                else face_bbox
            )
            last_raw_frame = frame
            last_smooth_landmarks = smooth_landmarks
            last_features = features
            last_normalizer = normalizer
            last_pnp_success = pnp_success
            last_assessment = assessment
            last_bayesian_state = bayesian_state
            last_roi_bounds = current_roi_bounds
            last_shift_vector = (dx_cam, dy_cam)

            # ── Periodic Telemetry Log (every 3.0 seconds) ──
            now_t = time.perf_counter()
            cam_fps = getattr(capture, "actual_capture_fps", capture.fps) if capture.is_camera else None
            if now_t - last_telemetry_time >= 3.0:
                cpu_fps = visualizer.fps
                cpu_ms = (1000.0 / max(cpu_fps, 1.0)) if cpu_fps > 0 else 0.0
                cam_label = f"Camera: {cam_fps:.1f} FPS | " if cam_fps is not None else ""
                print(f"[DMS Telemetry] {cam_label}CPU Pipeline: {cpu_fps:.1f} FPS ({cpu_ms:.1f} ms/frame)")
                last_telemetry_time = now_t

            if not args.headless:
                try:
                    viz_frame = frame.copy() if len(frame.shape) == 3 else cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
                    visualizer.draw(
                        viz_frame,
                        smooth_landmarks,
                        features,
                        normalizer,
                        pnp_success=pnp_success,
                        assessment=assessment,
                        bayesian_state=bayesian_state,
                        calibration_progress=calibrator.progress,
                        roi_bounds=current_roi_bounds,
                        shift_vector=(dx_cam, dy_cam),
                        cam_fps=cam_fps,
                    )
                    cv2.imshow("Driver Monitor", viz_frame)
                    key = cv2.waitKey(1) & 0xFF
                    if key in (ord('q'), ord('Q')):
                        break
                    elif key in (ord('p'), ord('P')):
                        paused = True
                    elif key in (ord('u'), ord('U')):
                        bayesian_filter.unlatch(park_gear_confirmed=True)
                        classifier.reset()
                        print("[DMS] Simulating PARK gear: Bayesian lockout unlatched & classifier reset.")
                    else:
                        visualizer.handle_key(key)
                except cv2.error as exc:
                    print(f"[DMS] OpenCV GUI error: {exc}. Switching to headless mode.")
                    args.headless = True

    except KeyboardInterrupt:
        print("\n[DMS] Interrupted by user")
    except Exception as exc:
        logger.exception("[DMS] Unexpected error in pipeline loop: %s", exc)
        raise
    finally:
        print(f"[DMS] Processed {frame_id} frames")
        capture.release()
        detector.close()
        if csv_logger is not None:
            csv_logger.close()
            print(f"[DMS] CSV saved to: {csv_logger.filepath}")
        if not args.headless:
            try:
                cv2.destroyAllWindows()
            except cv2.error:
                pass


def main():
    """Entry point."""
    args = parse_args()

    # Load configuration from file or use defaults
    if args.config:
        config = PipelineConfig.from_json(args.config)
    else:
        config = DEFAULT_CONFIG

    run_pipeline(config, args)


if __name__ == "__main__":
    main()
