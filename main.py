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
import sys
import time
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
from pipeline.imu_gate import SoftwareIMUGate
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
        "--no-mesh", action="store_true",
        help="Hide face mesh wireframe in visualization")
    parser.add_argument(
        "--no-iris", action="store_true",
        help="Hide iris crosshairs in visualization")
    parser.add_argument(
        "--no-axes", action="store_true",
        help="Hide head pose axes in visualization")
    return parser.parse_args()


class CSVLogger:
    """Thread-safe CSV logger for FrameFeatures."""

    def __init__(self, filepath: str):
        self.filepath = filepath
        self._file = open(filepath, "w", newline="", buffering=1)
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
    print(f"[DMS] Initializing pipeline (source={source})...")
    capture = FrameCapture(source=source, width=args.width, height=args.height)
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
    )
    normalizer = PnPNormalizer(
        frame_width=capture.frame_width,
        frame_height=capture.frame_height,
    )
    pose_gate = HeadPoseGate(
        yaw_limit=config.yaw_limit,
        pitch_down_limit=config.pitch_down_limit,
        pitch_up_limit=config.pitch_up_limit,
    )
    extractor = FeatureExtractor(
        blink_threshold=config.ear_blink_threshold,
        mar_yawn_threshold=config.mar_yawn_threshold,
        perclos_window_sec=config.perclos_window_sec,
        perclos_alert_threshold=config.perclos_alert_threshold,
        microsleep_frames=config.microsleep_frames,
        yawn_frames=config.yawn_frames,
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
        fps=capture.fps,
    )
    imu_gate = SoftwareIMUGate(
        accel_threshold=config.head_accel_threshold,
        freeze_duration_ms=config.imu_freeze_ms,
        window_size=config.imu_accel_window,
    )

    # Filter bank (lazy-initialized on first valid frame)
    filter_bank: Optional[OneEuroFilterBank] = None

    # Visualization
    visualizer = Visualizer(
        show_mesh=not args.no_mesh,
        show_iris=not args.no_iris,
        show_pose_axes=not args.no_axes,
    )

    # CSV logger
    csv_logger: Optional[CSVLogger] = None
    if args.save_csv:
        csv_logger = CSVLogger(args.csv_path)
        print(f"[DMS] CSV logging to: {args.csv_path}")

    print(f"[DMS] Pipeline ready — {capture.frame_width}x{capture.frame_height} "
          f"@ {capture.fps:.1f} fps (est.)")
    print(f"[DMS] Input type: {'Grayscale (NIR)' if capture.is_grayscale else 'BGR (RGB camera)'}")
    if not args.headless:
        print("[DMS] Press 'q' to quit, 'p' to pause/resume")

    # ── Main loop ──
    paused = False
    frame_id = 0
    t0 = time.time()

    try:
        while capture.is_open():
            if paused:
                key = cv2.waitKey(50) & 0xFF
                if key == ord('p'):
                    paused = False
                elif key == ord('q'):
                    break
                continue

            frame = capture.read()
            if frame is None:
                break

            timestamp = time.time() - t0
            frame_id += 1

            # ── Stage 1: Preprocessing ──
            processed = preprocessor.process(frame)

            # ── Stage 2: Face Mesh Detection ──
            landmarks = detector.detect(processed)

            if landmarks is None:
                # No face detected — show frame without annotations
                if not args.headless:
                    viz_frame = frame.copy() if len(frame.shape) == 3 else cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
                    visualizer.draw(viz_frame, None, None, None)
                    cv2.imshow("Driver Monitor", viz_frame)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        break
                    elif key == ord('p'):
                        paused = True
                continue

            # ── Stage 3: 1-Euro Temporal Filtering ──
            # Filter raw 2D/3D landmarks FIRST to eliminate detector sub-pixel jitter
            # before non-linear PnP optimization, preventing pose chatter.
            if filter_bank is None:
                filter_bank = OneEuroFilterBank(
                    t0=timestamp,
                    x0=landmarks,
                    landmark_min_cutoff=config.landmark_min_cutoff,
                    landmark_beta=config.landmark_beta,
                    ear_min_cutoff=config.ear_min_cutoff,
                    ear_beta=config.ear_beta,
                )
                smooth_landmarks = landmarks
            else:
                smooth_landmarks = filter_bank.filter_landmarks(timestamp, landmarks)

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
                euler_angles=euler if pnp_success else None
            )

            # ── Stage 8: Feature Extraction ──
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
            )

            # ── Stage 9: Smooth scalar features ──
            if filter_bank is not None and pose_state.face_valid and not imu_frozen:
                features.ear_avg = filter_bank.filter_scalar(
                    timestamp, features.ear_avg, "ear_avg")
                features.mar = filter_bank.filter_scalar(
                    timestamp, features.mar, "mar")

            # ── Output ──
            if csv_logger is not None:
                csv_logger.log(features)

            # ── Visualization ──
            if not args.headless:
                viz_frame = frame.copy() if len(frame.shape) == 3 else cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
                visualizer.draw(viz_frame, landmarks, features, normalizer)
                cv2.imshow("Driver Monitor", viz_frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('p'):
                    paused = True

    except KeyboardInterrupt:
        print("\n[DMS] Interrupted by user")
    finally:
        print(f"[DMS] Processed {frame_id} frames")
        capture.release()
        detector.close()
        if csv_logger is not None:
            csv_logger.close()
            print(f"[DMS] CSV saved to: {csv_logger.filepath}")
        if not args.headless:
            cv2.destroyAllWindows()


def main():
    """Entry point."""
    args = parse_args()

    # Build config from defaults (could be extended to load from YAML/JSON)
    config = DEFAULT_CONFIG

    run_pipeline(config, args)


if __name__ == "__main__":
    main()
