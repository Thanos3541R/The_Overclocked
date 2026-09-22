"""
idd_ingest.py

This script is the bridge between raw IDD (Impaired Driving Dataset) dataset videos 
and the Driver Monitoring System training pipeline. It runs each clip through the 
full DMS feature extraction pipeline end-to-end and outputs windowed feature 
tensors tagged with ground-truth impairment labels. 

These outputs are the input to classifier training (replacing the current 
placeholder thresholds in classifier.py).

Requirements:
- Toyota IDD Dataset structured directories or labels.csv
- pipeline.training_buffer available in the pipeline module
"""

import os
import argparse
import csv
import glob
import time
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Optional
import cv2
import numpy as np

# DMS Pipeline Imports
from config import PipelineConfig, DEFAULT_CONFIG
from pipeline.preprocessing import LightingNormalizer
from pipeline.face_mesh_detector import FaceMeshDetector
from pipeline.one_euro_filter import OneEuroFilterBank
from pipeline.pnp_normalizer import PnPNormalizer
from pipeline.head_pose_estimator import HeadPoseGate
from pipeline.imu_gate import SoftwareIMUGate
from pipeline.feature_extractor import FeatureExtractor, FrameFeatures

# Try to import training buffer (assuming it exists or will be created)
try:
    from pipeline.training_buffer import TrainingBuffer, GatedFrameStrategy
except ImportError:
    # Dummy classes for type hinting if not available
    class GatedFrameStrategy:
        DROP = "drop"
        HOLD_LAST = "hold_last"
        INTERPOLATE = "interpolate"
    class TrainingBuffer:
        def __init__(self, *args, **kwargs):
            pass
        def add_sample(self, *args, **kwargs):
            pass
        def extract_windows(self):
            return []
        def save_to_disk(self, *args, **kwargs):
            pass


@dataclass
class ClipInfo:
    video_path: str
    subject_id: str
    condition_label: str
    session_id: str
    metadata: Dict


class IDDDatasetDiscovery:
    """Discovers video clips and ground-truth labels for IDD ingestion."""
    
    @staticmethod
    def discover_clips(idd_root: str) -> List[ClipInfo]:
        idd_path = Path(idd_root)
        
        # Strategy A: Structured CSV
        labels_csv = idd_path / "labels.csv"
        ann_csv = idd_path / "annotations.csv"
        csv_path = labels_csv if labels_csv.exists() else ann_csv
        
        if csv_path.exists():
            print(f"Dataset Discovery: Found structured CSV at {csv_path}")
            return IDDDatasetDiscovery._parse_csv(csv_path, idd_path)
            
        # Strategy B: Directory Convention
        print("Dataset Discovery: Falling back to directory convention discovery")
        clips = IDDDatasetDiscovery._parse_directories(idd_path)
        
        if not clips:
            raise RuntimeError(
                f"Dataset Discovery failed. Expected either:\n"
                f"1. {labels_csv.name} or {ann_csv.name} with columns [video_file, subject_id, condition]\n"
                f"2. Directory structure like <subject_id>/<condition>/*.mp4 or <condition>/<subject_id>/*.mp4"
            )
            
        return clips

    @staticmethod
    def _parse_csv(csv_path: Path, root_path: Path) -> List[ClipInfo]:
        clips = []
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            # Find matching column names
            headers = [h.lower() for h in reader.fieldnames] if reader.fieldnames else []
            vid_col = next((c for c in headers if 'video' in c or 'file' in c), None)
            sub_col = next((c for c in headers if 'subject' in c or 'sub' in c), None)
            cond_col = next((c for c in headers if 'condition' in c or 'label' in c), None)
            
            if not all([vid_col, sub_col, cond_col]):
                print(f"Warning: CSV missing expected columns. Found: {headers}")
                
            for i, row in enumerate(reader):
                vid_file = row.get(vid_col) if vid_col else list(row.values())[0]
                sub_id = row.get(sub_col, f"sub_{i}") if sub_col else f"sub_{i}"
                cond = row.get(cond_col, "unknown") if cond_col else "unknown"
                
                vid_path = root_path / vid_file
                if not vid_path.exists():
                    # Try looking for it
                    found = list(root_path.rglob(Path(vid_file).name))
                    if found:
                        vid_path = found[0]
                
                if vid_path.exists():
                    session_id = vid_path.stem
                    clips.append(ClipInfo(
                        video_path=str(vid_path.absolute()),
                        subject_id=sub_id,
                        condition_label=cond.lower(),
                        session_id=session_id,
                        metadata=row
                    ))
        return clips

    @staticmethod
    def _parse_directories(root_path: Path) -> List[ClipInfo]:
        clips = []
        valid_conditions = [
            'sober', 'impaired', 'drowsy', 'intoxicated', 'alert',
            'drunk', 'alcohol', 'control', 'normal'
        ]
        video_extensions = [
            "*.mp4", "*.avi", "*.mov", "*.mkv", "*.webm",
            "*.MP4", "*.AVI", "*.MOV", "*.MKV", "*.WEBM"
        ]
        video_files = []
        for ext in video_extensions:
            video_files.extend(root_path.rglob(ext))

        # Deduplicate paths
        seen = set()
        deduped = []
        for v in video_files:
            abs_p = str(v.resolve())
            if abs_p not in seen:
                seen.add(abs_p)
                deduped.append(v)
        video_files = deduped
        
        for vid in video_files:
            parts = vid.relative_to(root_path).parts
            if len(parts) < 2:
                continue
                
            condition = "unknown"
            subject_id = "unknown"
            
            # Check if condition is in directory names
            for part in parts:
                p_lower = part.lower()
                if p_lower in valid_conditions:
                    condition = p_lower
                elif 'sub' in p_lower or p_lower.isdigit():
                    subject_id = part
            
            if condition == "unknown" and len(parts) >= 3:
                # Infer by position if not matched
                condition = parts[0].lower() if parts[0].lower() in valid_conditions else parts[1].lower()
                subject_id = parts[1] if parts[0].lower() in valid_conditions else parts[0]
                
            clips.append(ClipInfo(
                video_path=str(vid.absolute()),
                subject_id=subject_id,
                condition_label=condition,
                session_id=vid.stem,
                metadata={}
            ))
            
        return clips


def process_clip(
    clip: ClipInfo, 
    output_dir: Path,
    config: PipelineConfig,
    window_sec: float = 3.0,
    stride_sec: float = 1.5,
    min_fill_ratio: float = 0.70,
    assumed_fps: float = 30.0,
    gated_strategy: str = "hold_last",
    headless: bool = True,
    extractor: Optional[FeatureExtractor] = None,
) -> Dict:
    """Process a single video clip through the full DMS pipeline.

    Mirrors the pipeline stages from main.py:
      1. Preprocessing (CLAHE + gamma)
      2. Face mesh detection (478 landmarks)
      3. PnP pose estimation
      4. Landmark normalization
      5. 1-Euro filtering
      6. Head pose gating
      7. Software IMU gating
      8. Feature extraction (22 signals + booleans)
      9. Training buffer accumulation

    Returns:
        Dict with processing statistics for the summary CSV.
    """
    start_time = time.time()
    
    # Map string strategy to GatedFrameStrategy enum
    strategy_map = {
        'drop': GatedFrameStrategy.DROP,
        'hold_last': GatedFrameStrategy.HOLD_LAST,
        'interpolate': GatedFrameStrategy.INTERPOLATE,
    }
    strategy_enum = strategy_map.get(gated_strategy.lower(), GatedFrameStrategy.HOLD_LAST)
    
    # Open video to get FPS before initializing components
    cap = cv2.VideoCapture(clip.video_path)
    if not cap.isOpened():
        raise IOError(f"Could not open video file: {clip.video_path}")
    
    vid_fps = cap.get(cv2.CAP_PROP_FPS)
    fps = vid_fps if vid_fps and vid_fps > 0 else assumed_fps
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or config.frame_width
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or config.frame_height
    
    # Initialize pipeline components (fresh per clip)
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
        frame_width=frame_w,
        frame_height=frame_h,
    )
    pose_gate = HeadPoseGate(
        yaw_limit=config.yaw_limit,
        pitch_down_limit=config.pitch_down_limit,
        pitch_up_limit=config.pitch_up_limit,
    )
    imu_gate = SoftwareIMUGate(
        accel_threshold=config.head_accel_threshold,
        freeze_duration_ms=config.imu_freeze_ms,
        window_size=config.imu_accel_window,
    )
    if extractor is None:
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
            fps=fps,
        )
    training_buffer = TrainingBuffer(
        subject_id=clip.subject_id,
        fps=fps,
        window_size_sec=window_sec,
        stride_sec=stride_sec,
        min_fill_ratio=min_fill_ratio,
        gated_strategy=strategy_enum,
    )

    # 1-Euro filter bank (lazy-initialized on first valid frame)
    filter_bank = None
    
    frames_processed = 0
    frames_valid = 0
    frames_gated = 0
    
    try:
        frame_id = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            frame_id += 1
            frames_processed += 1
            timestamp = (frame_id - 1) / fps  # seconds
            
            # Stage 1: Preprocessing
            processed = preprocessor.process(frame)
            
            # Stage 2: Face mesh detection
            landmarks = detector.detect(processed)
            
            if landmarks is None:
                continue
            
            # Stage 3: 1-Euro Temporal Filtering
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

            # Stage 4: PnP pose estimation on smoothed landmarks
            pnp_success = normalizer.solve_pose(smooth_landmarks)

            # Stage 5: Landmark normalization
            norm_landmarks = normalizer.normalize_landmarks(smooth_landmarks)

            # Stage 6: Head pose gating
            euler = normalizer.get_euler_angles()
            pose_state = pose_gate.check(euler)

            # Stage 7: Software IMU gating with rotational decoupling
            imu_frozen = imu_gate.update(
                timestamp=timestamp,
                tvec=normalizer.tvec if pnp_success else None,
                euler_angles=euler if pnp_success else None
            )
            if imu_frozen:
                frames_gated += 1

            # Stage 8: Feature extraction
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
                frame=processed,
            )
            
            # Stage 9: Accumulate into training buffer
            training_buffer.add_sample(features)
            frames_valid += 1
            
            if not headless:
                cv2.putText(frame, f"Frame {frame_id}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.imshow("IDD Ingest", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                    
    finally:
        cap.release()
        if hasattr(detector, 'close'):
            detector.close()
        if not headless:
            cv2.destroyAllWindows()
            
    # Extract windows and tag with condition label
    windows = training_buffer.extract_windows()
    for w in windows:
        w.label = clip.condition_label
    n_windows = len(windows)
    
    # Save to disk
    out_file = output_dir / f"{clip.subject_id}_{clip.session_id}.npz"
    training_buffer.save_to_disk(str(out_file))
    
    processing_time = time.time() - start_time
    
    return {
        "clip_path": clip.video_path,
        "subject_id": clip.subject_id,
        "condition_label": clip.condition_label,
        "n_frames_total": total_frames if total_frames > 0 else frames_processed,
        "n_frames_valid": frames_valid,
        "n_frames_gated": frames_gated,
        "n_windows_extracted": n_windows,
        "processing_time_sec": processing_time,
        "error": None
    }


def main():
    parser = argparse.ArgumentParser(description="Ingest Toyota IDD dataset for DMS training.")
    parser.add_argument("--idd-root", required=True, help="Root directory of the IDD dataset")
    parser.add_argument("--output-dir", default="./idd_output/", help="Output directory for .npz files")
    parser.add_argument("--window-sec", type=float, default=3.0, help="Feature window duration in seconds (default 3.0s)")
    parser.add_argument("--stride-sec", type=float, default=1.5, help="Window stride in seconds (default 1.5s)")
    parser.add_argument("--min-fill-ratio", type=float, default=0.70, help="Minimum non-empty fill ratio (default 0.70)")
    parser.add_argument("--max-clips", type=int, default=None, help="Max clips to process (for debugging)")
    parser.add_argument("--fps", type=float, default=30.0, help="Assumed FPS if metadata unavailable")
    parser.add_argument("--headless", action="store_true", default=True, help="No visualization")
    parser.add_argument("--show", dest="headless", action="store_false", help="Enable visualization")
    parser.add_argument("--resume", action="store_true", help="Skip existing .npz outputs")
    parser.add_argument("--gated-strategy", choices=['drop', 'hold_last', 'interpolate'], 
                        default='hold_last', help="How to handle IMU-gated frames")
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"--- IDD Dataset Ingestion ---")
    print(f"Dataset root: {args.idd_root}")
    print(f"Output dir: {output_dir}")
    print(f"Window: {args.window_sec}s, Stride: {args.stride_sec}s, Min Fill: {args.min_fill_ratio*100:.0f}%")
    
    # Discover clips
    try:
        clips = IDDDatasetDiscovery.discover_clips(args.idd_root)
    except Exception as e:
        print(f"Error during dataset discovery: {e}")
        return
        
    print(f"Found {len(clips)} total clips.")
    
    if args.max_clips:
        clips = clips[:args.max_clips]
        print(f"Limiting to {len(clips)} clips (--max-clips).")
        
    config = DEFAULT_CONFIG
    
    summary_data = []
    
    total_clips = len(clips)
    success_count = 0
    fail_count = 0
    total_frames_processed = 0
    total_proc_time = 0.0
    extractors_by_subject: Dict[str, FeatureExtractor] = {}
    
    for i, clip in enumerate(clips):
        progress_str = f"[{i+1}/{total_clips}] Processing {clip.subject_id}/{clip.condition_label}/{clip.session_id} ..."
        print(progress_str, end="", flush=True)
        
        # Check resume
        out_file = output_dir / f"{clip.subject_id}_{clip.session_id}.npz"
        if args.resume and out_file.exists():
            print(" SKIPPED (already exists)")
            continue
            
        try:
            # Maintain persistent rolling feature extraction memory per subject to prevent sparse window aliasing
            subj_extractor = extractors_by_subject.setdefault(
                clip.subject_id,
                FeatureExtractor(
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
                    fps=args.fps,
                )
            )

            res = process_clip(
                clip=clip,
                output_dir=output_dir,
                config=config,
                window_sec=args.window_sec,
                stride_sec=args.stride_sec,
                min_fill_ratio=args.min_fill_ratio,
                assumed_fps=args.fps,
                gated_strategy=args.gated_strategy,
                headless=args.headless,
                extractor=subj_extractor,
            )
            
            success_count += 1
            total_frames_processed += res['n_frames_total']
            total_proc_time += res['processing_time_sec']
            
            cond = res['condition_label']
            condition_counts[cond] = condition_counts.get(cond, 0) + res['n_windows_extracted']
            
            summary_data.append(res)
            
            print(f" {res['n_frames_total']} frames, {res['n_windows_extracted']} windows extracted ({res['processing_time_sec']:.1f}s)")
            
        except Exception as e:
            fail_count += 1
            error_msg = str(e)
            print(f" FAILED: {error_msg}")
            
            summary_data.append({
                "clip_path": clip.video_path,
                "subject_id": clip.subject_id,
                "condition_label": clip.condition_label,
                "n_frames_total": 0,
                "n_frames_valid": 0,
                "n_frames_gated": 0,
                "n_windows_extracted": 0,
                "processing_time_sec": 0,
                "error": error_msg
            })
            
    # Write summary CSV
    summary_path = output_dir / "ingest_summary.csv"
    if summary_data:
        keys = ["clip_path", "subject_id", "condition_label", "n_frames_total", 
                "n_frames_valid", "n_frames_gated", "n_windows_extracted", 
                "processing_time_sec", "error"]
        try:
            with open(summary_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=keys)
                writer.writeheader()
                writer.writerows(summary_data)
        except Exception as e:
            print(f"\nFailed to write summary CSV: {e}")
            
    # Print summary statistics
    print("\n--- Ingestion Summary ---")
    print(f"Total clips processed: {success_count + fail_count}")
    print(f"Successful: {success_count}, Failed: {fail_count}")
    
    print("Windows extracted by condition:")
    if not condition_counts:
        print("  None")
    for cond, count in condition_counts.items():
        print(f"  {cond}: {count}")
        
    print(f"Total frames processed: {total_frames_processed}")
    if total_proc_time > 0:
        avg_fps = total_frames_processed / total_proc_time
        print(f"Average processing speed: {avg_fps:.1f} fps")
        
    if fail_count > 0:
        print("Failed clips:")
        for res in summary_data:
            if res.get('error'):
                print(f"  - {res['subject_id']}/{Path(res['clip_path']).name}: {res['error']}")

if __name__ == "__main__":
    main()
