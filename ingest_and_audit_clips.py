"""YouTube Clip Dataset Ingestion & Quality Auditor for Driver Monitoring.

Ingests short video clips (e.g. 10s YouTube clips) from --sober-dir and --drunk-dir,
audits each clip for failure modes (low face detection, jump cuts / person switches,
excessive head pose / IMU gating), groups clips by source video subject_id,
applies asymmetric sliding-window strides to balance class representation,
saves valid .npz training archives, and outputs comprehensive audit summaries
(dataset_audit_summary.csv and dataset_audit_summary.json).
"""
import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from config import DEFAULT_CONFIG
from pipeline.face_mesh_detector import FaceMeshDetector
from pipeline.feature_extractor import FeatureExtractor
from pipeline.head_pose_estimator import HeadPoseGate
from pipeline.imu_gate import SoftwareIMUGate
from pipeline.one_euro_filter import OneEuroFilterBank
from pipeline.pnp_normalizer import PnPNormalizer
from pipeline.preprocessing import LightingNormalizer
from pipeline.stabilizer import KinematicStabilizer
from pipeline.training_buffer import TrainingBuffer, TrainingWindow


VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"}


def infer_subject_id(filepath: str, label: str) -> str:
    r"""Extract source video/person subject_id from clip filepath and condition label.

    When clips are split from a longer video (e.g. videoA_clip01.mp4, videoA_part2.mp4,
    yt_abc123_001.mp4, or inside a subfolder Sober/person_1/clip01.mp4), extracts the base
    video/person prefix by stripping trailing clip/part/segment suffixes (_clip\d+, _part\d+,
    _seg\d+, -\d+$, _\d+$) or using the parent subfolder name if nested, prefixed with
    'sober_' or 'drunk_'.

    If a file has a trailing clip/part number > 5, it is sub-grouped into batches of 6
    (e.g. _grp01, _grp02) to guarantee sufficient distinct subject groups for StratifiedGroupKFold
    on compilation videos.
    """
    p = Path(filepath.strip())
    stem = p.stem.strip()

    part_match = re.search(r"(?:_part|_clip|_seg|part|clip|seg|-|_|\s)(\d+)$", stem, flags=re.IGNORECASE)
    part_num = int(part_match.group(1)) if part_match else None

    # Pattern for trailing clip/part/seg suffixes
    suffix_pattern = r"(_clip\d+|_part\d+|_seg\d+|-\d+$|_\d+$)"
    clean_stem = re.sub(suffix_pattern, "", stem, flags=re.IGNORECASE).strip()

    parent_name = p.parent.name.strip()
    parent_lower = parent_name.lower()
    generic_folders = {"sober", "drunk", "control", "impaired", "videos", "clips", "", "."}

    # If the stem is generic (e.g. "clip01" -> "clip"), check parent directory
    if not clean_stem or clean_stem.lower() in {"clip", "part", "seg", "video"}:
        if parent_lower not in generic_folders:
            clean_stem = re.sub(suffix_pattern, "", parent_name, flags=re.IGNORECASE).strip()
        if not clean_stem:
            clean_stem = "subject"
    elif parent_lower not in generic_folders and parent_name:
        if re.match(r"^(clip|part|seg|video)", stem, flags=re.IGNORECASE):
            clean_stem = parent_name

    # Standardize label prefix
    lbl = label.strip().lower()
    clean_stem = re.sub(rf"^{lbl}[_\-\s]*", "", clean_stem, flags=re.IGNORECASE).strip()
    clean_stem = clean_stem.strip("_- ")
    if not clean_stem:
        clean_stem = "unknown"

    sub_id = f"{lbl}_{clean_stem}"
    if part_num is not None and (part_num // 6) > 0:
        sub_id = f"{sub_id}_grp{part_num // 6:02d}"

    return sub_id


def discover_video_clips(directory: str, label: str) -> List[Tuple[str, str]]:
    """Recursively discover video files within a directory and pair with label."""
    if not directory or not os.path.exists(directory):
        return []

    found = []
    for root, _, files in os.walk(directory):
        for file in sorted(files):
            ext = os.path.splitext(file)[1].lower()
            if ext in VIDEO_EXTENSIONS:
                full_path = os.path.abspath(os.path.join(root, file))
                found.append((full_path, label))
    return sorted(found, key=lambda x: x[0])


def audit_and_ingest_clip(
    video_path: str,
    label: str,
    output_dir: str,
    stride_sec: float,
    window_sec: float = 3.0,
    headless: bool = True,
) -> Dict[str, Any]:
    """Run full DMS pipeline on a video clip, audit for artifacts, and extract training windows.

    Args:
        video_path: Path to the video file.
        label: Condition label ('sober' or 'drunk').
        output_dir: Directory where valid .npz archives will be saved.
        stride_sec: Window sliding stride in seconds.
        window_sec: Window duration in seconds (default 3.0s).
        headless: Headless flag for execution.

    Returns:
        Audit record dict with metrics, status, and window counts.
    """
    subject_id = infer_subject_id(video_path, label)
    stem = Path(video_path).stem.strip()

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {
            "clip_path": video_path,
            "subject_id": subject_id,
            "label": label,
            "status": "REJECTED_CORRUPT",
            "total_frames": 0,
            "fps": 0.0,
            "duration_s": 0.0,
            "face_detected_frames": 0,
            "face_detection_rate_pct": 0.0,
            "valid_ungated_frames": 0,
            "valid_ungated_rate_pct": 0.0,
            "scene_cut_count": 0,
            "windows_extracted": 0,
            "archive_file": "",
        }

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    source_fps = cap.get(cv2.CAP_PROP_FPS)
    fps = float(source_fps) if source_fps and source_fps > 5.0 else 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1280
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 720
    duration_s = float(total_frames / fps) if total_frames > 0 else 0.0

    config = DEFAULT_CONFIG
    prep = LightingNormalizer(
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
    pnp = PnPNormalizer(width, height, focal_length=config.focal_length_px)
    pose_gate = HeadPoseGate(
        yaw_limit=config.yaw_limit,
        pitch_down_limit=config.pitch_down_limit,
        pitch_up_limit=config.pitch_up_limit,
    )
    imu_gate = SoftwareIMUGate(
        accel_threshold=config.head_accel_threshold,
        freeze_duration_ms=config.imu_freeze_ms,
        window_size=config.imu_accel_window,
        continuous_accel_min=config.continuous_vibration_accel_min,
        continuous_duration_sec=config.continuous_vibration_duration_s,
        max_freeze_duty_cycle=config.max_freeze_duty_cycle,
    )
    stabilizer = KinematicStabilizer()
    filter_bank: Optional[OneEuroFilterBank] = None

    def make_extractor():
        return FeatureExtractor(
            blink_threshold=config.ear_blink_threshold,
            mar_yawn_threshold=config.mar_yawn_threshold,
            perclos_window_sec=config.perclos_window_sec,
            perclos_alert_threshold=config.perclos_alert_threshold,
            fps=fps,
            deg_per_iris_unit=config.deg_per_iris_unit,
            saccade_deg_per_unit=config.saccade_deg_per_unit,
            sway_window_sec=config.sway_window_sec,
        )

    extractor = make_extractor()
    buffer = TrainingBuffer(
        subject_id=subject_id,
        fps=fps,
        window_size_sec=window_sec,
        stride_sec=stride_sec,
        include_extended_features=True,
    )

    face_detected_frames = 0
    valid_ungated_frames = 0
    scene_cut_count = 0
    prev_bbox_center: Optional[np.ndarray] = None
    prev_iod: Optional[float] = None
    accumulated_windows: List[TrainingWindow] = []

    frame_idx = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_idx += 1
            timestamp = (frame_idx - 1) / fps

            face_bbox = (
                detector.roi_tracker.get_crop_bounds(frame.shape)
                if detector.roi_tracker.is_tracking
                else None
            )
            processed_frame = prep.process(frame, face_bbox=face_bbox)

            face_center = (
                (detector.roi_tracker.cx, detector.roi_tracker.cy)
                if detector.roi_tracker.is_tracking and detector.roi_tracker.cx is not None
                else None
            )
            face_scale = (
                detector.roi_tracker.scale if detector.roi_tracker.is_tracking else None
            )
            dx_cam, dy_cam, is_bumpy = stabilizer.estimate_frame_shift(
                frame, face_center=face_center, face_scale=face_scale
            )
            landmarks = detector.detect(processed_frame, shift_hint=(dx_cam, dy_cam))

            if landmarks is not None:
                curr_center = np.array(
                    [
                        (landmarks[:, 0].min() + landmarks[:, 0].max()) / 2.0,
                        (landmarks[:, 1].min() + landmarks[:, 1].max()) / 2.0,
                    ]
                )
                curr_iod = float(np.linalg.norm(landmarks[263, :2] - landmarks[33, :2]))

                # Jump-cut / scene-cut detection
                if prev_bbox_center is not None and prev_iod is not None:
                    center_jump = float(np.linalg.norm(curr_center - prev_bbox_center))
                    iod_pct_change = abs(curr_iod - prev_iod) / max(prev_iod, 1e-5)
                    if center_jump > 180.0 or iod_pct_change > 0.35:
                        scene_cut_count += 1
                        # Harvest complete windows accumulated before the cut
                        accumulated_windows.extend(buffer.extract_windows(label=label))
                        # Reset tracking state so windows never bridge across a cut
                        buffer.reset()
                        filter_bank = None
                        extractor = make_extractor()
                        pnp.reset()
                        imu_gate.reset()
                        stabilizer.reset()

                prev_bbox_center = curr_center
                prev_iod = curr_iod

                face_detected_frames += 1

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
                    eff_vibration = max(
                        stabilizer.vibration_score,
                        1.0 if imu_gate.continuous_vibration_active else 0.0,
                    )
                    smooth_landmarks = filter_bank.filter_landmarks(
                        timestamp, landmarks, vibration_score=eff_vibration
                    )

                pnp_success = pnp.solve_pose(smooth_landmarks)
                norm_landmarks = pnp.normalize_landmarks(smooth_landmarks)
                euler = pnp.get_euler_angles()
                pose_state = pose_gate.check(euler)

                imu_frozen = imu_gate.update(
                    timestamp=timestamp,
                    tvec=pnp.tvec if pnp_success else None,
                    euler_angles=euler if pnp_success else None,
                )
                eff_vibration = max(
                    stabilizer.vibration_score,
                    1.0 if imu_gate.continuous_vibration_active else 0.0,
                )

                features = extractor.compute(
                    landmarks=norm_landmarks,
                    timestamp=timestamp,
                    frame_id=frame_idx,
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
            else:
                features = extractor.compute(
                    landmarks=None,
                    timestamp=timestamp,
                    frame_id=frame_idx,
                    face_valid=False,
                    imu_gated=False,
                    gating_reason="no_face",
                    frame=frame,
                    vibration_score=stabilizer.vibration_score,
                )

            if features.face_valid and not features.imu_gated:
                valid_ungated_frames += 1

            buffer.add_sample(features)

        # Harvest final trailing windows
        accumulated_windows.extend(buffer.extract_windows(label=label))

    finally:
        detector.close()
        cap.release()

    total_actual_frames = max(frame_idx, total_frames, 1)
    face_detection_rate_pct = float(face_detected_frames / total_actual_frames * 100.0)
    valid_ungated_rate_pct = float(valid_ungated_frames / total_actual_frames * 100.0)

    # Quarantine / Rejection Rules
    if face_detection_rate_pct < 65.0 or valid_ungated_rate_pct < 50.0:
        status = "REJECTED_LOW_FACE"
    elif scene_cut_count >= 2:
        status = "WARNING_JUMP_CUTS"
    else:
        status = "ACCEPTED"

    archive_file = ""
    if status in ("ACCEPTED", "WARNING_JUMP_CUTS") and accumulated_windows:
        os.makedirs(output_dir, exist_ok=True)
        archive_name = f"{subject_id}_{stem}.npz"
        archive_path = os.path.join(output_dir, archive_name)

        features_arr = np.stack([w.features for w in accumulated_windows])
        timestamps_arr = np.stack([w.timestamps for w in accumulated_windows])
        subject_ids_arr = np.array([w.subject_id for w in accumulated_windows], dtype=str)
        labels_arr = np.array(
            [w.label if w.label is not None else "" for w in accumulated_windows], dtype=str
        )

        meta = {
            "fps": fps,
            "window_size_sec": window_sec,
            "stride_sec": stride_sec,
            "min_fill_ratio": buffer.min_fill_ratio,
            "gated_strategy": buffer.gated_strategy.value,
            "clip_status": status,
            "scene_cut_count": scene_cut_count,
            "source_clip": str(video_path),
        }

        np.savez(
            archive_path,
            features=features_arr,
            timestamps=timestamps_arr,
            feature_names=np.array(buffer.feature_names, dtype=str),
            subject_ids=subject_ids_arr,
            session_id=buffer.session_id,
            labels=labels_arr,
            metadata=json.dumps(meta),
        )
        archive_file = archive_name

    return {
        "clip_path": str(video_path),
        "subject_id": subject_id,
        "label": label,
        "status": status,
        "total_frames": int(total_actual_frames),
        "fps": float(fps),
        "duration_s": float(duration_s),
        "face_detected_frames": int(face_detected_frames),
        "face_detection_rate_pct": round(face_detection_rate_pct, 2),
        "valid_ungated_frames": int(valid_ungated_frames),
        "valid_ungated_rate_pct": round(valid_ungated_rate_pct, 2),
        "scene_cut_count": int(scene_cut_count),
        "windows_extracted": int(len(accumulated_windows)),
        "archive_file": archive_file,
    }


def ingest_dataset(
    sober_dir: Optional[str] = None,
    drunk_dir: Optional[str] = None,
    output_dir: str = "./training_data/",
    sober_stride: Optional[float] = None,
    drunk_stride: Optional[float] = None,
    window_sec: float = 3.0,
    max_clips: Optional[int] = None,
    headless: bool = True,
    clean_output: bool = True,
) -> Dict[str, Any]:
    """Discover clips, audit each, extract training windows, and write audit summaries."""
    os.makedirs(output_dir, exist_ok=True)
    if clean_output:
        existing_npz = list(Path(output_dir).glob("*.npz"))
        if existing_npz:
            print(f"[Ingest Auditor] Cleaning {len(existing_npz)} existing .npz archives from {output_dir}")
            for p in existing_npz:
                try:
                    p.unlink()
                except Exception as e:
                    print(f"[Warning] Failed to delete {p}: {e}")

    all_clips: List[Tuple[str, str]] = []
    sober_clips: List[Tuple[str, str]] = []
    if sober_dir and os.path.exists(sober_dir):
        sober_clips = discover_video_clips(sober_dir, "sober")
        all_clips.extend(sober_clips)
        print(f"[Ingest Auditor] Discovered {len(sober_clips)} Sober clips in {sober_dir}")
    elif sober_dir:
        print(f"[Warning] Sober directory not found: {sober_dir}")

    drunk_clips: List[Tuple[str, str]] = []
    if drunk_dir and os.path.exists(drunk_dir):
        drunk_clips = discover_video_clips(drunk_dir, "drunk")
        all_clips.extend(drunk_clips)
        print(f"[Ingest Auditor] Discovered {len(drunk_clips)} Drunk clips in {drunk_dir}")
    elif drunk_dir:
        print(f"[Warning] Drunk directory not found: {drunk_dir}")

    # Auto-calculate strides if not specified to balance extracted window counts ~1:1
    sober_count = len(sober_clips)
    drunk_count = len(drunk_clips)
    if sober_stride is None or drunk_stride is None:
        if sober_count > 0 and drunk_count > 0:
            if sober_count >= drunk_count:
                auto_sober = 1.5
                auto_drunk = float(np.clip(1.5 * (drunk_count / max(sober_count, 1)), 0.5, 1.5))
            else:
                auto_drunk = 1.5
                auto_sober = float(np.clip(1.5 * (sober_count / max(drunk_count, 1)), 0.5, 1.5))
            if sober_stride is None:
                sober_stride = auto_sober
            if drunk_stride is None:
                drunk_stride = auto_drunk
        else:
            if sober_stride is None:
                sober_stride = 1.5
            if drunk_stride is None:
                drunk_stride = 0.5
        print(f"[Ingest Auditor] Auto-balanced strides: Sober={sober_stride:.2f}s, Drunk={drunk_stride:.2f}s")

    if not all_clips:
        print("[Ingest Auditor] No video clips found in specified directories.")

    if max_clips:
        all_clips = all_clips[:max_clips]

    audit_records: List[Dict[str, Any]] = []
    total_accepted = 0
    total_warnings = 0
    total_rejected = 0
    total_windows = 0
    sober_windows = 0
    drunk_windows = 0

    print(f"\nProcessing {len(all_clips)} video clips...")
    for idx, (vpath, lbl) in enumerate(all_clips, 1):
        stride = sober_stride if lbl == "sober" else drunk_stride
        print(f"[{idx}/{len(all_clips)}] Processing {Path(vpath).name} ({lbl.upper()}, stride={stride:.1f}s)...")
        rec = audit_and_ingest_clip(
            video_path=vpath,
            label=lbl,
            output_dir=output_dir,
            stride_sec=stride,
            window_sec=window_sec,
            headless=headless,
        )
        audit_records.append(rec)

        if rec["status"] == "ACCEPTED":
            total_accepted += 1
        elif rec["status"] == "WARNING_JUMP_CUTS":
            total_warnings += 1
        else:
            total_rejected += 1

        n_win = rec["windows_extracted"]
        total_windows += n_win
        if lbl == "sober":
            sober_windows += n_win
        else:
            drunk_windows += n_win

        print(
            f"   -> Status: {rec['status']} | Face: {rec['face_detection_rate_pct']}% | "
            f"Ungated: {rec['valid_ungated_rate_pct']}% | Cuts: {rec['scene_cut_count']} | "
            f"Windows: {n_win}"
        )

    # Write summary CSV
    csv_path = os.path.join(output_dir, "dataset_audit_summary.csv")
    csv_headers = [
        "clip_path",
        "subject_id",
        "label",
        "status",
        "total_frames",
        "fps",
        "duration_s",
        "face_detected_frames",
        "face_detection_rate_pct",
        "valid_ungated_frames",
        "valid_ungated_rate_pct",
        "scene_cut_count",
        "windows_extracted",
        "archive_file",
    ]
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write(",".join(csv_headers) + "\n")
        for r in audit_records:
            row = [
                str(r["clip_path"]),
                str(r["subject_id"]),
                str(r["label"]),
                str(r["status"]),
                str(r["total_frames"]),
                str(r["fps"]),
                str(r["duration_s"]),
                str(r["face_detected_frames"]),
                str(r["face_detection_rate_pct"]),
                str(r["valid_ungated_frames"]),
                str(r["valid_ungated_rate_pct"]),
                str(r["scene_cut_count"]),
                str(r["windows_extracted"]),
                str(r["archive_file"]),
            ]
            f.write(",".join(row) + "\n")

    imbalance_ratio = round(sober_windows / max(1, drunk_windows), 2)
    summary_data = {
        "total_clips": len(all_clips),
        "accepted_clips": total_accepted,
        "warning_jump_cut_clips": total_warnings,
        "rejected_clips": total_rejected,
        "total_windows": total_windows,
        "sober_windows": sober_windows,
        "drunk_windows": drunk_windows,
        "imbalance_ratio": imbalance_ratio,
        "clips": audit_records,
    }

    json_path = os.path.join(output_dir, "dataset_audit_summary.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)

    print(f"\n{'='*70}")
    print(f"  DATASET AUDIT & INGESTION COMPLETE")
    print(f"{'='*70}")
    print(f"  Total Clips Audited : {len(all_clips)}")
    print(f"  Accepted Clips      : {total_accepted}")
    print(f"  Warning Jump Cuts   : {total_warnings}")
    print(f"  Rejected Clips      : {total_rejected}")
    print(f"  Total Windows       : {total_windows} (Sober: {sober_windows}, Drunk: {drunk_windows})")
    print(f"  Class Balance Ratio : {imbalance_ratio}:1 (Sober : Drunk)")
    print(f"  Summary CSV         : {os.path.abspath(csv_path)}")
    print(f"  Summary JSON        : {os.path.abspath(json_path)}")
    print(f"{'='*70}\n")

    return summary_data


def main():
    parser = argparse.ArgumentParser(
        description="YouTube Clip Dataset Ingestion & Quality Auditor for Driver Monitoring."
    )
    parser.add_argument("--sober-dir", default=None, help="Directory containing Sober video clips")
    parser.add_argument("--drunk-dir", default=None, help="Directory containing Drunk video clips")
    parser.add_argument(
        "--output-dir",
        default="./training_data/",
        help="Directory to save .npz feature archives and audit summaries",
    )
    parser.add_argument(
        "--sober-stride",
        type=float,
        default=None,
        help="Sliding window stride in seconds for Sober clips (default: auto-balanced)",
    )
    parser.add_argument(
        "--drunk-stride",
        type=float,
        default=None,
        help="Sliding window stride in seconds for Drunk clips (default: auto-balanced)",
    )
    parser.add_argument(
        "--clean-output",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Clean existing .npz archives from --output-dir before saving (default: True)",
    )
    parser.add_argument(
        "--window-sec",
        type=float,
        default=3.0,
        help="Window duration in seconds (default 3.0s)",
    )
    parser.add_argument(
        "--max-clips",
        type=int,
        default=None,
        help="Maximum total clips to process (optional, for debugging)",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        default=True,
        help="Run without displaying GUI windows (default True)",
    )

    args = parser.parse_args()
    ingest_dataset(
        sober_dir=args.sober_dir,
        drunk_dir=args.drunk_dir,
        output_dir=args.output_dir,
        sober_stride=args.sober_stride,
        drunk_stride=args.drunk_stride,
        window_sec=args.window_sec,
        max_clips=args.max_clips,
        headless=args.headless,
        clean_output=args.clean_output,
    )


if __name__ == "__main__":
    main()
