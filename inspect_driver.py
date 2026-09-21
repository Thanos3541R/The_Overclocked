"""Driver Impairment Inspector — Static Image & Video Diagnostic Tool.

Run on any photo or video file to analyze biological markers of intoxication:
    python inspect_driver.py --image path/to/driver_photo.jpg
    python inspect_driver.py --video path/to/driving_clip.mp4

Outputs:
  1. Detailed console diagnostic card with biomarker breakdown.
  2. Annotated output image/video with diagnosis badge and feature telemetry.
"""
import argparse
import os
import sys
import cv2
import numpy as np

from pipeline.preprocessing import LightingNormalizer
from pipeline.face_mesh_detector import FaceMeshDetector
from pipeline.pnp_normalizer import PnPNormalizer
from pipeline.feature_extractor import FeatureExtractor
from pipeline.classifier import ImpairmentClassifier, ImpairmentAssessment
from utils.landmarks import (
    RIGHT_EYE_EAR_FLAT, LEFT_EYE_EAR_FLAT,
    RIGHT_IRIS_CENTER, LEFT_IRIS_CENTER,
    RIGHT_EYE_CONTOUR, LEFT_EYE_CONTOUR,
    INNER_LIP_UPPER, INNER_LIP_LOWER
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Inspect driver image or video for alcohol & drowsiness impairment"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--image", type=str, help="Path to input face image (JPG, PNG, etc.)")
    group.add_argument("--capture", action="store_true", help="Capture a live snapshot from the webcam and inspect")
    group.add_argument("--video", type=str, help="Path to input video file (MP4, AVI, etc.)")

    parser.add_argument("--output", type=str, default="inspection_result.jpg",
                        help="Path to save annotated inspection output")
    parser.add_argument("--headless", action="store_true",
                        help="Do not display OpenCV popup window")
    return parser.parse_args()


def print_diagnostic_card(assessment: ImpairmentAssessment, ear_l: float, ear_r: float,
                          mar: float, pitch: float, roll: float):
    """Print an executive diagnostic summary to terminal."""
    bar_width = 30
    filled = int(assessment.risk_score / 100.0 * bar_width)
    bar = "#" * filled + "-" * (bar_width - filled)

    status_color = {
        "SOBER": "\033[92m",                  # Green
        "MILD_IMPAIRMENT": "\033[93m",        # Yellow
        "HIGH_RISK_INTOXICATED": "\033[91m",  # Red
    }.get(assessment.classification, "\033[0m")
    reset = "\033[0m"

    print("\n" + "=" * 68)
    print("       DRIVER IMPAIRMENT DIAGNOSTIC REPORT (BIOMARKER ANALYSIS)")
    print("=" * 68)
    print(f" Classification : {status_color}{assessment.classification}{reset}")
    print(f" Risk Score     : [{bar}] {assessment.risk_score:.1f}%")
    print(f" Confidence     : {assessment.confidence * 100:.1f}%\n")

    print(" -- BIOMETRIC TELEMETRY --")
    print(f"  * Left Eye Opening (EAR)  : {ear_l:.3f}")
    print(f"  * Right Eye Opening (EAR) : {ear_r:.3f}")
    print(f"  * Ocular Asymmetry |dEAR| : {abs(ear_l - ear_r):.3f}")
    print(f"  * Mouth Opening (MAR)     : {mar:.3f}")
    print(f"  * Head Posture (Pitch)    : {pitch:+.1f} deg ({'Slumping forward' if pitch < -10 else 'Upright'})")
    print(f"  * Lateral Tilt (Roll)     : {roll:+.1f} deg\n")

    print(" -- BIOMARKER CONTRIBUTIONS --")
    for marker, score in assessment.marker_scores.items():
        marker_name = marker.replace('_', ' ').title()
        print(f"  * {marker_name:<20}: {score:5.1f}% contribution")

    if assessment.primary_indicators:
        print("\n -- CLINICAL OBSERVATIONS / FLAGS --")
        for ind in assessment.primary_indicators:
            print(f"  [!] {ind}")
    print("=" * 68 + "\n")


def draw_static_annotation(frame: np.ndarray, landmarks: np.ndarray,
                           assessment: ImpairmentAssessment, pnp: PnPNormalizer,
                           ear_l: float, ear_r: float, mar: float) -> np.ndarray:
    """Draw professional diagnostic overlay onto the image."""
    vis = frame.copy()
    h, w = vis.shape[:2]

    # Draw eye contours
    r_eye = landmarks[RIGHT_EYE_CONTOUR, :2].astype(np.int32)
    l_eye = landmarks[LEFT_EYE_CONTOUR, :2].astype(np.int32)
    cv2.polylines(vis, [r_eye], True, (0, 255, 255), 1, cv2.LINE_AA)
    cv2.polylines(vis, [l_eye], True, (0, 255, 255), 1, cv2.LINE_AA)

    # Draw iris centers
    r_iris = tuple(landmarks[RIGHT_IRIS_CENTER, :2].astype(int))
    l_iris = tuple(landmarks[LEFT_IRIS_CENTER, :2].astype(int))
    cv2.drawMarker(vis, r_iris, (0, 0, 255), cv2.MARKER_CROSS, 8, 2)
    cv2.drawMarker(vis, l_iris, (0, 0, 255), cv2.MARKER_CROSS, 8, 2)

    # Draw inner mouth contours
    upper_lip = landmarks[INNER_LIP_UPPER, :2].astype(np.int32)
    lower_lip = landmarks[INNER_LIP_LOWER, :2].astype(np.int32)
    cv2.polylines(vis, [upper_lip], False, (255, 100, 0), 1, cv2.LINE_AA)
    cv2.polylines(vis, [lower_lip], False, (255, 100, 0), 1, cv2.LINE_AA)

    # Draw 3D pose axes from nose tip
    if pnp.rvec is not None and pnp.tvec is not None:
        axis_len = 80.0
        axes_3d = np.array([
            [axis_len, 0.0, 0.0],
            [0.0, axis_len, 0.0],
            [0.0, 0.0, axis_len],
            [0.0, 0.0, 0.0]
        ], dtype=np.float64)
        img_pts, _ = cv2.projectPoints(axes_3d, pnp.rvec, pnp.tvec,
                                      pnp.camera_matrix, pnp.dist_coeffs)
        origin = tuple(img_pts[3].ravel().astype(int))
        colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0)]  # X, Y, Z
        for i, color in enumerate(colors):
            end = tuple(img_pts[i].ravel().astype(int))
            cv2.arrowedLine(vis, origin, end, color, 2, cv2.LINE_AA)

    # Header Badge (Diagnosis banner)
    badge_bg = {
        "SOBER": (0, 160, 0),                 # Green
        "MILD_IMPAIRMENT": (0, 165, 255),     # Orange
        "HIGH_RISK_INTOXICATED": (0, 0, 200), # Red
    }.get(assessment.classification, (50, 50, 50))

    banner_text = f"[{assessment.classification}] RISK: {assessment.risk_score:.1f}% (Conf: {assessment.confidence*100:.0f}%)"
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), _ = cv2.getTextSize(banner_text, font, 0.75, 2)
    badge_w = max(tw + 30, 420)
    cv2.rectangle(vis, (15, 15), (15 + badge_w, 15 + th + 25), badge_bg, -1)
    cv2.rectangle(vis, (15, 15), (15 + badge_w, 15 + th + 25), (255, 255, 255), 2)
    cv2.putText(vis, banner_text, (30, 15 + th + 12), font, 0.75, (255, 255, 255), 2, cv2.LINE_AA)

    # Telemetry Box (bottom-left)
    lines = [
        f"EAR: L:{ear_l:.2f} R:{ear_r:.2f} (Droop/Ptosis)",
        f"MAR: {mar:.2f} (Jaw slackness)",
        f"Pose: Pitch:{pnp.get_euler_angles()[0]:+.1f} Roll:{pnp.get_euler_angles()[2]:+.1f}",
    ]
    for ind in assessment.primary_indicators[:2]:
        lines.append(f"! {ind[:42]}")

    ty = h - len(lines) * 24 - 15
    for i, line in enumerate(lines):
        y_pos = ty + i * 24
        (lw, lh), _ = cv2.getTextSize(line, font, 0.5, 1)
        cv2.rectangle(vis, (15, y_pos - lh - 4), (25 + lw, y_pos + 4), (0, 0, 0), -1)
        cv2.putText(vis, line, (20, y_pos), font, 0.5, (0, 255, 255) if line.startswith('!') else (255, 255, 255), 1, cv2.LINE_AA)

    return vis


def inspect_image(image_path: str, output_path: str, headless: bool):
    """Run full inspection on a single static image."""
    if not os.path.exists(image_path):
        print(f"[Error] File not found: {image_path}")
        return

    frame = cv2.imread(image_path)
    if frame is None:
        print(f"[Error] Could not read image: {image_path}")
        return

    h, w = frame.shape[:2]
    print(f"\n[DMS Inspector] Processing '{image_path}' ({w}x{h})...")

    # Initialize pipeline
    prep = LightingNormalizer()
    detector = FaceMeshDetector()
    pnp = PnPNormalizer(w, h)
    classifier = ImpairmentClassifier()

    # Stage 1: Lighting normalization
    processed = prep.process(frame)

    # Stage 2: Face mesh detection
    landmarks = detector.detect(processed)
    if landmarks is None:
        print("[DMS Inspector] NO FACE DETECTED in image. Ensure face is clearly visible and illuminated.")
        detector.close()
        return

    # Stage 3: PnP Pose
    pnp.solve_pose(landmarks)
    euler = pnp.get_euler_angles()
    pitch, yaw, roll = euler if euler is not None else (0.0, 0.0, 0.0)

    # Stage 4: Feature extraction
    ear_r = FeatureExtractor.compute_ear(landmarks, RIGHT_EYE_EAR_FLAT)
    ear_l = FeatureExtractor.compute_ear(landmarks, LEFT_EYE_EAR_FLAT)
    mar = FeatureExtractor.compute_mar(landmarks)

    # Stage 5: Classification
    assessment = classifier.evaluate_static_image(
        frame=frame,
        landmarks=landmarks,
        ear_left=ear_l,
        ear_right=ear_r,
        mar=mar,
        head_pitch=pitch,
        head_roll=roll
    )

    # Terminal Report
    print_diagnostic_card(assessment, ear_l, ear_r, mar, pitch, roll)

    # Visual Annotation
    annotated = draw_static_annotation(frame, landmarks, assessment, pnp, ear_l, ear_r, mar)
    cv2.imwrite(output_path, annotated)
    print(f"[DMS Inspector] Annotated diagnostic image saved to: {output_path}")

    detector.close()

    if not headless:
        cv2.imshow("DMS Driver Inspection", annotated)
        print("[DMS Inspector] Press any key on the image window to close.")
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def capture_from_webcam(output_path: str, headless: bool):
    """Capture a live frame from webcam and run diagnostic inspection."""
    print("[DMS Inspector] Initializing camera for snapshot capture...")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[Error] Could not open webcam index 0.")
        return

    # Let camera warm up / adjust auto-exposure
    for _ in range(10):
        ret, frame = cap.read()
    cap.release()

    if not ret or frame is None:
        print("[Error] Failed to read frame from webcam.")
        return

    raw_snapshot_path = "captured_snapshot.jpg"
    cv2.imwrite(raw_snapshot_path, frame)
    print(f"[DMS Inspector] Snapshot captured and saved to '{raw_snapshot_path}'")
    inspect_image(raw_snapshot_path, output_path, headless)


def main():
    args = parse_args()
    if args.image:
        inspect_image(args.image, args.output, args.headless)
    elif args.capture:
        capture_from_webcam(args.output, args.headless)
    elif args.video:
        print("[DMS Inspector] Video inspection mode. For full continuous stream, use:")
        print(f"  python main.py --source \"{args.video}\" --save-csv")


if __name__ == "__main__":
    main()
