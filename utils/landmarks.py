"""MediaPipe Face Mesh landmark index constants for driver monitoring.

All indices follow MediaPipe's subject-anatomical convention:
  - 'Right' = subject's right eye = viewer's LEFT in an unmirrored camera feed
  - 'Left'  = subject's left eye  = viewer's RIGHT in an unmirrored camera feed

Landmark groups are organized by function:
  - PnP anchors: rigid skull-bound points for head pose / bounce cancellation
  - Eye landmarks: for Eye Aspect Ratio (EAR) and blink detection
  - Iris landmarks: for gaze estimation (requires refine_landmarks=True)
  - Mouth landmarks: for Mouth Aspect Ratio (MAR) and yawn detection
"""
import numpy as np
from typing import Dict, List, Tuple

# ═══════════════════════════════════════════════════════════════════
# PnP Rigid Anchor Points (skull-bound, minimal expression flex)
# ═══════════════════════════════════════════════════════════════════
NOSE_TIP = 4              # Pronasale
CHIN = 152                # Menton
RIGHT_EYE_OUTER = 33      # Subject's right, outer canthus
LEFT_EYE_OUTER = 263      # Subject's left, outer canthus
RIGHT_MOUTH_CORNER = 61   # Right commissure
LEFT_MOUTH_CORNER = 291   # Left commissure
NOSE_BRIDGE = 168         # Rhinion / sellion bridge
MID_EYEBROW = 9           # Glabella / sellion midpoint

RIGHT_EYE_INNER = 133     # Subject's right, inner canthus
LEFT_EYE_INNER = 362      # Subject's left, inner canthus

# Ordered list matching the canonical 3D model in pnp_normalizer.py
PNP_LANDMARK_INDICES = [NOSE_TIP, CHIN, RIGHT_EYE_OUTER,
                        LEFT_EYE_OUTER, RIGHT_MOUTH_CORNER, LEFT_MOUTH_CORNER]

# Expanded 8-point rigid anchor set (adds midline vertical pitch rigidity)
PNP_LANDMARK_INDICES_8 = [NOSE_TIP, CHIN, RIGHT_EYE_OUTER,
                          LEFT_EYE_OUTER, RIGHT_MOUTH_CORNER, LEFT_MOUTH_CORNER,
                          NOSE_BRIDGE, MID_EYEBROW]

# ═══════════════════════════════════════════════════════════════════
# Eye Landmarks for EAR (6-point Soukupová & Čech 2016 formula)
# ═══════════════════════════════════════════════════════════════════
# Format: P1=outer_corner, P2=upper1, P3=upper2, P4=inner_corner, P5=lower2, P6=lower1
#         EAR = (||P2-P6|| + ||P3-P5||) / (2 * ||P1-P4||)

# Subject's RIGHT eye (viewer's left)
RIGHT_EYE_EAR_INDICES = {
    'P1': 33,    # outer corner
    'P2': 160,   # upper eyelid (pair 1)
    'P3': 158,   # upper eyelid (pair 2)
    'P4': 133,   # inner corner
    'P5': 153,   # lower eyelid (pair 2)
    'P6': 144,   # lower eyelid (pair 1)
}

# Subject's LEFT eye (viewer's right)
LEFT_EYE_EAR_INDICES = {
    'P1': 263,   # outer corner
    'P2': 385,   # upper eyelid (pair 1)
    'P3': 387,   # upper eyelid (pair 2)
    'P4': 362,   # inner corner
    'P5': 373,   # lower eyelid (pair 2)
    'P6': 380,   # lower eyelid (pair 1)
}

# Flat arrays for fast numpy indexing
RIGHT_EYE_EAR_FLAT = [33, 160, 158, 133, 153, 144]
LEFT_EYE_EAR_FLAT = [263, 385, 387, 362, 373, 380]

# Full eye contours (for visualization)
RIGHT_EYE_CONTOUR = [33, 7, 163, 144, 145, 153, 154, 155, 133,
                     173, 157, 158, 159, 160, 161, 246]
LEFT_EYE_CONTOUR = [362, 382, 381, 380, 374, 373, 390, 249,
                    263, 466, 388, 387, 386, 385, 384, 398]

# ═══════════════════════════════════════════════════════════════════
# Iris Landmarks (refine_landmarks=True, indices 468–477)
# ═══════════════════════════════════════════════════════════════════
# Subject's LEFT eye iris (viewer's right)
LEFT_IRIS_CENTER = 468
LEFT_IRIS_RING = [469, 470, 471, 472]

# Subject's RIGHT eye iris (viewer's left)
RIGHT_IRIS_CENTER = 473
RIGHT_IRIS_RING = [474, 475, 476, 477]

# Fast-moving oculomotor landmarks (8 vertical eyelid points + 10 iris points)
# Kept on a high-bandwidth filter path so blink upstroke and saccade velocities are not damped
DYNAMIC_EYELID_INDICES = [160, 158, 153, 144, 385, 387, 373, 380]
DYNAMIC_IRIS_INDICES = [468, 469, 470, 471, 472, 473, 474, 475, 476, 477]
DYNAMIC_EYELID_IRIS_INDICES = DYNAMIC_EYELID_INDICES + DYNAMIC_IRIS_INDICES

# ═══════════════════════════════════════════════════════════════════
# Mouth Landmarks for MAR (Mouth Aspect Ratio)
# ═══════════════════════════════════════════════════════════════════
MOUTH_RIGHT = 61           # Right mouth corner
MOUTH_LEFT = 291           # Left mouth corner
UPPER_LIP_INNER = 13       # Inner upper lip midpoint
LOWER_LIP_INNER = 14       # Inner lower lip midpoint

# Dense 3-pair MAR vertical landmarks
# MAR = (||V1_top-V1_bot|| + ||V2_top-V2_bot|| + ||V3_top-V3_bot||) / (3 * ||H_left-H_right||)
MAR_VERTICAL_PAIRS: List[Tuple[int, int]] = [
    (81, 178),    # right vertical pair
    (13, 14),     # center vertical pair (inner lips)
    (311, 402),   # left vertical pair
]
MAR_HORIZONTAL = (61, 291)  # mouth corners

# Inner lip contours (for visualization)
INNER_LIP_UPPER = [78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308]
INNER_LIP_LOWER = [308, 324, 318, 402, 317, 14, 87, 178, 88, 95, 78]

# ═══════════════════════════════════════════════════════════════════
# Canonical 3D Face Model for PnP (mm, origin at nose tip)
# ═══════════════════════════════════════════════════════════════════
# Matches PNP_LANDMARK_INDICES order: nose, chin, R_eye, L_eye, R_mouth, L_mouth
# Coordinates aligned with OpenCV camera frame (+X right, +Y down towards chin, +Z away into face)
CANONICAL_FACE_3D = np.array([
    [0.0,      0.0,      0.0],       # Nose tip (idx 4)
    [0.0,     73.3,     14.4],       # Chin (idx 152)
    [-50.0,  -37.8,     30.0],       # Right eye outer (idx 33) - 100mm outer bi-canthal width
    [50.0,   -37.8,     30.0],       # Left eye outer (idx 263)
    [-33.3,   33.3,     27.8],       # Right mouth (idx 61)
    [33.3,    33.3,     27.8],       # Left mouth (idx 291)
], dtype=np.float64)

# Expanded 8-point canonical 3D face model with midline vertical rigidity (nose bridge + glabella)
CANONICAL_FACE_3D_8 = np.vstack([
    CANONICAL_FACE_3D,
    np.array([
        [0.0,   -22.0,    15.0],       # Nose bridge (idx 168)
        [0.0,   -45.0,    22.0],       # Mid-eyebrow / glabella (idx 9)
    ], dtype=np.float64),
])

# ═══════════════════════════════════════════════════════════════════
# Cheek Landmarks for Facial Flushing / Vasodilation
# ═══════════════════════════════════════════════════════════════════
# Legacy 5-point infraorbital cheek patches (kept for backward compatibility)
RIGHT_CHEEK = [117, 118, 101, 205, 50]
LEFT_CHEEK = [346, 347, 330, 425, 280]
CHEEK_LANDMARKS = RIGHT_CHEEK + LEFT_CHEEK

# Full Cheek Regions (infraorbital rim, malar prominence, zygomatic arch, buccal area)
# Subject's RIGHT cheek (viewer's left) - 22 anatomical landmarks
FULL_RIGHT_CHEEK = [
    116, 117, 118, 101, 123, 50, 205, 147, 187, 207, 216, 213, 192, 138, 214, 212, 172, 58, 132, 93, 234, 127
]

# Subject's LEFT cheek (viewer's right) - 22 anatomical landmarks
FULL_LEFT_CHEEK = [
    345, 346, 347, 330, 352, 280, 425, 376, 411, 427, 436, 433, 416, 367, 434, 432, 397, 288, 361, 323, 454, 356
]

# Combined 44-point full cheek matrix
FULL_CHEEK_LANDMARKS = FULL_RIGHT_CHEEK + FULL_LEFT_CHEEK

# Smooth closed perimeter contours for full cheek outlines (convex-like perimeter loops)
RIGHT_CHEEK_OUTLINE = [101, 216, 212, 172, 58, 132, 93, 234, 127]
LEFT_CHEEK_OUTLINE = [330, 436, 432, 397, 288, 361, 323, 454, 356]

# ═══════════════════════════════════════════════════════════════════
# Forehead Region Landmarks (Hairline Arc, Frontal Patch & Contours)
# ═══════════════════════════════════════════════════════════════════
# 27-landmark continuous closed polygon loop bounding the entire forehead:
# Upper hairline arc (127 -> 10 -> 356) and lower supraorbital / glabella arc (356 -> 9 -> 127)
FOREHEAD_CONTOUR = [
    127, 162, 21, 54, 103, 67, 109, 10, 338, 297, 332, 284, 251, 389, 356,
    300, 293, 334, 296, 336, 9, 107, 66, 105, 63, 70, 127
]

# 20 frontal bone interior landmarks for T-zone tracking and micro-vascular sampling
FOREHEAD_PATCH = [
    10, 151, 9, 8, 108, 69, 104, 68, 109, 67, 103, 54,
    337, 299, 333, 298, 338, 297, 332, 284
]

# Comprehensive vasodilation tracking landmarks (full cheeks + forehead T-zone: 64 points)
VASODILATION_LANDMARKS = FULL_CHEEK_LANDMARKS + FOREHEAD_PATCH

