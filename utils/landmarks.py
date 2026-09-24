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

# Ordered list matching the canonical 3D model in pnp_normalizer.py
PNP_LANDMARK_INDICES = [NOSE_TIP, CHIN, RIGHT_EYE_OUTER,
                        LEFT_EYE_OUTER, RIGHT_MOUTH_CORNER, LEFT_MOUTH_CORNER]

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

# ═══════════════════════════════════════════════════════════════════
# Cheek Landmarks for Facial Flushing / Vasodilation
# ═══════════════════════════════════════════════════════════════════
# Subject's RIGHT cheek (viewer's left)
RIGHT_CHEEK = [117, 118, 101, 205, 50]

# Subject's LEFT cheek (viewer's right)
LEFT_CHEEK = [346, 347, 330, 425, 280]

CHEEK_LANDMARKS = RIGHT_CHEEK + LEFT_CHEEK

