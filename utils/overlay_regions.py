"""Landmark regions and connection definitions for color-coded HUD overlay.

Defines static NumPy index arrays and connection sets once at module startup:
  1. Right eyelid contour (viewer's left): Cyan
  2. Left eyelid contour (viewer's right): Sky Blue
  3. Right iris + eye corners: Amber / Gold
  4. Left iris + eye corners: Spring Green (distinct shade from right eye)
  5. PnP rigid anchors: Magenta
  6. Eyebrows: Orchid
  7. Lips: Coral
  8. Cheek patches: Rosy Peach
  9. Full mesh tessellation: Muted Gray

All coordinates and connection indices reuse definitions from `utils.landmarks`
and MediaPipe Tasks FaceLandmarksConnections.
Drawing is vectorized into batch cv2.polylines calls per region (zero per-landmark loops).
"""
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import cv2
import numpy as np

from utils.landmarks import (
    RIGHT_EYE_CONTOUR,
    LEFT_EYE_CONTOUR,
    RIGHT_IRIS_CENTER,
    LEFT_IRIS_CENTER,
    RIGHT_IRIS_RING,
    LEFT_IRIS_RING,
    PNP_LANDMARK_INDICES,
    RIGHT_CHEEK,
    LEFT_CHEEK,
    CHEEK_LANDMARKS,
    RIGHT_EYE_OUTER,
    LEFT_EYE_OUTER,
    NOSE_TIP,
    CHIN,
    RIGHT_MOUTH_CORNER,
    LEFT_MOUTH_CORNER,
)

# Inner eye canthi indices (P4 in EAR)
RIGHT_EYE_INNER = 133
LEFT_EYE_INNER = 362


# ═══════════════════════════════════════════════════════════════════
# MediaPipe Connection Sets Extraction with Static Fallbacks
# ═══════════════════════════════════════════════════════════════════

def _extract_mp_connections(name: str, fallback: List[Tuple[int, int]]) -> np.ndarray:
    """Safely extract connection pairs from MediaPipe Tasks or fallback."""
    try:
        import mediapipe as mp
        if hasattr(mp, "tasks") and hasattr(mp.tasks, "vision") and hasattr(mp.tasks.vision, "FaceLandmarksConnections"):
            conn_cls = mp.tasks.vision.FaceLandmarksConnections
            if hasattr(conn_cls, name):
                raw = getattr(conn_cls, name)
                return np.array([(c.start, c.end) for c in raw], dtype=np.int32)
        elif hasattr(mp, "solutions") and hasattr(mp.solutions, "face_mesh"):
            sol_name = "FACEMESH_" + name.replace("FACE_LANDMARKS_", "")
            if hasattr(mp.solutions.face_mesh, sol_name):
                raw = getattr(mp.solutions.face_mesh, sol_name)
                return np.array(list(raw), dtype=np.int32)
    except Exception:
        pass
    return np.array(fallback, dtype=np.int32)


# Fallback connection pairs for core regions
_FALLBACK_RIGHT_EYE = [
    (33, 7), (7, 163), (163, 144), (144, 145), (145, 153), (153, 154),
    (154, 155), (155, 133), (33, 246), (246, 161), (161, 160), (160, 159),
    (159, 158), (158, 157), (157, 173), (173, 133)
]
_FALLBACK_LEFT_EYE = [
    (263, 249), (249, 390), (390, 373), (373, 374), (374, 380), (380, 381),
    (381, 382), (382, 362), (263, 466), (466, 388), (388, 387), (387, 386),
    (386, 385), (385, 384), (384, 398), (398, 362)
]
_FALLBACK_RIGHT_EYEBROW = [
    (46, 53), (53, 52), (52, 65), (65, 55), (70, 63), (63, 105), (105, 66), (66, 107)
]
_FALLBACK_LEFT_EYEBROW = [
    (276, 283), (283, 282), (282, 295), (295, 285), (300, 293), (293, 334), (334, 296), (296, 336)
]
_FALLBACK_LIPS = [
    (61, 146), (146, 91), (91, 181), (181, 84), (84, 17), (17, 314), (314, 405), (405, 321),
    (321, 375), (375, 291), (61, 185), (185, 40), (40, 39), (39, 37), (37, 0), (0, 267),
    (267, 269), (269, 270), (270, 409), (409, 291), (78, 95), (95, 88), (88, 178), (178, 87),
    (87, 14), (14, 317), (317, 402), (402, 318), (318, 324), (324, 308), (78, 191), (191, 80),
    (80, 81), (81, 82), (82, 13), (13, 312), (312, 311), (311, 310), (310, 415), (415, 308)
]


# Static connection arrays extracted at startup
RIGHT_EYE_CONNECTIONS = _extract_mp_connections("FACE_LANDMARKS_RIGHT_EYE", _FALLBACK_RIGHT_EYE)
LEFT_EYE_CONNECTIONS = _extract_mp_connections("FACE_LANDMARKS_LEFT_EYE", _FALLBACK_LEFT_EYE)
RIGHT_EYEBROW_CONNECTIONS = _extract_mp_connections("FACE_LANDMARKS_RIGHT_EYEBROW", _FALLBACK_RIGHT_EYEBROW)
LEFT_EYEBROW_CONNECTIONS = _extract_mp_connections("FACE_LANDMARKS_LEFT_EYEBROW", _FALLBACK_LEFT_EYEBROW)
EYEBROW_CONNECTIONS = np.vstack([RIGHT_EYEBROW_CONNECTIONS, LEFT_EYEBROW_CONNECTIONS]) if (
    len(RIGHT_EYEBROW_CONNECTIONS) > 0 and len(LEFT_EYEBROW_CONNECTIONS) > 0
) else np.zeros((0, 2), dtype=np.int32)
LIPS_CONNECTIONS = _extract_mp_connections("FACE_LANDMARKS_LIPS", _FALLBACK_LIPS)
TESSELATION_CONNECTIONS = _extract_mp_connections("FACE_LANDMARKS_TESSELATION", [])

# Iris ring connections (4 edges per iris)
RIGHT_IRIS_CONNECTIONS = np.array([
    (RIGHT_IRIS_RING[0], RIGHT_IRIS_RING[1]),
    (RIGHT_IRIS_RING[1], RIGHT_IRIS_RING[2]),
    (RIGHT_IRIS_RING[2], RIGHT_IRIS_RING[3]),
    (RIGHT_IRIS_RING[3], RIGHT_IRIS_RING[0]),
], dtype=np.int32)

LEFT_IRIS_CONNECTIONS = np.array([
    (LEFT_IRIS_RING[0], LEFT_IRIS_RING[1]),
    (LEFT_IRIS_RING[1], LEFT_IRIS_RING[2]),
    (LEFT_IRIS_RING[2], LEFT_IRIS_RING[3]),
    (LEFT_IRIS_RING[3], LEFT_IRIS_RING[0]),
], dtype=np.int32)

# Cheek patch polygon contours (5 edges per cheek)
RIGHT_CHEEK_CONNECTIONS = np.array([
    (RIGHT_CHEEK[0], RIGHT_CHEEK[1]),
    (RIGHT_CHEEK[1], RIGHT_CHEEK[2]),
    (RIGHT_CHEEK[2], RIGHT_CHEEK[3]),
    (RIGHT_CHEEK[3], RIGHT_CHEEK[4]),
    (RIGHT_CHEEK[4], RIGHT_CHEEK[0]),
], dtype=np.int32)

LEFT_CHEEK_CONNECTIONS = np.array([
    (LEFT_CHEEK[0], LEFT_CHEEK[1]),
    (LEFT_CHEEK[1], LEFT_CHEEK[2]),
    (LEFT_CHEEK[2], LEFT_CHEEK[3]),
    (LEFT_CHEEK[3], LEFT_CHEEK[4]),
    (LEFT_CHEEK[4], LEFT_CHEEK[0]),
], dtype=np.int32)

CHEEK_CONNECTIONS = np.vstack([RIGHT_CHEEK_CONNECTIONS, LEFT_CHEEK_CONNECTIONS])


# ═══════════════════════════════════════════════════════════════════
# Landmark Overlay Region Class
# ═══════════════════════════════════════════════════════════════════

def _dim(color: Tuple[int, int, int], factor: float = 0.35) -> Tuple[int, int, int]:
    """Dim a BGR color tuple for gated/frozen frames."""
    return tuple(max(0, min(255, int(c * factor))) for c in color)  # type: ignore


@dataclass(frozen=True)
class OverlayRegion:
    """Encapsulates a color-coded facial region for vectorized batch rendering."""
    name: str
    connections: np.ndarray        # shape (M, 2), int32
    point_indices: np.ndarray      # shape (K,), int32
    point_segments: np.ndarray     # shape (K, 2), int32 - precomputed [idx, idx] segments
    color: Tuple[int, int, int]    # BGR format
    dimmed_color: Tuple[int, int, int]
    line_thickness: int = 1
    point_thickness: int = 3

    def draw(self, frame: np.ndarray, pts: np.ndarray, dimmed: bool = False) -> None:
        """Render all lines and landmark points in single OpenCV batch calls.

        Zero per-landmark Python loops: uses vectorized NumPy slicing into pts.
        """
        c = self.dimmed_color if dimmed else self.color
        # Batch draw lines
        if len(self.connections) > 0:
            lines = pts[self.connections]
            cv2.polylines(frame, lines, isClosed=False, color=c,
                          thickness=self.line_thickness, lineType=cv2.LINE_AA)
        # Batch draw landmark anchor points
        if len(self.point_segments) > 0:
            pt_segs = pts[self.point_segments]
            cv2.polylines(frame, pt_segs, isClosed=False, color=c,
                          thickness=self.point_thickness, lineType=cv2.LINE_AA)


def _make_region(
    name: str,
    connections: np.ndarray,
    point_indices: Optional[List[int]],
    color: Tuple[int, int, int],
    line_thickness: int = 1,
    point_thickness: int = 3,
) -> OverlayRegion:
    """Helper to build an OverlayRegion with precomputed point segments."""
    pts_arr = np.array(point_indices if point_indices is not None else [], dtype=np.int32)
    pt_segs = np.column_stack([pts_arr, pts_arr]) if len(pts_arr) > 0 else np.zeros((0, 2), dtype=np.int32)
    return OverlayRegion(
        name=name,
        connections=connections,
        point_indices=pts_arr,
        point_segments=pt_segs,
        color=color,
        dimmed_color=_dim(color),
        line_thickness=line_thickness,
        point_thickness=point_thickness,
    )


# ═══════════════════════════════════════════════════════════════════
# Definition of the 8 Color-Coded Key Regions + Full Mesh
# ═══════════════════════════════════════════════════════════════════

# 1. Right Eyelid Contour (viewer's left) - Cyan
RIGHT_EYELID_REGION = _make_region(
    name="R Eyelid",
    connections=RIGHT_EYE_CONNECTIONS,
    point_indices=None,
    color=(255, 255, 0),  # Cyan
    line_thickness=1,
)

# 2. Left Eyelid Contour (viewer's right) - Sky Blue
LEFT_EYELID_REGION = _make_region(
    name="L Eyelid",
    connections=LEFT_EYE_CONNECTIONS,
    point_indices=None,
    color=(255, 180, 0),  # Sky Blue
    line_thickness=1,
)

# 3. Right Iris + Eye Corners (viewer's left) - Amber / Gold
# Iris center (473), outer canthus (33), inner canthus (133)
RIGHT_IRIS_CORNERS_REGION = _make_region(
    name="R Iris/Corners",
    connections=RIGHT_IRIS_CONNECTIONS,
    point_indices=[RIGHT_IRIS_CENTER, RIGHT_EYE_OUTER, RIGHT_EYE_INNER],
    color=(0, 215, 255),  # Amber / Gold
    line_thickness=1,
    point_thickness=3,
)

# 4. Left Iris + Eye Corners (viewer's right) - Spring Green
# Iris center (468), outer canthus (263), inner canthus (362)
# Distinct shade from right eye
LEFT_IRIS_CORNERS_REGION = _make_region(
    name="L Iris/Corners",
    connections=LEFT_IRIS_CONNECTIONS,
    point_indices=[LEFT_IRIS_CENTER, LEFT_EYE_OUTER, LEFT_EYE_INNER],
    color=(50, 255, 120),  # Spring Green
    line_thickness=1,
    point_thickness=3,
)

# 5. PnP Rigid Anchors - Magenta
# [Nose tip (4), Chin (152), R eye outer (33), L eye outer (263), R mouth (61), L mouth (291)]
PNP_ANCHORS_REGION = _make_region(
    name="PnP Anchors",
    connections=np.zeros((0, 2), dtype=np.int32),
    point_indices=PNP_LANDMARK_INDICES,
    color=(255, 0, 255),  # Magenta
    line_thickness=1,
    point_thickness=4,    # Prominent anchor points
)

# 6. Eyebrows - Orchid
EYEBROWS_REGION = _make_region(
    name="Eyebrows",
    connections=EYEBROW_CONNECTIONS,
    point_indices=None,
    color=(210, 130, 180),  # Orchid
    line_thickness=1,
)

# 7. Lips - Coral
LIPS_REGION = _make_region(
    name="Lips",
    connections=LIPS_CONNECTIONS,
    point_indices=None,
    color=(100, 80, 255),  # Coral
    line_thickness=1,
)

# 8. Cheek Patches - Rosy Peach
CHEEK_PATCHES_REGION = _make_region(
    name="Cheek Patches",
    connections=CHEEK_CONNECTIONS,
    point_indices=CHEEK_LANDMARKS,
    color=(160, 160, 255),  # Rosy Peach
    line_thickness=1,
    point_thickness=2,
)

# Full 3D Face Mesh Wireframe - Muted Gray
FULL_MESH_REGION = _make_region(
    name="Full Mesh",
    connections=TESSELATION_CONNECTIONS,
    point_indices=None,
    color=(140, 140, 140),  # Muted Gray
    line_thickness=1,
)


# Ordered list of the 8 key regions for HUD rendering
KEY_REGIONS: Tuple[OverlayRegion, ...] = (
    RIGHT_EYELID_REGION,
    LEFT_EYELID_REGION,
    RIGHT_IRIS_CORNERS_REGION,
    LEFT_IRIS_CORNERS_REGION,
    PNP_ANCHORS_REGION,
    EYEBROWS_REGION,
    LIPS_REGION,
    CHEEK_PATCHES_REGION,
)

# All regions including full mesh
ALL_REGIONS: Tuple[OverlayRegion, ...] = KEY_REGIONS + (FULL_MESH_REGION,)
