"""Head pose estimation and gating for driver monitoring.

Extracts yaw, pitch, roll from PnP rotation and applies gating rules:
  - |yaw| > 35°: face too far sideways → pause feature extraction
  - pitch < -30°: head tilted too far down → pause
  - pitch > 25°: head tilted too far up → pause

When gated, the system holds the last known good feature values
rather than outputting garbage from extreme angles.
"""
from typing import Optional, Tuple
from dataclasses import dataclass


@dataclass
class PoseState:
    """Current head pose state."""
    yaw: float = 0.0      # degrees, + = looking left (subject's left)
    pitch: float = 0.0     # degrees, + = looking up
    roll: float = 0.0      # degrees, + = tilting right
    face_valid: bool = True
    gating_reason: str = ""


class HeadPoseGate:
    """Head pose gating for rejecting extreme face angles."""

    def __init__(self, yaw_limit: float = 35.0,
                 pitch_down_limit: float = -30.0,
                 pitch_up_limit: float = 25.0,
                 yaw_offset: float = 0.0,
                 pitch_offset: float = 0.0,
                 roll_offset: float = 0.0):
        """
        Args:
            yaw_limit: Maximum absolute yaw before gating (degrees).
            pitch_down_limit: Minimum pitch before gating (degrees, negative = down).
            pitch_up_limit: Maximum pitch before gating (degrees).
            yaw_offset: Baseline yaw offset from off-axis mounting (degrees).
            pitch_offset: Baseline pitch offset from mounting angle (degrees).
            roll_offset: Baseline roll offset from mounting angle (degrees).
        """
        self.yaw_limit = yaw_limit
        self.pitch_down_limit = pitch_down_limit
        self.pitch_up_limit = pitch_up_limit
        self.yaw_offset = float(yaw_offset)
        self.pitch_offset = float(pitch_offset)
        self.roll_offset = float(roll_offset)
        self._last_valid_state = PoseState()

    def check(self, euler_angles: Optional[Tuple[float, float, float]]) -> PoseState:
        """Check head pose and determine if face is in valid range.

        Args:
            euler_angles: (pitch, yaw, roll) in degrees from PnP,
                          or None if PnP failed.

        Returns:
            PoseState with face_valid=True if within limits.
        """
        if euler_angles is None:
            return PoseState(
                yaw=self._last_valid_state.yaw,
                pitch=self._last_valid_state.pitch,
                roll=self._last_valid_state.roll,
                face_valid=False,
                gating_reason="no_pose",
            )

        pitch, yaw, roll = euler_angles
        state = PoseState(yaw=yaw, pitch=pitch, roll=roll)

        rel_pitch = pitch - self.pitch_offset
        rel_yaw = yaw - self.yaw_offset

        # Check gating conditions
        if abs(rel_yaw) > self.yaw_limit:
            state.face_valid = False
            state.gating_reason = f"yaw={rel_yaw:.1f}° exceeds ±{self.yaw_limit}°"
        elif rel_pitch < self.pitch_down_limit:
            state.face_valid = False
            state.gating_reason = f"pitch={rel_pitch:.1f}° below {self.pitch_down_limit}°"
        elif rel_pitch > self.pitch_up_limit:
            state.face_valid = False
            state.gating_reason = f"pitch={rel_pitch:.1f}° above {self.pitch_up_limit}°"
        else:
            state.face_valid = True
            state.gating_reason = ""
            self._last_valid_state = state

        return state

    def reset(self) -> None:
        """Reset head pose gating state to default."""
        self._last_valid_state = PoseState()

    @property
    def last_valid_state(self) -> PoseState:
        """Return the last state where face was within valid range."""
        return self._last_valid_state
