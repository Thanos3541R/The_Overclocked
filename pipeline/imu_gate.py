"""Software-based IMU gating for vehicle shock detection.

When no hardware CAN bus / IMU is available, this module computes a
software fallback: the second derivative (acceleration) of the nose tip
3D translation vector from PnP. If head acceleration exceeds 2.5 m/s²
over a 3-frame window, the system triggers a 150ms feature freeze.

The underlying assumption: if the entire face translates rapidly without
corresponding head rotation, it's likely a vehicle shock (pothole, speed bump)
rather than intentional driver movement.
"""
import time
import numpy as np
from collections import deque
from typing import Deque, Optional, Tuple


class SoftwareIMUGate:
    """Software fallback for vehicle shock detection via nose tip acceleration.

    Decouples true vehicle road shocks (potholes, curb strikes) from voluntary
    driver head movements (mirror checks, shoulder turns):
      1. Applies low-pass smoothing to the 3D translation vector (tvec) to eliminate
         high-frequency landmark tracking noise that corrupts finite differences.
      2. Computes linear acceleration via a 5-point non-uniform Savitzky-Golay
         quadratic polynomial fit over a sliding window (drastically suppressing
         1/dt² noise amplification).
      3. Computes angular rotational velocity from Euler angles.
      4. Only triggers feature freeze if linear acceleration exceeds the threshold
         AND rotational velocity is low (< 20 deg/s), because road shocks produce
         vertical chassis translation without intentional head rotation.
    """

    def __init__(self, accel_threshold: float = 2.5,
                 freeze_duration_ms: float = 150.0,
                 max_rot_vel_for_shock: float = 20.0,
                 alpha: float = 0.3,
                 window_size: int = 5):
        """
        Args:
            accel_threshold: Acceleration threshold in m/s² to trigger freeze.
            freeze_duration_ms: Duration to freeze features after shock (ms).
            max_rot_vel_for_shock: Max head rotation speed (deg/s) for road shock.
            alpha: Low-pass filter weight for translation vector smoothing (0..1).
            window_size: Number of frames for acceleration estimation (default 5).
        """
        self.accel_threshold = accel_threshold
        self.freeze_duration_sec = freeze_duration_ms / 1000.0
        self.max_rot_vel_for_shock = max_rot_vel_for_shock
        self.alpha = alpha
        self.window_size = window_size

        self.t_smooth_prev: Optional[np.ndarray] = None
        self.t_vel_prev: Optional[np.ndarray] = None
        self.last_euler: Optional[np.ndarray] = None
        self.last_timestamp: Optional[float] = None
        self._freeze_until: float = 0.0
        self._history: Deque[Tuple[float, np.ndarray]] = deque(maxlen=self.window_size)

    def _estimate_acceleration(self) -> float:
        """Estimate 3D linear acceleration magnitude (m/s²) using a non-uniform
        quadratic polynomial fit (Savitzky-Golay filter over sliding window).

        Fits p(tau) = a * tau^2 + b * tau + c to the last N points (3 <= N <= window_size),
        where tau = t - t_current <= 0.
        At the current instant (tau = 0):
          Position p(0) = c
          Velocity p'(0) = b
          Acceleration p''(0) = 2*a (in mm/s²)

        This drastically suppresses the 1/dt² noise amplification of raw 2-point
        finite-difference derivatives while correctly tracking true physical road shock.

        Returns:
            Acceleration magnitude in m/s².
        """
        # Prune points outside the temporal locality window (freeze_duration_sec)
        # while retaining at least 3 points for polynomial fitting
        if self._history:
            curr_t = self._history[-1][0]
            while len(self._history) > 3 and (curr_t - self._history[0][0]) > self.freeze_duration_sec:
                self._history.popleft()

        if len(self._history) < 3:
            return 0.0

        times = np.array([item[0] for item in self._history], dtype=np.float64)
        positions = np.array([item[1] for item in self._history], dtype=np.float64)

        # Total time span check to prevent singular Vandermonde matrix
        total_dt = times[-1] - times[0]
        if total_dt < 1e-5:
            return 0.0

        # Shift time so tau=0 corresponds to the current/latest sample
        tau = times - times[-1]

        # Design matrix V: [tau^2, tau, 1]
        V = np.column_stack([tau**2, tau, np.ones_like(tau)])

        try:
            # Solve least-squares: V @ C = positions (shape N x 3) -> C shape (3, 3)
            C, _, _, _ = np.linalg.lstsq(V, positions, rcond=None)
            # Quadratic coefficients for [x, y, z] are row 0 of C
            # Second derivative at tau=0 is 2 * a
            accel_mm_s2 = 2.0 * C[0, :]
            accel_norm_m_s2 = float(np.linalg.norm(accel_mm_s2)) / 1000.0  # mm/s² -> m/s²
            return accel_norm_m_s2
        except Exception:
            return 0.0

    def update(self, timestamp: float,
               tvec: Optional[np.ndarray],
               euler_angles: Optional[Tuple[float, float, float]] = None) -> bool:
        """Update with new nose tip position and check for vehicle shock.

        Args:
            timestamp: Current frame timestamp (seconds).
            tvec: Translation vector from PnP (3,) or (3,1), or None.
            euler_angles: (pitch, yaw, roll) in degrees from PnP, or None.

        Returns:
            True if currently in freeze mode (shock detected),
            False if operating normally.
        """
        if tvec is None:
            return timestamp < self._freeze_until

        pos = np.asarray(tvec, dtype=np.float64).flatten()[:3]

        if self.last_timestamp is None or self.t_smooth_prev is None:
            self.t_smooth_prev = pos.copy()
            self._history.append((timestamp, self.t_smooth_prev.copy()))
            self.t_vel_prev = np.zeros(3, dtype=np.float64)
            if euler_angles is not None:
                self.last_euler = np.asarray(euler_angles, dtype=np.float64).copy()
            self.last_timestamp = timestamp
            return False

        dt = max(timestamp - self.last_timestamp, 1e-4)

        # 1. Low-pass filter tvec to eliminate landmark sub-pixel jitter
        t_smooth = self.alpha * pos + (1.0 - self.alpha) * self.t_smooth_prev
        self._history.append((timestamp, t_smooth.copy()))

        # 2. Savitzky-Golay polynomial acceleration estimate (m/s²)
        linear_accel_m_s2 = self._estimate_acceleration()

        # 3. Compute head rotational velocity (deg/s) if euler angles available
        rot_vel = 0.0
        if euler_angles is not None:
            curr_euler = np.asarray(euler_angles, dtype=np.float64)
            if self.last_euler is not None:
                rot_vel = float(np.linalg.norm((curr_euler - self.last_euler) / dt))
            self.last_euler = curr_euler.copy()

        # 4. Decoupled Shock Gate: Only trigger shock if linear acceleration is high
        # AND rotational velocity is low (rules out intentional head turns).
        # Only arm a new freeze if not already within an active freeze window.
        is_pothole_shock = (linear_accel_m_s2 > self.accel_threshold) and (
            euler_angles is None or rot_vel < self.max_rot_vel_for_shock
        )

        if is_pothole_shock and timestamp >= self._freeze_until:
            self._freeze_until = timestamp + self.freeze_duration_sec

        # Update state
        self.t_smooth_prev = t_smooth
        self.last_timestamp = timestamp

        return timestamp < self._freeze_until

    @property
    def is_frozen(self) -> bool:
        """Check if currently in freeze state."""
        return time.time() < self._freeze_until

    def reset(self):
        """Reset gate state."""
        self.t_smooth_prev = None
        self.t_vel_prev = None
        self.last_euler = None
        self.last_timestamp = None
        self._freeze_until = 0.0
        self._history.clear()

