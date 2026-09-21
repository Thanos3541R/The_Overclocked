"""Vectorized 1-Euro adaptive low-pass filter (Casiez et al., CHI 2012).

Designed for real-time smoothing of facial landmark coordinate arrays.
At low motion speeds, increases smoothing to eliminate AI detector jitter.
At high speeds, reduces smoothing to zero to prevent tracking lag.

This implementation is fully vectorized with NumPy — no Python loops
over individual landmarks or coordinates.
"""
import numpy as np


class VectorizedOneEuroFilter:
    """1-Euro filter operating on (N, D) arrays (e.g., 478 landmarks × 3 coords).

    All landmarks are filtered in a single vectorized operation per frame.
    Typical per-frame cost: < 0.4 ms on CPU for 478 × 3 array.
    """

    def __init__(self, t0: float, x0: np.ndarray,
                 min_cutoff: float = 1.0,
                 beta: float = 0.007,
                 d_cutoff: float = 1.0):
        """
        Args:
            t0: Initial timestamp (seconds).
            x0: Initial signal value, shape (N,) or (N, D).
            min_cutoff: Minimum cutoff frequency (Hz). Lower = more smoothing
                        at rest. Typical: 1.0 for landmarks, 1.5 for EAR.
            beta: Speed coefficient. Higher = faster response to motion.
                  Typical: 0.007 for landmarks, 0.003 for EAR.
            d_cutoff: Derivative cutoff frequency (Hz). Usually 1.0.
        """
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.d_cutoff = float(d_cutoff)

        self.x_prev = np.asarray(x0, dtype=np.float64).copy()
        self.dx_prev = np.zeros_like(self.x_prev, dtype=np.float64)
        self.t_prev = float(t0)

    def _smoothing_factor(self, dt: float, cutoff: np.ndarray) -> np.ndarray:
        """Compute exponential smoothing factor alpha from cutoff frequency."""
        r = 2.0 * np.pi * cutoff * dt
        return r / (r + 1.0)

    def __call__(self, t: float, x: np.ndarray) -> np.ndarray:
        """Filter a new sample.

        Args:
            t: Current timestamp (seconds).
            x: Current signal value, same shape as x0.

        Returns:
            Filtered signal, same shape as x.
        """
        dt = float(t) - self.t_prev
        if dt <= 1e-9:
            return self.x_prev.copy()

        x = np.asarray(x, dtype=np.float64)

        # 1. Filtered derivative (velocity estimate)
        dx = (x - self.x_prev) / dt
        alpha_d = self._smoothing_factor(
            dt, np.full_like(x, self.d_cutoff))
        dx_hat = alpha_d * dx + (1.0 - alpha_d) * self.dx_prev

        # 2. Adaptive cutoff = min_cutoff + beta * |velocity|
        cutoff = self.min_cutoff + self.beta * np.abs(dx_hat)

        # 3. Filter the signal with adaptive alpha
        alpha = self._smoothing_factor(dt, cutoff)
        x_hat = alpha * x + (1.0 - alpha) * self.x_prev

        # Update state
        self.x_prev = x_hat
        self.dx_prev = dx_hat
        self.t_prev = float(t)

        return x_hat

    def reset(self, t: float, x: np.ndarray):
        """Reset filter state (e.g., when face is re-detected after loss)."""
        self.x_prev = np.asarray(x, dtype=np.float64).copy()
        self.dx_prev = np.zeros_like(self.x_prev)
        self.t_prev = float(t)


class OneEuroFilterBank:
    """Convenience wrapper: manages a 1-Euro filter for (478, 3) landmarks
    plus separate scalar filters for EAR and MAR signals.
    """

    def __init__(self, t0: float, x0: np.ndarray,
                 landmark_min_cutoff: float = 1.0,
                 landmark_beta: float = 0.007,
                 ear_min_cutoff: float = 1.5,
                 ear_beta: float = 0.003):
        """
        Args:
            t0: Initial timestamp.
            x0: Initial landmarks, shape (N, 3).
            landmark_min_cutoff: Smoothing for landmarks.
            landmark_beta: Speed coefficient for landmarks.
            ear_min_cutoff: Smoothing for EAR/MAR scalars.
            ear_beta: Speed coefficient for EAR/MAR.
        """
        self.landmark_filter = VectorizedOneEuroFilter(
            t0, x0,
            min_cutoff=landmark_min_cutoff,
            beta=landmark_beta,
        )
        self._ear_filters = {}  # lazy-initialized scalar filters
        self._ear_min_cutoff = ear_min_cutoff
        self._ear_beta = ear_beta

    def filter_landmarks(self, t: float, landmarks: np.ndarray) -> np.ndarray:
        """Filter landmark array."""
        return self.landmark_filter(t, landmarks)

    def filter_scalar(self, t: float, value: float, key: str) -> float:
        """Filter a named scalar signal (e.g., 'ear_left', 'mar')."""
        if key not in self._ear_filters:
            self._ear_filters[key] = VectorizedOneEuroFilter(
                t, np.array([value]),
                min_cutoff=self._ear_min_cutoff,
                beta=self._ear_beta,
            )
            return value
        result = self._ear_filters[key](t, np.array([value]))
        return float(result[0])

    def reset(self, t: float, landmarks: np.ndarray):
        """Reset all filters."""
        self.landmark_filter.reset(t, landmarks)
        self._ear_filters.clear()
