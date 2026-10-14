"""Vectorized 1-Euro adaptive low-pass filter (Casiez et al., CHI 2012).

Designed for real-time smoothing of facial landmark coordinate arrays.
At low motion speeds, increases smoothing to eliminate AI detector jitter.
At high speeds, reduces smoothing to zero to prevent tracking lag.

Split-Bandwidth Oculomotor Preservation:
  During a blink, the eyelid closes and reverses direction to reopen within 50-100ms.
  A low derivative cutoff (d_cutoff = 1.0 Hz) causes the closing and opening velocity
  vectors to cancel out at the reversal point, dropping the filter cutoff back to
  min_cutoff (1.0 Hz) and crushing blink upstroke velocity by >75%.
  OneEuroFilterBank solves this in a single vectorized pass by assigning a high-bandwidth
  profile (min_cutoff=8.0 Hz, beta=0.80, d_cutoff=10.0 Hz) to the 18 fast-moving eyelid
  and iris landmarks (DYNAMIC_EYELID_IRIS_INDICES) while keeping all structural and
  PnP skull-anchor landmarks heavily smoothed (min_cutoff=1.0 Hz, beta=0.007, d_cutoff=1.0 Hz).

This implementation is fully vectorized with NumPy — no Python loops
over individual landmarks or coordinates.
"""
from typing import Optional, Union
import numpy as np

from utils.landmarks import (
    DYNAMIC_EYELID_INDICES,
    DYNAMIC_IRIS_INDICES,
    DYNAMIC_EYELID_IRIS_INDICES,
)


class VectorizedOneEuroFilter:
    """1-Euro filter operating on (N, D) arrays (e.g., 478 landmarks × 3 coords).

    All landmarks are filtered in a single vectorized operation per frame.
    Supports both scalar and per-landmark array parameters for min_cutoff, beta, and d_cutoff.
    Typical per-frame cost: < 0.4 ms on CPU for 478 × 3 array.
    """

    def __init__(self, t0: float, x0: np.ndarray,
                 min_cutoff: Union[float, np.ndarray] = 1.0,
                 beta: Union[float, np.ndarray] = 0.007,
                 d_cutoff: Union[float, np.ndarray] = 1.0):
        """
        Args:
            t0: Initial timestamp (seconds).
            x0: Initial signal value, shape (N,) or (N, D).
            min_cutoff: Minimum cutoff frequency (Hz), scalar or broadcastable array.
                        Lower = more smoothing at rest. Typical: 1.0 for landmarks, 1.5 for EAR.
            beta: Speed coefficient, scalar or broadcastable array.
                  Higher = faster response to motion. Typical: 0.007 for landmarks, 0.003 for EAR.
            d_cutoff: Derivative cutoff frequency (Hz), scalar or broadcastable array. Usually 1.0.
        """
        self.x_prev = np.asarray(x0, dtype=np.float64).copy()
        self.dx_prev = np.zeros_like(self.x_prev, dtype=np.float64)
        self.t_prev = float(t0)

        self.min_cutoff = (
            np.asarray(min_cutoff, dtype=np.float64)
            if isinstance(min_cutoff, np.ndarray)
            else float(min_cutoff)
        )
        self.beta = (
            np.asarray(beta, dtype=np.float64)
            if isinstance(beta, np.ndarray)
            else float(beta)
        )
        self.d_cutoff = (
            np.asarray(d_cutoff, dtype=np.float64)
            if isinstance(d_cutoff, np.ndarray)
            else float(d_cutoff)
        )

    @staticmethod
    def _smoothing_factor(dt: float, cutoff: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
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

        # If input shape changed dynamically, re-anchor state safely
        if (x.shape != self.x_prev.shape or
                (isinstance(self.min_cutoff, np.ndarray) and
                 self.min_cutoff.shape[0] != x.shape[0])):
            self.reset(t, x)
            return self.x_prev.copy()

        # 1. Filtered derivative (velocity estimate)
        dx = (x - self.x_prev) / dt
        alpha_d = self._smoothing_factor(dt, self.d_cutoff)
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
        if (isinstance(self.min_cutoff, np.ndarray) and self.x_prev.ndim >= 1 and
                self.min_cutoff.shape[0] != self.x_prev.shape[0]):
            n_curr = self.min_cutoff.shape[0]
            n_new = self.x_prev.shape[0]
            if n_new < n_curr:
                self.min_cutoff = self.min_cutoff[:n_new].copy()
                self.beta = self.beta[:n_new].copy()
                self.d_cutoff = self.d_cutoff[:n_new].copy()
            else:
                pad_len = n_new - n_curr
                last_min = self.min_cutoff[-1:] if n_curr > 0 else np.array([[1.0]])
                last_beta = self.beta[-1:] if n_curr > 0 else np.array([[0.007]])
                last_d = self.d_cutoff[-1:] if n_curr > 0 else np.array([[1.0]])
                self.min_cutoff = np.vstack([self.min_cutoff, np.repeat(last_min, pad_len, axis=0)])
                self.beta = np.vstack([self.beta, np.repeat(last_beta, pad_len, axis=0)])
                self.d_cutoff = np.vstack([self.d_cutoff, np.repeat(last_d, pad_len, axis=0)])


class OneEuroFilterBank:
    """Manages a split-bandwidth vectorized 1-Euro filter for (478, 3) landmarks
    plus optional scalar filters.

    Structural/PnP landmarks use low cutoff (0.7 Hz) to eliminate pose chatter,
    eyelid landmarks use (3.0 Hz min_cutoff, 0.80 beta, 10.0 Hz d_cutoff) for fast blink kinetics,
    and iris landmarks use (1.5 Hz min_cutoff, 0.40 beta, 5.0 Hz d_cutoff) to eliminate sub-pixel
    tremor while tracking fast saccades.
    """

    def __init__(self, t0: float, x0: np.ndarray,
                 landmark_min_cutoff: float = 0.7,
                 landmark_beta: float = 0.007,
                 landmark_d_cutoff: float = 1.0,
                 eyelid_min_cutoff: float = 3.0,
                 eyelid_beta: float = 0.80,
                 eyelid_d_cutoff: float = 10.0,
                 iris_min_cutoff: float = 1.5,
                 iris_beta: float = 0.40,
                 iris_d_cutoff: float = 5.0,
                 preserve_oculomotor: bool = True,
                 ear_min_cutoff: float = 1.5,
                 ear_beta: float = 0.003,
                 oculomotor_min_cutoff: Optional[float] = None,
                 oculomotor_beta: Optional[float] = None,
                 oculomotor_d_cutoff: Optional[float] = None):
        """
        Args:
            t0: Initial timestamp.
            x0: Initial landmarks, shape (N, 3).
            landmark_min_cutoff: Smoothing for rigid/structural landmarks (default 0.7 Hz).
            landmark_beta: Speed coefficient for rigid/structural landmarks (default 0.007).
            landmark_d_cutoff: Derivative cutoff for rigid/structural landmarks (default 1.0 Hz).
            eyelid_min_cutoff: Smoothing for fast eyelid landmarks (default 3.0 Hz).
            eyelid_beta: Speed coefficient for fast eyelid landmarks (default 0.80).
            eyelid_d_cutoff: Derivative cutoff for fast eyelid landmarks (default 10.0 Hz).
            iris_min_cutoff: Smoothing for iris landmarks (default 1.5 Hz).
            iris_beta: Speed coefficient for iris landmarks (default 0.40).
            iris_d_cutoff: Derivative cutoff for iris landmarks (default 5.0 Hz).
            preserve_oculomotor: If True, applies high-bandwidth profiles to eyelid and iris landmarks.
            ear_min_cutoff: Smoothing for auxiliary scalar signals.
            ear_beta: Speed coefficient for auxiliary scalar signals.
            oculomotor_min_cutoff: Optional override for legacy oculomotor cutoff.
            oculomotor_beta: Optional override for legacy oculomotor beta.
            oculomotor_d_cutoff: Optional override for legacy oculomotor derivative cutoff.
        """
        if oculomotor_min_cutoff is not None:
            eyelid_min_cutoff = oculomotor_min_cutoff
            iris_min_cutoff = oculomotor_min_cutoff
        if oculomotor_beta is not None:
            eyelid_beta = oculomotor_beta
            iris_beta = oculomotor_beta
        if oculomotor_d_cutoff is not None:
            eyelid_d_cutoff = oculomotor_d_cutoff
            iris_d_cutoff = oculomotor_d_cutoff

        x0_arr = np.asarray(x0, dtype=np.float64)
        n_pts = x0_arr.shape[0] if x0_arr.ndim >= 1 else 0

        eyelid_indices = [i for i in DYNAMIC_EYELID_INDICES if i < n_pts]
        iris_indices = [i for i in DYNAMIC_IRIS_INDICES if i < n_pts]

        if preserve_oculomotor and x0_arr.ndim == 2 and (eyelid_indices or iris_indices):
            min_cutoff_vec = np.full((n_pts, 1), float(landmark_min_cutoff), dtype=np.float64)
            beta_vec = np.full((n_pts, 1), float(landmark_beta), dtype=np.float64)
            d_cutoff_vec = np.full((n_pts, 1), float(landmark_d_cutoff), dtype=np.float64)

            if eyelid_indices:
                min_cutoff_vec[eyelid_indices, 0] = float(eyelid_min_cutoff)
                beta_vec[eyelid_indices, 0] = float(eyelid_beta)
                d_cutoff_vec[eyelid_indices, 0] = float(eyelid_d_cutoff)

            if iris_indices:
                min_cutoff_vec[iris_indices, 0] = float(iris_min_cutoff)
                beta_vec[iris_indices, 0] = float(iris_beta)
                d_cutoff_vec[iris_indices, 0] = float(iris_d_cutoff)

            self.landmark_filter = VectorizedOneEuroFilter(
                t0, x0_arr,
                min_cutoff=min_cutoff_vec,
                beta=beta_vec,
                d_cutoff=d_cutoff_vec,
            )
        else:
            self.landmark_filter = VectorizedOneEuroFilter(
                t0, x0_arr,
                min_cutoff=landmark_min_cutoff,
                beta=landmark_beta,
                d_cutoff=landmark_d_cutoff,
            )

        self._ear_filters = {}  # lazy-initialized scalar filters
        self._ear_min_cutoff = ear_min_cutoff
        self._ear_beta = ear_beta
        self._landmark_min_cutoff = landmark_min_cutoff
        self._landmark_beta = landmark_beta
        self._landmark_d_cutoff = landmark_d_cutoff
        self._eyelid_min_cutoff = eyelid_min_cutoff
        self._eyelid_beta = eyelid_beta
        self._eyelid_d_cutoff = eyelid_d_cutoff
        self._iris_min_cutoff = iris_min_cutoff
        self._iris_beta = iris_beta
        self._iris_d_cutoff = iris_d_cutoff
        self._preserve_oculomotor = preserve_oculomotor

    def filter_landmarks(self, t: float, landmarks: np.ndarray) -> np.ndarray:
        """Filter landmark array."""
        lms = np.asarray(landmarks, dtype=np.float64)
        if (lms.shape != self.landmark_filter.x_prev.shape or
                (isinstance(self.landmark_filter.min_cutoff, np.ndarray) and
                 self.landmark_filter.min_cutoff.shape[0] != lms.shape[0])):
            self.reset(t, lms)
            return self.landmark_filter.x_prev.copy()
        return self.landmark_filter(t, lms)

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
        lms = np.asarray(landmarks, dtype=np.float64)
        if (lms.shape != self.landmark_filter.x_prev.shape or
                (isinstance(self.landmark_filter.min_cutoff, np.ndarray) and
                 self.landmark_filter.min_cutoff.shape[0] != lms.shape[0])):
            self.__init__(
                t0=t,
                x0=lms,
                landmark_min_cutoff=self._landmark_min_cutoff,
                landmark_beta=self._landmark_beta,
                landmark_d_cutoff=self._landmark_d_cutoff,
                eyelid_min_cutoff=self._eyelid_min_cutoff,
                eyelid_beta=self._eyelid_beta,
                eyelid_d_cutoff=self._eyelid_d_cutoff,
                iris_min_cutoff=self._iris_min_cutoff,
                iris_beta=self._iris_beta,
                iris_d_cutoff=self._iris_d_cutoff,
                preserve_oculomotor=self._preserve_oculomotor,
                ear_min_cutoff=self._ear_min_cutoff,
                ear_beta=self._ear_beta,
            )
        else:
            self.landmark_filter.reset(t, landmarks)
            self._ear_filters.clear()
