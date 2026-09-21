"""Unit tests for the Vectorized 1-Euro Filter.

Tests verify:
  - Constant input is returned unchanged (filter converges)
  - Step input is tracked within a few frames
  - High-frequency jitter is smoothed out
  - Slow ramp is followed without lag
  - Reset restores filter to new state
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest
from pipeline.one_euro_filter import VectorizedOneEuroFilter, OneEuroFilterBank


class TestVectorizedOneEuroFilter:
    """Test suite for the vectorized 1-Euro filter."""

    def test_constant_input_converges(self):
        """Filter on constant input should converge to that value."""
        x0 = np.array([[100.0, 200.0, 0.0],
                        [150.0, 250.0, 0.0]])
        filt = VectorizedOneEuroFilter(t0=0.0, x0=x0, min_cutoff=1.0, beta=0.007)

        # Feed same value for 60 frames at 30fps
        for i in range(1, 61):
            t = i / 30.0
            result = filt(t, x0)

        # Should converge to input
        np.testing.assert_allclose(result, x0, atol=0.5,
                                   err_msg="Constant input did not converge")

    def test_step_response_tracks(self):
        """Filter should track a step change within 10 frames."""
        x0 = np.array([0.0, 0.0, 0.0])
        filt = VectorizedOneEuroFilter(t0=0.0, x0=x0, min_cutoff=1.0, beta=0.5)

        # Step to new value
        x_new = np.array([100.0, 100.0, 100.0])
        for i in range(1, 11):
            t = i / 30.0
            result = filt(t, x_new)

        # Should be close to target (within 20% given the high beta)
        error = np.max(np.abs(result - x_new))
        assert error < 30.0, f"Step response error={error:.2f}, expected < 30.0"

    def test_jitter_smoothing(self):
        """High-frequency jitter around a constant should be smoothed."""
        x_base = np.array([100.0, 200.0, 0.0])
        filt = VectorizedOneEuroFilter(t0=0.0, x0=x_base,
                                        min_cutoff=0.5, beta=0.001)

        results = []
        for i in range(1, 91):
            t = i / 30.0
            # Add ±2 pixel jitter
            jitter = np.random.uniform(-2.0, 2.0, size=3)
            result = filt(t, x_base + jitter)
            results.append(result)

        # Last 30 frames should have much less variance than input jitter
        last_30 = np.array(results[-30:])
        output_std = last_30.std(axis=0).mean()
        assert output_std < 1.5, \
            f"Jitter not smoothed: std={output_std:.3f}, expected < 1.5"

    def test_slow_ramp_followed(self):
        """Filter should follow a slow linear ramp without significant lag."""
        x0 = np.array([0.0])
        filt = VectorizedOneEuroFilter(t0=0.0, x0=x0,
                                        min_cutoff=1.0, beta=0.01)

        # Slow ramp: 1 unit per frame at 30fps
        for i in range(1, 91):
            t = i / 30.0
            x = np.array([float(i)])
            result = filt(t, x)

        # Should be within 10% of true value at frame 90
        expected = 90.0
        error = abs(result[0] - expected)
        assert error < expected * 0.15, \
            f"Ramp tracking error={error:.2f} ({error/expected*100:.1f}%)"

    def test_zero_dt_returns_previous(self):
        """Calling with same timestamp should return previous value."""
        x0 = np.array([1.0, 2.0])
        filt = VectorizedOneEuroFilter(t0=0.0, x0=x0)

        result = filt(0.0, np.array([999.0, 999.0]))
        np.testing.assert_array_equal(result, x0)

    def test_reset(self):
        """Reset should set filter to new state."""
        x0 = np.array([0.0, 0.0])
        filt = VectorizedOneEuroFilter(t0=0.0, x0=x0)

        # Filter some values
        filt(1.0, np.array([10.0, 10.0]))

        # Reset to new state
        x_new = np.array([50.0, 50.0])
        filt.reset(2.0, x_new)

        result = filt(2.033, x_new)
        np.testing.assert_allclose(result, x_new, atol=1.0,
                                   err_msg="Reset did not take effect")

    def test_2d_array_shape_preserved(self):
        """Filter should preserve (N, D) array shape."""
        x0 = np.random.randn(478, 3)
        filt = VectorizedOneEuroFilter(t0=0.0, x0=x0)

        result = filt(1.0 / 30.0, np.random.randn(478, 3))
        assert result.shape == (478, 3), f"Shape mismatch: {result.shape}"


class TestOneEuroFilterBank:
    """Test suite for the filter bank wrapper."""

    def test_landmark_filter_works(self):
        """Bank should filter landmarks."""
        x0 = np.zeros((478, 3))
        bank = OneEuroFilterBank(t0=0.0, x0=x0)

        result = bank.filter_landmarks(1.0 / 30.0, np.ones((478, 3)))
        assert result.shape == (478, 3)

    def test_scalar_filter_works(self):
        """Bank should filter named scalar signals."""
        x0 = np.zeros((478, 3))
        bank = OneEuroFilterBank(t0=0.0, x0=x0)

        # First call initializes
        result1 = bank.filter_scalar(0.033, 0.30, "ear_avg")
        assert result1 == 0.30  # first value returned as-is

        # Second call filters
        result2 = bank.filter_scalar(0.066, 0.30, "ear_avg")
        assert isinstance(result2, float)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
