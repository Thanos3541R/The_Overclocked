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

    def test_split_bandwidth_smoothing_cutoffs(self):
        """OneEuroFilterBank decouples cutoffs for structural, eyelid, and iris landmarks."""
        from utils.landmarks import DYNAMIC_EYELID_INDICES, DYNAMIC_IRIS_INDICES
        x0 = np.zeros((478, 3))
        bank = OneEuroFilterBank(t0=0.0, x0=x0)

        min_cutoffs = bank.landmark_filter.min_cutoff
        betas = bank.landmark_filter.beta
        d_cutoffs = bank.landmark_filter.d_cutoff

        # All 440 structural landmarks
        all_indices = set(range(478))
        oculomotor_indices = set(DYNAMIC_EYELID_INDICES + DYNAMIC_IRIS_INDICES)
        structural_indices = list(all_indices - oculomotor_indices)
        assert len(structural_indices) == 440
        assert len(DYNAMIC_EYELID_INDICES) == 28
        assert len(DYNAMIC_IRIS_INDICES) == 10

        # Structural landmarks: min_cutoff = 0.7 Hz, beta = 0.007, d_cutoff = 1.0 Hz
        for idx in structural_indices:
            assert min_cutoffs[idx, 0] == 0.7
            assert betas[idx, 0] == 0.007
            assert d_cutoffs[idx, 0] == 1.0

        # Eyelid landmarks: min_cutoff = 2.0 Hz, beta = 0.50, d_cutoff = 8.0 Hz
        for idx in DYNAMIC_EYELID_INDICES:
            assert min_cutoffs[idx, 0] == 2.0
            assert betas[idx, 0] == 0.50
            assert d_cutoffs[idx, 0] == 8.0

        # Iris landmarks: min_cutoff = 1.0 Hz, beta = 0.20, d_cutoff = 3.0 Hz
        for idx in DYNAMIC_IRIS_INDICES:
            assert min_cutoffs[idx, 0] == 1.0
            assert betas[idx, 0] == 0.20
            assert d_cutoffs[idx, 0] == 3.0

    def test_split_bandwidth_smoothing_cutoffs_468_points(self):
        """OneEuroFilterBank decouples cutoffs for 468-point models (refine_landmarks=False)."""
        from utils.landmarks import DYNAMIC_EYELID_INDICES
        x0 = np.zeros((468, 3))
        bank = OneEuroFilterBank(t0=0.0, x0=x0)

        min_cutoffs = bank.landmark_filter.min_cutoff
        betas = bank.landmark_filter.beta
        d_cutoffs = bank.landmark_filter.d_cutoff

        # All 440 structural landmarks
        all_indices = set(range(468))
        eyelid_indices = set(DYNAMIC_EYELID_INDICES)
        structural_indices = list(all_indices - eyelid_indices)
        assert len(structural_indices) == 440
        assert len(DYNAMIC_EYELID_INDICES) == 28

        # Structural landmarks: min_cutoff = 0.7 Hz, beta = 0.007, d_cutoff = 1.0 Hz
        for idx in structural_indices:
            assert min_cutoffs[idx, 0] == 0.7
            assert betas[idx, 0] == 0.007
            assert d_cutoffs[idx, 0] == 1.0

        # Eyelid landmarks: min_cutoff = 2.0 Hz, beta = 0.50, d_cutoff = 8.0 Hz
        for idx in DYNAMIC_EYELID_INDICES:
            assert min_cutoffs[idx, 0] == 2.0
            assert betas[idx, 0] == 0.50
            assert d_cutoffs[idx, 0] == 8.0

    def test_filter_bank_reset_with_differing_shape(self):
        """reset() re-anchors filter state cleanly when landmark shape changes dynamically."""
        bank = OneEuroFilterBank(t0=0.0, x0=np.zeros((478, 3)))
        assert bank.landmark_filter.x_prev.shape == (478, 3)

        # Reset with 468 points (mode change)
        bank.reset(t=1.0, landmarks=np.ones((468, 3)))
        assert bank.landmark_filter.x_prev.shape == (468, 3)
        res = bank.filter_landmarks(1.033, np.ones((468, 3)) * 2.0)
        assert res.shape == (468, 3)


    def test_dynamic_shape_change_without_explicit_reset(self):
        """filter_landmarks automatically re-anchors without broadcast error when shape changes dynamically."""
        bank = OneEuroFilterBank(t0=0.0, x0=np.zeros((478, 3)))
        # Frame 1 with 468 points (MediaPipe fallback without irises)
        out1 = bank.filter_landmarks(1.0, np.ones((468, 3)))
        assert out1.shape == (468, 3)
        # Frame 2 with 468 points - must not crash with broadcast error
        out2 = bank.filter_landmarks(1.033, np.ones((468, 3)) * 2.0)
        assert out2.shape == (468, 3)

        # Switch back to 478 points dynamically
        out3 = bank.filter_landmarks(2.0, np.ones((478, 3)))
        assert out3.shape == (478, 3)
        out4 = bank.filter_landmarks(2.033, np.ones((478, 3)) * 2.0)
        assert out4.shape == (478, 3)

        # Standalone VectorizedOneEuroFilter dynamic shape resilience
        v_filter = VectorizedOneEuroFilter(
            t0=0.0,
            x0=np.zeros((478, 3)),
            min_cutoff=np.ones((478, 1)),
            beta=np.ones((478, 1)) * 0.1,
            d_cutoff=np.ones((478, 1)),
        )
        r1 = v_filter(1.0, np.ones((468, 3)))
        assert r1.shape == (468, 3)
        r2 = v_filter(1.033, np.ones((468, 3)) * 2.0)
        assert r2.shape == (468, 3)

    def test_1d_parameter_array_resilience(self):
        """1D parameter arrays (shape (N,)) cleanly expand and pad without vstack ValueError."""
        v_filter = VectorizedOneEuroFilter(
            t0=0.0,
            x0=np.zeros((10, 2)),
            min_cutoff=np.ones(10),
            beta=np.ones(10) * 0.1,
            d_cutoff=np.ones(10),
        )
        # Expand from 10 to 15 points
        out = v_filter(1.0, np.ones((15, 2)))
        assert out.shape == (15, 2)
        assert v_filter.min_cutoff.shape == (15,)

    def test_nan_and_inf_input_immunity(self):
        """Filter gracefully rejects NaN and Inf inputs without permanently poisoning internal state."""
        bank = OneEuroFilterBank(t0=0.0, x0=np.zeros((478, 3)))
        # Valid frame 1
        valid_out = bank.filter_landmarks(1.0, np.ones((478, 3)))
        assert not np.isnan(valid_out).any()

        # Frame 2 contains NaNs (detector failure / occlusion)
        nan_input = np.full((478, 3), np.nan)
        held_out = bank.filter_landmarks(1.033, nan_input)
        # Should return held previous clean state without NaN corruption
        assert not np.isnan(held_out).any()
        assert np.array_equal(held_out, valid_out)

        # Frame 3 contains valid numbers again - filter must NOT be poisoned
        resumed_out = bank.filter_landmarks(1.066, np.ones((478, 3)) * 2.0)
        assert not np.isnan(resumed_out).any()

        # Scalar filter with NaN
        assert bank.filter_scalar(1.0, float('nan'), "ear_left") != 0.0 or np.isnan(bank.filter_scalar(1.0, float('nan'), "ear_left"))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
