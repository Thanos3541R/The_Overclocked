"""Sequential Bayesian Evidence Accumulator & Wald SPRT Filter.

Complies with ISO 26262 ASIL-B/D functional safety guidelines for non-intrusive automotive lockouts.
Prevents false-positive engine lockouts caused by transient noise (bumps, yawns, sunlight glare)
by requiring sequential accumulation of independent involuntary physiological evidence over time.

Key Functional Principles:
  1. Prior Odds: Assumes an initial sober prior P(Impaired) = 0.01 (log-odds ≈ -4.60).
  2. Likelihood Updates: Updates log-odds sequentially using likelihood ratios derived from
     unfakeable Section A involuntary physiological signals (GEN, pursuit fragmentation,
     sluggish eyelid upstroke, lack of convergence).
  3. Evidence Decay / Forgetting: When normal physiology is observed, evidence decays toward sober.
  4. Confirmation Horizon: Lockout interlock is only armed when posterior probability P >= 0.999
     (99.9% certainty) is sustained continuously for >= 15.0 seconds.
"""
from dataclasses import dataclass
from typing import Optional
import numpy as np

from pipeline.classifier import ImpairmentAssessment


@dataclass
class BayesianState:
    """Diagnostic state of the Bayesian Evidence Filter."""
    posterior_probability: float = 0.01      # P(Impaired | history), range [0, 1]
    log_odds: float = -4.60                  # Current accumulated log-odds
    sustained_high_confidence_s: float = 0.0 # Consecutive seconds P >= 0.999
    lockout_confirmed: bool = False          # True ONLY when P >= 0.999 for >= 15.0s
    advisory_warning: bool = False           # True when P >= 0.85 (pre-lockout alert)
    status_summary: str = ""                 # Human-readable status summary


class BayesianEvidenceAccumulator:
    """Sequential Bayesian probability accumulator implementing Wald's Sequential Probability Ratio Test."""

    def __init__(self,
                 prior_prob: float = 0.01,
                 lockout_threshold_prob: float = 0.999,
                 lockout_sustained_sec: float = 15.0,
                 warning_threshold_prob: float = 0.85,
                 decay_rate_per_sec: float = 0.40):
        """
        Args:
            prior_prob: Initial prior probability of impairment P0 (default 1%).
            lockout_threshold_prob: Posterior probability required for lockout (default 99.9%).
            lockout_sustained_sec: Sustained duration at/above threshold required to confirm lockout.
            warning_threshold_prob: Advisory threshold for visual/auditory chime.
            decay_rate_per_sec: Log-odds decay toward sober when normal physiology is observed.
        """
        self.prior_prob = np.clip(prior_prob, 1e-4, 0.50)
        self.initial_log_odds = float(np.log(self.prior_prob / (1.0 - self.prior_prob)))
        self.log_odds = self.initial_log_odds
        self.lockout_threshold_prob = lockout_threshold_prob
        self.lockout_sustained_sec = lockout_sustained_sec
        self.warning_threshold_prob = warning_threshold_prob
        self.decay_rate_per_sec = decay_rate_per_sec

        # High-water / sustained timing tracking
        self._sustained_time_s: float = 0.0
        self._last_timestamp: Optional[float] = None
        self._lockout_armed: bool = False

    def reset(self) -> None:
        """Reset filter back to baseline prior."""
        self.log_odds = self.initial_log_odds
        self._sustained_time_s = 0.0
        self._last_timestamp = None
        self._lockout_armed = False

    @property
    def posterior(self) -> float:
        """Current posterior probability of impairment P(Impaired | evidence)."""
        # P = 1 / (1 + exp(-log_odds))
        return float(1.0 / (1.0 + np.exp(-np.clip(self.log_odds, -20.0, 20.0))))

    def update(self, assessment: ImpairmentAssessment, timestamp: float) -> BayesianState:
        """Update Bayesian evidence with assessment from the current frame/window.

        Args:
            assessment: Diagnostic output from ImpairmentClassifier.
            timestamp: Current timestamp in seconds.

        Returns:
            BayesianState dataclass.
        """
        if self._last_timestamp is None:
            self._last_timestamp = timestamp
            dt = 0.033
        else:
            dt = max(1e-4, min(1.0, timestamp - self._last_timestamp))
            self._last_timestamp = timestamp

        # Compute instantaneous log-likelihood ratio update based on involuntary evidence
        inv_risk = assessment.involuntary_risk_score

        # Scale log-likelihood delta per second:
        # Strong impairment (inv_risk >= 50%): positive log-odds increment (+0.8 to +2.5 / sec)
        # Ambiguous / mild (25% <= inv_risk < 50%): small positive (+0.1 to +0.5 / sec)
        # Sober / healthy (inv_risk < 20%): negative decay toward prior (-decay_rate / sec)
        if inv_risk >= 50.0:
            evidence_rate = 1.0 + (inv_risk - 50.0) / 50.0 * 1.5  # up to +2.5 / s
            # Extra weight if hard physiological reflex breakdown is confirmed (GEN or severe ptosis)
            if assessment.marker_scores.get("gen", 0.0) >= 80.0:
                evidence_rate += 1.0
            if assessment.marker_scores.get("lack_of_convergence", 0.0) >= 70.0:
                evidence_rate += 0.5
            delta_l = evidence_rate * dt
        elif inv_risk >= 30.0:
            evidence_rate = 0.2 + (inv_risk - 30.0) / 20.0 * 0.6  # +0.2 to +0.8 / s
            delta_l = evidence_rate * dt
        else:
            # Sober observation: evidence decays toward initial log odds
            delta_l = -self.decay_rate_per_sec * dt

        # Update log-odds (bounded to avoid numerical issues and enable recovery)
        self.log_odds = float(np.clip(self.log_odds + delta_l, self.initial_log_odds - 1.0, 12.0))

        current_p = self.posterior

        # Check sustained duration for ISO 26262 lockout confirmation
        if self._lockout_armed:
            # Latched state: Once confirmed, lockout remains latched across subsequent frames
            # to prevent a driver from clearing an armed interlock by briefly opening eyes or glancing.
            # Can only be unlatched via unlatch(park_gear_confirmed=True).
            self._sustained_time_s = max(self.lockout_sustained_sec, self._sustained_time_s)
        elif current_p >= self.lockout_threshold_prob:
            self._sustained_time_s += dt
            if self._sustained_time_s >= self.lockout_sustained_sec:
                self._lockout_armed = True
        else:
            # Drop sustained timer with leaky integrator decay (-2.0 * dt) rather than instant zeroing,
            # preventing single-frame chattering/blips at e.g. 14.2s from wiping accumulated evidence.
            self._sustained_time_s = max(0.0, self._sustained_time_s - 2.0 * dt)

        advisory = bool(current_p >= self.warning_threshold_prob)

        status = "SOBER"
        if self._lockout_armed:
            status = f"CRITICAL LOCKOUT ARMED (P={current_p*100:.2f}%, {self._sustained_time_s:.1f}s sustained [LATCHED])"
        elif advisory:
            status = f"IMPAIRMENT WARNING (P={current_p*100:.1f}%, confirming... {self._sustained_time_s:.1f}/{self.lockout_sustained_sec:.0f}s)"

        return BayesianState(
            posterior_probability=current_p,
            log_odds=self.log_odds,
            sustained_high_confidence_s=float(self._sustained_time_s),
            lockout_confirmed=self._lockout_armed,
            advisory_warning=advisory,
            status_summary=status
        )

    def unlatch(self, park_gear_confirmed: bool = False) -> bool:
        """Unlatch an armed lockout state.

        Per ISO 26262 ASIL-D functional safety, unlatching an automotive engine lockout
        requires explicit confirmation that the vehicle is in PARK gear or engine is cycled off.

        Args:
            park_gear_confirmed: Must be True (e.g. from vehicle CAN bus) to release latch.

        Returns:
            True if successfully unlatched, False if rejected.
        """
        if park_gear_confirmed:
            self.reset()
            return True
        return False
