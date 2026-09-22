"""Unit tests for sequential Bayesian evidence accumulation (pipeline/bayesian_filter.py)."""
import pytest
import numpy as np

from pipeline.bayesian_filter import BayesianEvidenceAccumulator, BayesianState
from pipeline.classifier import ImpairmentAssessment


def test_bayesian_filter_initial_state():
    bf = BayesianEvidenceAccumulator(prior_prob=0.01, lockout_threshold_prob=0.999, lockout_sustained_sec=15.0)
    assert abs(bf.posterior - 0.01) < 0.005
    assert not bf._lockout_armed


def test_sober_evidence_decays_log_odds():
    bf = BayesianEvidenceAccumulator(prior_prob=0.05, decay_rate_per_sec=0.5)
    
    # Send sober assessments
    sober_assess = ImpairmentAssessment(
        risk_score=5.0,
        involuntary_risk_score=5.0,
        classification="SOBER"
    )

    for i in range(30):
        t = i * 0.1
        state = bf.update(sober_assess, timestamp=t)

    assert state.posterior_probability < 0.05
    assert not state.lockout_confirmed
    assert not state.advisory_warning


def test_intermittent_noise_does_not_trigger_lockout():
    bf = BayesianEvidenceAccumulator(lockout_threshold_prob=0.999, lockout_sustained_sec=15.0)
    
    # 2-second burst of high risk (e.g. pothole shock or false trigger)
    noise_assess = ImpairmentAssessment(
        risk_score=75.0,
        involuntary_risk_score=60.0,
        classification="HIGH_RISK_INTOXICATED"
    )

    for i in range(20):
        t = i * 0.1
        state = bf.update(noise_assess, timestamp=t)

    # 2 seconds is far below the 15-second sustained horizon
    assert not state.lockout_confirmed


def test_sustained_involuntary_evidence_confirms_lockout():
    bf = BayesianEvidenceAccumulator(
        prior_prob=0.01,
        lockout_threshold_prob=0.999,
        lockout_sustained_sec=10.0,
        warning_threshold_prob=0.85
    )

    severe_assess = ImpairmentAssessment(
        risk_score=90.0,
        involuntary_risk_score=85.0,
        marker_scores={"gen": 90.0, "lack_of_convergence": 80.0},
        classification="HIGH_RISK_INTOXICATED"
    )

    # Feed 15 seconds of severe evidence at 10 Hz
    states = []
    for i in range(160):
        t = i * 0.1
        s = bf.update(severe_assess, timestamp=t)
        states.append(s)

    # Posterior should reach high certainty (>99.9%)
    assert states[-1].posterior_probability >= 0.999
    # Sustained time should exceed 10.0s
    assert states[-1].sustained_high_confidence_s >= 10.0
    # Lockout confirmed flag must be True
    assert states[-1].lockout_confirmed
