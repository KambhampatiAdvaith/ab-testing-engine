import pytest
import numpy as np
from src.unit3_hypothesis_testing.bayesian_estimation import BayesianABTest


def test_bayesian_update():
    """Verify posterior parameters after update."""
    bayes = BayesianABTest(prior_alpha=1.0, prior_beta=1.0)
    bayes.update('A', successes=30, failures=70)
    alpha_a, beta_a = bayes.get_posterior('A')
    assert alpha_a == 31.0
    assert beta_a == 71.0


def test_probability_b_beats_a():
    """With clearly better B, P(B>A) should be high (> 0.9)."""
    np.random.seed(42)
    bayes = BayesianABTest(prior_alpha=1.0, prior_beta=1.0)
    bayes.update('A', successes=50, failures=950)
    bayes.update('B', successes=150, failures=850)
    prob = bayes.probability_b_beats_a(n_simulations=100000)
    assert prob > 0.99, f"Expected P(B>A) > 0.99, got {prob}"


def test_bayesian_expected_loss():
    """Expected loss of the worse variant should be higher."""
    np.random.seed(42)
    bayes = BayesianABTest()
    bayes.update('A', successes=50, failures=950)
    bayes.update('B', successes=150, failures=850)
    loss_a = bayes.expected_loss('A')
    loss_b = bayes.expected_loss('B')
    assert loss_a > loss_b


def test_bayesian_prior_only():
    """With only prior, P(B>A) should be near 0.5."""
    np.random.seed(42)
    bayes = BayesianABTest()
    prob = bayes.probability_b_beats_a(n_simulations=100000)
    assert abs(prob - 0.5) < 0.05
