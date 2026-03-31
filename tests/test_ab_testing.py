import pytest
import numpy as np
from backend.src.unit3_hypothesis_testing.ab_test_engine import ABTestEngine


def test_ztest_significant():
    """Test with clearly different rates should be significant."""
    engine = ABTestEngine()
    result = engine.run_ztest(
        control_clicks=100, control_total=1000,
        variant_clicks=150, variant_total=1000,
        alpha=0.05
    )
    assert result['is_significant'] == True
    assert result['z_statistic'] > 0  # variant > control, so z > 0
    assert result['p_value'] < 0.05


def test_ztest_not_significant():
    """Test with similar rates should not be significant."""
    engine = ABTestEngine()
    result = engine.run_ztest(
        control_clicks=100, control_total=1000,
        variant_clicks=102, variant_total=1000,
        alpha=0.05
    )
    assert result['is_significant'] == False
    assert result['p_value'] > 0.05


def test_sample_size_calculation():
    """Verify sample size formula gives reasonable results."""
    engine = ABTestEngine()
    n = engine.calculate_sample_size(
        baseline_rate=0.10,
        min_detectable_effect=0.02,
        alpha=0.05,
        power=0.8
    )
    assert n > 500
    assert n < 50000
    assert isinstance(n, int)


def test_ztest_returns_all_keys():
    """Verify result dict has all expected keys."""
    engine = ABTestEngine()
    result = engine.run_ztest(50, 500, 60, 500)
    expected_keys = {'z_statistic', 'p_value', 'confidence_interval', 'is_significant', 'conclusion'}
    assert expected_keys.issubset(set(result.keys()))


def test_ztest_confidence_interval():
    """CI should contain the observed difference."""
    engine = ABTestEngine()
    result = engine.run_ztest(100, 1000, 120, 1000)
    ci_low, ci_high = result['confidence_interval']
    assert ci_low < ci_high
    diff = result['variant_rate'] - result['control_rate']
    assert ci_low <= diff <= ci_high
