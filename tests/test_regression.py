import pytest
import numpy as np
from src.unit4_regression.linear_regression import MultipleLinearRegression
from src.unit4_regression.hypothesis_tests import t_test_coefficients, f_test_overall


def test_regression_simple():
    """Fit a simple known linear relationship, verify coefficients."""
    np.random.seed(42)
    n = 200
    X = np.random.randn(n, 1)
    y = 3.0 + 2.5 * X[:, 0] + np.random.normal(0, 0.1, n)

    model = MultipleLinearRegression()
    model.fit(X, y)

    assert abs(model.coefficients_[0] - 3.0) < 0.1, f"Intercept: {model.coefficients_[0]}"
    assert abs(model.coefficients_[1] - 2.5) < 0.1, f"Slope: {model.coefficients_[1]}"


def test_r_squared():
    """Perfect fit should give R²≈1."""
    X = np.array([[1.0], [2.0], [3.0], [4.0], [5.0]])
    y = np.array([2.0, 4.0, 6.0, 8.0, 10.0])

    model = MultipleLinearRegression()
    model.fit(X, y)

    assert abs(model.r_squared() - 1.0) < 1e-6


def test_regression_multiple_features():
    """Test with multiple features."""
    np.random.seed(0)
    n = 300
    X = np.random.randn(n, 3)
    true_coefs = np.array([1.0, 2.0, -1.5, 3.0])
    y = true_coefs[0] + X @ true_coefs[1:] + np.random.normal(0, 0.5, n)

    model = MultipleLinearRegression()
    model.fit(X, y)

    for i, (est, true) in enumerate(zip(model.coefficients_, true_coefs)):
        assert abs(est - true) < 0.2, f"Coef {i}: expected {true}, got {est}"


def test_residuals():
    """Residuals should sum to approximately zero for a fitted model with intercept."""
    np.random.seed(1)
    X = np.random.randn(100, 2)
    y = 1.0 + X[:, 0] + 2 * X[:, 1] + np.random.normal(0, 0.5, 100)

    model = MultipleLinearRegression()
    model.fit(X, y)

    resid = model.residuals()
    assert len(resid) == 100
    assert abs(resid.sum()) < 1e-6


def test_t_test_coefficients():
    """t-test should identify significant coefficient."""
    np.random.seed(42)
    n = 500
    X = np.random.randn(n, 1)
    y = 5.0 + 3.0 * X[:, 0] + np.random.normal(0, 0.5, n)

    model = MultipleLinearRegression()
    model.fit(X, y)
    t_results = t_test_coefficients(model)

    assert t_results['p_values'][1] < 0.001


def test_f_test():
    """F-test on a model with significant predictors."""
    np.random.seed(42)
    n = 200
    X = np.random.randn(n, 2)
    y = 1.0 + 2.0 * X[:, 0] + 0.5 * X[:, 1] + np.random.normal(0, 0.5, n)

    model = MultipleLinearRegression()
    model.fit(X, y)
    f_result = f_test_overall(model)

    assert f_result['f_statistic'] > 1.0
    assert f_result['p_value'] < 0.05
