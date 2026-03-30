import numpy as np
from math import lgamma


def _normal_cdf(x: float) -> float:
    """Standard normal CDF via Abramowitz & Stegun error function approximation."""
    def _erf(z: float) -> float:
        if z == 0:
            return 0.0
        t = 1.0 / (1.0 + 0.3275911 * abs(z))
        poly = t * (0.254829592 + t * (-0.284496736 + t * (1.421413741 + t * (-1.453152027 + t * 1.061405429))))
        result = 1.0 - poly * np.exp(-z * z)
        return float(np.sign(z)) * result
    return 0.5 * (1.0 + _erf(x / np.sqrt(2)))


def _t_cdf_approx(t: float, df: int) -> float:
    """
    Approximate CDF of the t-distribution.

    For df > 30, uses normal approximation (adequate for large samples).
    For smaller df, uses regularized incomplete beta function approximation
    via numerical integration.

    Args:
        t: t-statistic
        df: Degrees of freedom

    Returns:
        P(T <= t) for T ~ t(df)
    """
    if df > 30:
        return _normal_cdf(t)

    x = df / (df + t ** 2)
    a, b = df / 2.0, 0.5

    n_steps = 1000
    ts = np.linspace(1e-10, x - 1e-10, n_steps)
    log_integrand = (a - 1) * np.log(ts) + (b - 1) * np.log(1 - ts)
    log_beta_val = lgamma(a) + lgamma(b) - lgamma(a + b)
    integrand = np.exp(log_integrand - log_beta_val)
    ibeta = float(np.trapz(integrand, ts))
    ibeta = float(np.clip(ibeta, 0, 1))

    if t >= 0:
        return 1.0 - 0.5 * ibeta
    else:
        return 0.5 * ibeta


def t_test_coefficients(model) -> dict:
    """
    Compute t-statistics and p-values for each regression coefficient.

    SE(beta_j) = sqrt(MSE * [(X^T X)^{-1}]_{jj})
    t_j = beta_j / SE(beta_j)
    p_j = 2 * P(T_{df} > |t_j|)

    where MSE = SS_res / (n - p - 1)

    Args:
        model: Fitted MultipleLinearRegression instance

    Returns:
        Dict with keys: std_errors, t_statistics, p_values, df, mse
    """
    residuals = model.residuals()
    n = model.n_
    p = model.p_
    df = n - p - 1

    mse = float(np.sum(residuals ** 2) / df)

    XtX = model.X_with_bias_.T @ model.X_with_bias_
    XtX_inv = np.linalg.inv(XtX)

    var_coefs = np.diag(XtX_inv) * mse
    std_errors = np.sqrt(np.maximum(var_coefs, 0))

    t_stats = model.coefficients_ / (std_errors + 1e-300)

    p_values = np.array([
        2.0 * (1.0 - _t_cdf_approx(abs(float(t)), df))
        for t in t_stats
    ])

    return {
        'std_errors': std_errors,
        't_statistics': t_stats,
        'p_values': p_values,
        'df': df,
        'mse': mse
    }


def f_test_overall(model) -> dict:
    """
    Compute the F-statistic for overall model significance.

    F = (SS_reg / p) / (SS_res / (n - p - 1))
    Under H0, F ~ F(p, n-p-1)

    Args:
        model: Fitted MultipleLinearRegression instance

    Returns:
        Dict with f_statistic, p_value, df_regression, df_residual,
        ms_regression, ms_residual
    """
    y_pred = model.X_with_bias_ @ model.coefficients_
    y_mean = float(np.mean(model.y_))

    ss_reg = float(np.sum((y_pred - y_mean) ** 2))
    ss_res = float(np.sum((model.y_ - y_pred) ** 2))

    p = model.p_
    n = model.n_
    df_reg = p
    df_res = n - p - 1

    if df_res <= 0:
        return {
            'f_statistic': float('nan'), 'p_value': float('nan'),
            'df_regression': df_reg, 'df_residual': df_res
        }

    ms_reg = ss_reg / df_reg
    ms_res = ss_res / df_res

    if ms_res == 0:
        f_stat = float('inf')
        p_value = 0.0
    else:
        f_stat = ms_reg / ms_res
        # Approximate p-value using chi-squared normal approximation
        chi2_stat = df_reg * f_stat
        z = np.sqrt(2 * chi2_stat) - np.sqrt(2 * df_reg - 1)
        p_value = float(np.clip(1.0 - _normal_cdf(float(z)), 0, 1))

    return {
        'f_statistic': float(f_stat),
        'p_value': p_value,
        'df_regression': df_reg,
        'df_residual': df_res,
        'ms_regression': float(ms_reg),
        'ms_residual': float(ms_res)
    }
