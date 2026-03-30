import numpy as np


def generate_conversion_data(n_control: int = 1000, n_variant: int = 1000,
                              control_rate: float = 0.10, variant_rate: float = 0.12,
                              seed: int = 42) -> tuple:
    """
    Generate synthetic A/B test conversion data using binomial sampling.

    Args:
        n_control: Number of users in the control group
        n_variant: Number of users in the variant group
        control_rate: True conversion rate for control
        variant_rate: True conversion rate for variant
        seed: Random seed for reproducibility

    Returns:
        Tuple of (control_data, variant_data) dicts, each with 'clicks' and 'total' keys
    """
    np.random.seed(seed)
    control_clicks = int(np.random.binomial(n_control, control_rate))
    variant_clicks = int(np.random.binomial(n_variant, variant_rate))
    return (
        {'clicks': control_clicks, 'total': n_control},
        {'clicks': variant_clicks, 'total': n_variant}
    )


def generate_regression_data(n: int = 200, n_features: int = 3,
                               true_coefficients: np.ndarray = None,
                               noise_std: float = 1.0, seed: int = 42) -> tuple:
    """
    Generate synthetic linear regression data.

    Args:
        n: Number of observations
        n_features: Number of predictor features
        true_coefficients: Array of shape (n_features + 1,) with [intercept, beta_1, ..., beta_p].
                           If None, random coefficients are used.
        noise_std: Standard deviation of the Gaussian noise
        seed: Random seed for reproducibility

    Returns:
        Tuple of (X, y, true_coefficients)
    """
    np.random.seed(seed)
    X = np.random.randn(n, n_features)
    if true_coefficients is None:
        true_coefficients = np.random.randn(n_features + 1)
    y = true_coefficients[0] + X @ true_coefficients[1:] + np.random.normal(0, noise_std, n)
    return X, y, true_coefficients
