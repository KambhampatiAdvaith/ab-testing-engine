import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os


def generate_ltv_dataset(n_users: int = 1000) -> tuple:
    """
    Generate a synthetic Customer Lifetime Value (LTV) dataset.

    Features (one-hot encoded categoricals + continuous):
    - region_na, region_eu: geographic region dummies (APAC is reference)
    - referral_paid, referral_social: referral source dummies (organic is reference)
    - device_mobile, device_tablet: device type dummies (desktop is reference)
    - avg_session_duration: continuous, minutes
    - pages_per_visit: continuous
    - days_since_signup: continuous, days

    True LTV = 50 + 20*region_na + 15*region_eu + 25*referral_paid
               + 10*referral_social - 30*device_mobile - 10*device_tablet
               + 8*avg_session_duration + 12*pages_per_visit
               + 0.05*days_since_signup + noise

    Args:
        n_users: Number of users to generate

    Returns:
        Tuple of (X, y, feature_names)
    """
    np.random.seed(42)

    region = np.random.choice(3, n_users)
    region_na = (region == 0).astype(float)
    region_eu = (region == 1).astype(float)

    referral = np.random.choice(3, n_users)
    referral_paid = (referral == 1).astype(float)
    referral_social = (referral == 2).astype(float)

    device = np.random.choice(3, n_users, p=[0.5, 0.4, 0.1])
    device_mobile = (device == 1).astype(float)
    device_tablet = (device == 2).astype(float)

    avg_session_duration = np.abs(np.random.normal(8, 3, n_users))
    pages_per_visit = np.abs(np.random.normal(5, 2, n_users))
    days_since_signup = np.random.uniform(1, 365, n_users)

    ltv = (
        50
        + 20 * region_na
        + 15 * region_eu
        + 25 * referral_paid
        + 10 * referral_social
        - 30 * device_mobile
        - 10 * device_tablet
        + 8 * avg_session_duration
        + 12 * pages_per_visit
        + 0.05 * days_since_signup
        + np.random.normal(0, 20, n_users)
    )

    X = np.column_stack([
        region_na, region_eu, referral_paid, referral_social,
        device_mobile, device_tablet, avg_session_duration,
        pages_per_visit, days_since_signup
    ])
    y = ltv

    feature_names = [
        'region_na', 'region_eu', 'referral_paid', 'referral_social',
        'device_mobile', 'device_tablet', 'avg_session_duration',
        'pages_per_visit', 'days_since_signup'
    ]

    return X, y, feature_names


def predict_lifetime_value(model, user_features: np.ndarray) -> np.ndarray:
    """
    Predict LTV for new users using a fitted regression model.

    Args:
        model: Fitted MultipleLinearRegression instance
        user_features: Feature matrix of shape (m, n_features)

    Returns:
        Predicted LTV values of shape (m,)
    """
    return model.predict(user_features)


def analyze_feature_importance(model, feature_names: list) -> None:
    """
    Plot a horizontal bar chart of feature importance by coefficient magnitude.

    Coefficients are sorted by absolute value and colored by sign.

    Args:
        model: Fitted MultipleLinearRegression instance
        feature_names: List of feature names (excluding intercept)
    """
    os.makedirs('outputs', exist_ok=True)

    coefs = model.coefficients_[1:]  # skip intercept

    sorted_idx = np.argsort(np.abs(coefs))
    sorted_coefs = coefs[sorted_idx]
    sorted_names = [feature_names[i] for i in sorted_idx]

    colors = ['coral' if c < 0 else 'steelblue' for c in sorted_coefs]

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(sorted_names, sorted_coefs, color=colors, alpha=0.8)
    ax.axvline(x=0, color='black', linewidth=0.8, linestyle='-')
    ax.set_xlabel('Coefficient Value')
    ax.set_title('Feature Importance for LTV Prediction\n(Coefficient Magnitude)')
    ax.grid(axis='x', alpha=0.3)

    for bar, coef in zip(bars, sorted_coefs):
        ax.text(
            coef + (0.5 if coef >= 0 else -0.5),
            bar.get_y() + bar.get_height() / 2,
            f'{coef:.2f}', va='center',
            ha='left' if coef >= 0 else 'right', fontsize=9
        )

    plt.tight_layout()
    plt.savefig('outputs/feature_importance.png', dpi=100, bbox_inches='tight')
    plt.close()
    print("Feature importance plot saved to outputs/feature_importance.png")
