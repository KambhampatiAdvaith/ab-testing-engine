import numpy as np
from .distributions import BinomialDistribution, NormalDistribution, UniformDistribution


def simulate_clicks(n_users: int, click_probability: float) -> np.ndarray:
    """
    Simulate user click events using a Bernoulli distribution (Binomial with n=1).

    Args:
        n_users: Number of users to simulate
        click_probability: Probability that each user clicks

    Returns:
        Array of 0/1 values indicating click (1) or no click (0)
    """
    dist = BinomialDistribution(n=1, p=click_probability)
    return dist.sample(n_users)


def simulate_session_times(n_users: int, distribution: str = 'normal',
                            params: dict = None) -> np.ndarray:
    """
    Simulate user session durations using a specified distribution.

    Args:
        n_users: Number of users to simulate
        distribution: 'normal' or 'uniform'
        params: Distribution parameters dict.
                For 'normal': {'mu': float, 'sigma': float}
                For 'uniform': {'a': float, 'b': float}

    Returns:
        Array of positive session durations in minutes
    """
    if params is None:
        params = {}
    if distribution == 'normal':
        mu = params.get('mu', 5.0)
        sigma = params.get('sigma', 1.5)
        dist = NormalDistribution(mu=mu, sigma=sigma)
        times = dist.sample(n_users)
        return np.abs(times)  # session times must be positive
    elif distribution == 'uniform':
        a = params.get('a', 1.0)
        b = params.get('b', 10.0)
        dist = UniformDistribution(a=a, b=b)
        return dist.sample(n_users)
    else:
        raise ValueError(f"Unknown distribution: {distribution}. Use 'normal' or 'uniform'.")


def simulate_traffic(n_users: int, n_days: int) -> dict:
    """
    Simulate website traffic data over multiple days.

    Each day generates n_users with clicks, session times, and page views.

    Args:
        n_users: Number of users per day
        n_days: Number of days to simulate

    Returns:
        Dictionary with keys: user_id, clicks, session_time, page_views, timestamp
        Each value is a numpy array.
    """
    all_data: dict = {
        'user_id': [],
        'clicks': [],
        'session_time': [],
        'page_views': [],
        'timestamp': []
    }
    for day in range(n_days):
        user_ids = np.arange(day * n_users, (day + 1) * n_users)
        clicks = simulate_clicks(n_users, click_probability=0.15)
        session_times = simulate_session_times(
            n_users, distribution='normal', params={'mu': 5.0, 'sigma': 2.0}
        )
        page_views_dist = BinomialDistribution(n=20, p=0.3)
        page_views = page_views_dist.sample(n_users)
        all_data['user_id'].extend(user_ids.tolist())
        all_data['clicks'].extend(clicks.tolist())
        all_data['session_time'].extend(session_times.tolist())
        all_data['page_views'].extend(page_views.tolist())
        all_data['timestamp'].extend([day] * n_users)
    for key in all_data:
        all_data[key] = np.array(all_data[key])
    return all_data
