import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os
from .kmeans import KMeans


def generate_user_behavior_data(n_users: int = 500) -> np.ndarray:
    """
    Generate synthetic user behavior data with 7 behavioral features.

    Four user types are generated:
    - Power Users: high engagement, high purchases, low bounce
    - Window Shoppers: high bounce rate, low session duration
    - Bots: extremely high page views, very short sessions
    - Casual Users: moderate behavior across all features

    Features:
        0: pages_per_session
        1: avg_session_duration (minutes)
        2: bounce_rate (0-1)
        3: purchase_frequency (purchases/month)
        4: support_tickets (tickets/month)
        5: login_frequency (logins/month)
        6: feature_usage_depth (0-1)

    Args:
        n_users: Total number of users to generate

    Returns:
        Data matrix of shape (n_users, 7), clipped to non-negative values
    """
    np.random.seed(42)

    n_power = n_users // 4
    power_users = np.column_stack([
        np.random.normal(12, 2, n_power),
        np.random.normal(25, 5, n_power),
        np.random.normal(0.1, 0.05, n_power),
        np.random.normal(8, 2, n_power),
        np.random.normal(1, 0.5, n_power),
        np.random.normal(20, 3, n_power),
        np.random.normal(0.9, 0.05, n_power),
    ])

    n_window = n_users // 4
    window_shoppers = np.column_stack([
        np.random.normal(2, 0.5, n_window),
        np.random.normal(1.5, 0.5, n_window),
        np.random.normal(0.8, 0.05, n_window),
        np.random.normal(0.2, 0.1, n_window),
        np.random.normal(0.1, 0.05, n_window),
        np.random.normal(2, 1, n_window),
        np.random.normal(0.1, 0.05, n_window),
    ])

    n_bots = n_users // 8
    bots = np.column_stack([
        np.random.normal(50, 5, n_bots),
        np.random.normal(0.3, 0.1, n_bots),
        np.random.normal(0.05, 0.02, n_bots),
        np.random.normal(0.0, 0.01, n_bots),
        np.random.normal(0.0, 0.01, n_bots),
        np.random.normal(50, 5, n_bots),
        np.random.normal(0.05, 0.02, n_bots),
    ])

    n_casual = n_users - n_power - n_window - n_bots
    casual_users = np.column_stack([
        np.random.normal(5, 1.5, n_casual),
        np.random.normal(8, 3, n_casual),
        np.random.normal(0.4, 0.1, n_casual),
        np.random.normal(1, 0.5, n_casual),
        np.random.normal(0.5, 0.3, n_casual),
        np.random.normal(7, 2, n_casual),
        np.random.normal(0.4, 0.1, n_casual),
    ])

    data = np.vstack([power_users, window_shoppers, bots, casual_users])
    return np.clip(data, 0, None)


def _normalize(data: np.ndarray) -> np.ndarray:
    """
    Min-max normalize each feature column to [0, 1].

    Args:
        data: Data matrix of shape (n, d)

    Returns:
        Normalized data matrix of shape (n, d)
    """
    mins = data.min(axis=0)
    maxs = data.max(axis=0)
    ranges = maxs - mins
    ranges[ranges == 0] = 1.0
    return (data - mins) / ranges


def discover_personas(data: np.ndarray, k: int = 4) -> np.ndarray:
    """
    Cluster users into personas using K-Means on normalized data.

    Args:
        data: User behavior data matrix of shape (n, 7)
        k: Number of personas to discover

    Returns:
        Cluster label array of shape (n,)
    """
    normalized = _normalize(data)
    km = KMeans(k=k, max_iterations=200)
    km.fit(normalized)
    return km.labels_


def analyze_personas(data: np.ndarray, labels: np.ndarray) -> dict:
    """
    Compute summary statistics for each cluster and auto-assign persona labels.

    Persona labeling heuristics (based on mean feature values):
    - Power User: high purchases (>5) and long sessions (>15 min)
    - Window Shopper: high bounce (>0.6) and short sessions (<3 min)
    - Bot: very high pages (>20) and very short sessions (<1 min)
    - Casual User: everything else

    Args:
        data: User behavior data matrix of shape (n, 7)
        labels: Cluster labels of shape (n,)

    Returns:
        Dict mapping cluster_id -> {'label', 'size', 'means'}
    """
    feature_names = [
        'pages_per_session', 'avg_session_duration', 'bounce_rate',
        'purchase_frequency', 'support_tickets', 'login_frequency', 'feature_usage_depth'
    ]

    personas = {}

    for cluster_id in np.unique(labels):
        mask = labels == cluster_id
        cluster_data = data[mask]
        means = cluster_data.mean(axis=0)

        pages = means[0]
        session_dur = means[1]
        bounce = means[2]
        purchases = means[3]
        logins = means[5]

        if purchases > 5 and session_dur > 15:
            label = "Power User"
        elif bounce > 0.6 and session_dur < 3:
            label = "Window Shopper"
        elif pages > 20 and session_dur < 1:
            label = "Bot"
        elif logins > 15 and session_dur > 20:
            label = "Power User"
        else:
            label = "Casual User"

        personas[cluster_id] = {
            'label': label,
            'size': int(mask.sum()),
            'means': dict(zip(feature_names, means.tolist()))
        }

    return personas


def _pca_2d(data: np.ndarray) -> np.ndarray:
    """
    Project data to 2D using PCA (manual eigendecomposition).

    Args:
        data: Data matrix of shape (n, d)

    Returns:
        2D projection of shape (n, 2)
    """
    centered = data - data.mean(axis=0)
    cov = centered.T @ centered / (len(data) - 1)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    idx = np.argsort(eigenvalues)[::-1]
    top2 = eigenvectors[:, idx[:2]]
    return centered @ top2


def visualize_personas(data: np.ndarray, labels: np.ndarray,
                       save_path: str = 'outputs/personas.png') -> None:
    """
    Visualize user personas using PCA scatter plot and radar chart.

    Args:
        data: User behavior data matrix of shape (n, 7)
        labels: Cluster labels of shape (n,)
        save_path: Path to save the visualization
    """
    dirpath = os.path.dirname(save_path)
    if dirpath:
        os.makedirs(dirpath, exist_ok=True)
    else:
        os.makedirs('outputs', exist_ok=True)

    personas = analyze_personas(data, labels)
    projected = _pca_2d(data)

    unique_labels = np.unique(labels)
    colors = plt.cm.Set1(np.linspace(0, 0.8, len(unique_labels)))

    fig = plt.figure(figsize=(16, 7))

    # PCA scatter plot
    ax1 = fig.add_subplot(121)
    for cluster_id, color in zip(unique_labels, colors):
        mask = labels == cluster_id
        persona_label = personas[cluster_id]['label']
        ax1.scatter(projected[mask, 0], projected[mask, 1],
                    c=[color], label=f'Cluster {cluster_id}: {persona_label}',
                    alpha=0.6, s=30)
    ax1.set_xlabel('PC1')
    ax1.set_ylabel('PC2')
    ax1.set_title('User Personas (PCA Projection)')
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)

    # Radar chart
    feature_names = ['pages/session', 'session_dur', 'bounce_rate',
                     'purchases', 'support', 'logins', 'feature_depth']
    n_features = len(feature_names)
    angles = np.linspace(0, 2 * np.pi, n_features, endpoint=False).tolist()
    angles += angles[:1]

    ax2 = fig.add_subplot(122, polar=True)

    all_means = np.array([list(personas[cid]['means'].values()) for cid in unique_labels])
    col_maxs = all_means.max(axis=0)
    col_maxs[col_maxs == 0] = 1.0

    for cluster_id, color in zip(unique_labels, colors):
        means = np.array(list(personas[cluster_id]['means'].values()))
        norm_means = means / col_maxs
        values = norm_means.tolist()
        values += values[:1]
        persona_label = personas[cluster_id]['label']
        ax2.plot(angles, values, 'o-', color=color, linewidth=2,
                 label=f'Cluster {cluster_id}: {persona_label}')
        ax2.fill(angles, values, color=color, alpha=0.1)

    ax2.set_xticks(angles[:-1])
    ax2.set_xticklabels(feature_names, fontsize=8)
    ax2.set_title('Feature Profiles by Persona', pad=20)
    ax2.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=8)

    plt.suptitle('User Persona Analysis', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=100, bbox_inches='tight')
    plt.close()
    print(f"Personas visualization saved to {save_path}")
