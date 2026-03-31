import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os


class KMeans:
    """
    K-Means clustering algorithm with K-Means++ initialization.

    All computations are implemented from scratch using NumPy.

    Algorithm:
    1. Initialize centroids using K-Means++ (avoids poor initialization)
    2. Assign each point to the nearest centroid (E-step)
    3. Update centroids to the mean of assigned points (M-step)
    4. Repeat until convergence (centroid shift < tolerance)

    Attributes:
        k: Number of clusters
        max_iterations: Maximum number of iterations
        tolerance: Convergence threshold for centroid shift
        centroids_: Learned centroids of shape (k, n_features)
        labels_: Cluster assignment for each training point
    """

    def __init__(self, k: int, max_iterations: int = 100, tolerance: float = 1e-4):
        """
        Initialize KMeans clustering.

        Args:
            k: Number of clusters
            max_iterations: Maximum number of E-M iterations
            tolerance: Stop when max centroid shift < tolerance
        """
        self.k = k
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.centroids_ = None
        self.labels_ = None
        self._X_fit = None

    def _init_plusplus(self, X: np.ndarray) -> np.ndarray:
        """
        K-Means++ centroid initialization.

        Select first centroid uniformly at random, then select subsequent
        centroids with probability proportional to squared distance from
        the nearest existing centroid.

        Args:
            X: Data matrix of shape (n, d)

        Returns:
            Initial centroids of shape (k, d)
        """
        n = X.shape[0]
        idx = np.random.randint(0, n)
        centroids = [X[idx].copy()]

        for _ in range(self.k - 1):
            dists = np.array([
                min(np.sum((x - c) ** 2) for c in centroids)
                for x in X
            ])
            probs = dists / dists.sum()
            cumprobs = np.cumsum(probs)
            r = np.random.uniform(0, 1)
            idx = int(np.searchsorted(cumprobs, r))
            idx = min(idx, n - 1)
            centroids.append(X[idx].copy())

        return np.array(centroids)

    def _assign_labels(self, X: np.ndarray) -> np.ndarray:
        """
        Assign each data point to its nearest centroid.

        Args:
            X: Data matrix of shape (n, d)

        Returns:
            Label array of shape (n,) with cluster indices
        """
        # Vectorized computation of squared distances
        # dists[i, j] = ||X[i] - centroids[j]||^2
        dists = np.sum((X[:, np.newaxis, :] - self.centroids_[np.newaxis, :, :]) ** 2, axis=2)
        return np.argmin(dists, axis=1)

    def fit(self, X: np.ndarray) -> None:
        """
        Fit K-Means to training data.

        Args:
            X: Data matrix of shape (n, d)
        """
        self.centroids_ = self._init_plusplus(X)
        self._X_fit = X

        for _ in range(self.max_iterations):
            labels = self._assign_labels(X)
            new_centroids = np.array([
                X[labels == k].mean(axis=0) if np.any(labels == k) else self.centroids_[k]
                for k in range(self.k)
            ])

            shift = float(np.max(np.linalg.norm(new_centroids - self.centroids_, axis=1)))
            self.centroids_ = new_centroids
            self.labels_ = labels

            if shift < self.tolerance:
                break

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Assign new data points to nearest centroid.

        Args:
            X: Data matrix of shape (n, d)

        Returns:
            Label array of shape (n,)
        """
        return self._assign_labels(X)

    def inertia(self) -> float:
        """
        Compute Within-Cluster Sum of Squares (WCSS / inertia).

        WCSS = sum_k sum_{x in cluster_k} ||x - centroid_k||^2

        Returns:
            Inertia value
        """
        if self.labels_ is None or self._X_fit is None:
            raise ValueError("Model not fitted yet. Call fit() first.")
        total = 0.0
        for k in range(self.k):
            mask = self.labels_ == k
            if np.any(mask):
                diffs = self._X_fit[mask] - self.centroids_[k]
                total += float(np.sum(diffs ** 2))
        return total

    def silhouette_score(self, X: np.ndarray) -> float:
        """
        Compute the mean Silhouette coefficient.

        For each point i:
            a(i) = mean intra-cluster distance
            b(i) = mean distance to nearest other cluster
            s(i) = (b(i) - a(i)) / max(a(i), b(i))

        Args:
            X: Data matrix of shape (n, d)

        Returns:
            Mean silhouette score in [-1, 1]
        """
        labels = self.predict(X)
        n = X.shape[0]
        scores = []

        for i in range(n):
            same_mask = labels == labels[i]
            same_mask[i] = False

            if same_mask.sum() == 0:
                scores.append(0.0)
                continue

            a = float(np.mean(np.sqrt(np.sum((X[same_mask] - X[i]) ** 2, axis=1))))

            b_vals = []
            for k in range(self.k):
                if k == labels[i]:
                    continue
                other_mask = labels == k
                if other_mask.sum() > 0:
                    b_k = float(np.mean(np.sqrt(np.sum((X[other_mask] - X[i]) ** 2, axis=1))))
                    b_vals.append(b_k)

            if not b_vals:
                scores.append(0.0)
                continue

            b = min(b_vals)
            denom = max(a, b)
            s = (b - a) / denom if denom > 0 else 0.0
            scores.append(s)

        return float(np.mean(scores))

    def elbow_method(self, X: np.ndarray, k_range: range,
                     save_path: str = 'outputs/elbow.png') -> list:
        """
        Run K-Means for multiple k values and plot the elbow curve.

        The elbow in the WCSS vs k curve suggests the optimal number of clusters.

        Args:
            X: Data matrix of shape (n, d)
            k_range: Range of k values to evaluate
            save_path: Path to save the elbow plot

        Returns:
            List of inertia values for each k
        """
        dirpath = os.path.dirname(save_path)
        if dirpath:
            os.makedirs(dirpath, exist_ok=True)
        else:
            os.makedirs('outputs', exist_ok=True)

        inertias = []
        for k in k_range:
            km = KMeans(k=k, max_iterations=self.max_iterations, tolerance=self.tolerance)
            km.fit(X)
            inertias.append(km.inertia())

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(list(k_range), inertias, 'bo-', markersize=8)
        ax.set_xlabel('Number of Clusters (k)')
        ax.set_ylabel('Inertia (WCSS)')
        ax.set_title('Elbow Method for Optimal k')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
        plt.close()
        print(f"Elbow plot saved to {save_path}")

        return inertias
