import pytest
import numpy as np
from backend.src.unit5_clustering.kmeans import KMeans


def test_kmeans_convergence():
    """Test on clearly separable clusters."""
    np.random.seed(42)
    cluster1 = np.random.randn(50, 2) + np.array([0, 0])
    cluster2 = np.random.randn(50, 2) + np.array([10, 10])
    cluster3 = np.random.randn(50, 2) + np.array([20, 0])
    X = np.vstack([cluster1, cluster2, cluster3])

    km = KMeans(k=3, max_iterations=100)
    km.fit(X)

    assert km.labels_ is not None
    assert len(km.labels_) == 150
    assert len(np.unique(km.labels_)) == 3
    assert km.inertia() < 5000


def test_kmeans_predict():
    """Verify predictions match training labels."""
    np.random.seed(0)
    cluster1 = np.random.randn(40, 2) + np.array([0, 0])
    cluster2 = np.random.randn(40, 2) + np.array([10, 10])
    X = np.vstack([cluster1, cluster2])

    km = KMeans(k=2, max_iterations=100)
    km.fit(X)

    train_labels = km.labels_
    pred_labels = km.predict(X)

    assert np.array_equal(train_labels, pred_labels)


def test_kmeans_inertia():
    """Inertia should decrease as k increases."""
    np.random.seed(42)
    X = np.random.randn(100, 2)

    inertias = []
    for k in [2, 3, 4]:
        km = KMeans(k=k, max_iterations=50)
        km.fit(X)
        inertias.append(km.inertia())

    assert inertias[0] >= inertias[-1]


def test_kmeans_centroids_shape():
    """Centroids should have shape (k, n_features)."""
    np.random.seed(1)
    X = np.random.randn(100, 5)

    km = KMeans(k=4)
    km.fit(X)

    assert km.centroids_.shape == (4, 5)
