"""Tests for the FastAPI endpoints."""
import pytest
from fastapi.testclient import TestClient
from backend.main import app

client = TestClient(app)


def test_root():
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_frequentist_significant():
    response = client.post("/api/v1/test/frequentist", json={
        "control_clicks": 100,
        "control_total": 1000,
        "variant_clicks": 150,
        "variant_total": 1000,
        "alpha": 0.05,
    })
    assert response.status_code == 200
    data = response.json()
    assert data["is_significant"] is True
    assert data["z_statistic"] > 0
    assert data["p_value"] < 0.05
    assert isinstance(data["confidence_interval"], list)
    assert len(data["confidence_interval"]) == 2


def test_frequentist_not_significant():
    response = client.post("/api/v1/test/frequentist", json={
        "control_clicks": 100,
        "control_total": 1000,
        "variant_clicks": 102,
        "variant_total": 1000,
    })
    assert response.status_code == 200
    data = response.json()
    assert data["is_significant"] is False
    assert data["p_value"] > 0.05


def test_frequentist_validation():
    response = client.post("/api/v1/test/frequentist", json={
        "control_clicks": 100,
        "control_total": 0,
        "variant_clicks": 100,
        "variant_total": 1000,
    })
    assert response.status_code == 422


def test_sample_size():
    response = client.post("/api/v1/test/sample-size", json={
        "baseline_rate": 0.10,
        "min_detectable_effect": 0.02,
        "alpha": 0.05,
        "power": 0.8,
    })
    assert response.status_code == 200
    data = response.json()
    assert "sample_size_per_group" in data
    assert data["sample_size_per_group"] > 500


def test_bayesian():
    response = client.post("/api/v1/test/bayesian", json={
        "control_successes": 50,
        "control_failures": 950,
        "variant_successes": 150,
        "variant_failures": 850,
    })
    assert response.status_code == 200
    data = response.json()
    assert data["probability_b_beats_a"] > 0.9
    assert "posterior_a" in data
    assert "posterior_b" in data
    assert "recommendation" in data


def test_personas():
    response = client.post("/api/v1/clustering/personas", json={
        "n_users": 100,
        "k": 4,
    })
    assert response.status_code == 200
    data = response.json()
    assert len(data["users"]) == 100
    assert len(data["personas"]) == 4


def test_kmeans():
    import numpy as np
    np.random.seed(42)
    cluster1 = (np.random.randn(20, 2) + [0, 0]).tolist()
    cluster2 = (np.random.randn(20, 2) + [10, 10]).tolist()
    response = client.post("/api/v1/clustering/kmeans", json={
        "data": cluster1 + cluster2,
        "k": 2,
    })
    assert response.status_code == 200
    data = response.json()
    assert len(data["labels"]) == 40
    assert data["k"] == 2
    assert data["inertia"] > 0


def test_clt():
    response = client.post("/api/v1/clt/demonstrate", json={
        "distribution": "normal",
        "sample_sizes": [5, 30],
        "n_simulations": 100,
        "params": {"mu": 0, "sigma": 1},
    })
    assert response.status_code == 200
    data = response.json()
    assert len(data["results"]) == 2
    assert data["results"][0]["sample_size"] == 5
    assert len(data["results"][0]["sample_means"]) == 100


def test_experiments_crud():
    response = client.post("/api/v1/experiments", json={
        "name": "Test Experiment",
        "description": "Testing the API",
        "hypothesis": "Variant B converts better",
        "baseline_rate": 0.1,
    })
    assert response.status_code == 200
    exp = response.json()
    assert exp["name"] == "Test Experiment"
    exp_id = exp["id"]

    response = client.get("/api/v1/experiments")
    assert response.status_code == 200

    response = client.get(f"/api/v1/experiments/{exp_id}")
    assert response.status_code == 200
    assert response.json()["id"] == exp_id
