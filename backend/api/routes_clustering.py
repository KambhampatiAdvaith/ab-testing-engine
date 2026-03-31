from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from backend.src.unit5_clustering.kmeans import KMeans
from backend.src.unit5_clustering.user_personas import (
    generate_user_behavior_data,
    discover_personas,
    analyze_personas,
)

router = APIRouter()

FEATURE_NAMES = [
    "pages_per_session",
    "avg_session_duration",
    "bounce_rate",
    "purchase_frequency",
    "support_tickets",
    "login_frequency",
    "feature_usage_depth",
]


class PersonaRequest(BaseModel):
    n_users: int = Field(500, ge=10, le=10000, description="Number of synthetic users")
    k: int = Field(4, ge=2, le=10, description="Number of clusters")


class KMeansRequest(BaseModel):
    data: list[list[float]] = Field(..., description="2D array of feature data")
    k: int = Field(3, ge=2, le=20, description="Number of clusters")
    max_iterations: int = Field(100, ge=1, le=1000)


@router.post("/clustering/personas")
def discover_user_personas(request: PersonaRequest):
    """Generate synthetic user data, cluster it, and return persona analysis."""
    try:
        data = generate_user_behavior_data(n_users=request.n_users)
        labels = discover_personas(data, k=request.k)
        analysis = analyze_personas(data, labels)

        users = []
        for i in range(len(data)):
            user = {FEATURE_NAMES[j]: float(data[i, j]) for j in range(data.shape[1])}
            user["cluster"] = int(labels[i])
            users.append(user)

        personas = {}
        for cluster_id, info in analysis.items():
            personas[str(cluster_id)] = {
                "label": info["label"],
                "size": info["size"],
                "means": info["means"],
            }

        return {
            "users": users,
            "personas": personas,
            "n_users": request.n_users,
            "k": request.k,
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/clustering/kmeans")
def run_kmeans(request: KMeansRequest):
    """Run K-Means clustering on provided data."""
    try:
        import numpy as np

        data = np.array(request.data)
        if data.ndim != 2:
            raise ValueError("Data must be a 2D array")

        km = KMeans(k=request.k, max_iterations=request.max_iterations)
        km.fit(data)

        labels = km.predict(data)
        inertia = km.inertia()
        silhouette = km.silhouette_score(data) if len(data) > request.k else None
        centroids = km.centroids_.tolist()

        return {
            "labels": labels.tolist(),
            "centroids": centroids,
            "inertia": float(inertia),
            "silhouette_score": float(silhouette) if silhouette is not None else None,
            "k": request.k,
            "n_samples": len(data),
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
