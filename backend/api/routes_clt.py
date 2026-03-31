from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from backend.src.unit1_2_random_variables.distributions import (
    BinomialDistribution,
    NormalDistribution,
    UniformDistribution,
)

router = APIRouter()


class CLTRequest(BaseModel):
    distribution: str = Field(
        "normal",
        description="Population distribution type: 'normal', 'uniform', or 'binomial'",
    )
    sample_sizes: list[int] = Field(
        [5, 30, 100],
        description="List of sample sizes to demonstrate CLT",
    )
    n_simulations: int = Field(1000, ge=100, le=10000)
    params: dict = Field(
        default_factory=dict,
        description="Distribution parameters (e.g., {'mu': 0, 'sigma': 1})",
    )


@router.post("/clt/demonstrate")
def demonstrate_clt(request: CLTRequest):
    """Demonstrate the Central Limit Theorem by sampling from a distribution."""
    try:
        import numpy as np

        if request.distribution == "normal":
            mu = request.params.get("mu", 0)
            sigma = request.params.get("sigma", 1)
            dist = NormalDistribution(mu=mu, sigma=sigma)
        elif request.distribution == "uniform":
            a = request.params.get("a", 0)
            b = request.params.get("b", 1)
            dist = UniformDistribution(a=a, b=b)
        elif request.distribution == "binomial":
            n = request.params.get("n", 10)
            p = request.params.get("p", 0.5)
            dist = BinomialDistribution(n=n, p=p)
        else:
            raise ValueError(
                f"Unknown distribution: {request.distribution}. "
                "Use 'normal', 'uniform', or 'binomial'."
            )

        pop_mean = dist.mean()
        pop_variance = dist.variance()
        results = []

        for sample_size in request.sample_sizes:
            sample_means = []
            for _ in range(request.n_simulations):
                sample = dist.sample(sample_size)
                sample_means.append(float(np.mean(sample)))

            theoretical_std = float(np.sqrt(pop_variance / sample_size))
            results.append({
                "sample_size": sample_size,
                "sample_means": sample_means,
                "empirical_mean": float(np.mean(sample_means)),
                "empirical_std": float(np.std(sample_means)),
                "theoretical_mean": float(pop_mean),
                "theoretical_std": theoretical_std,
            })

        return {
            "distribution": request.distribution,
            "population_mean": float(pop_mean),
            "population_variance": float(pop_variance),
            "results": results,
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
