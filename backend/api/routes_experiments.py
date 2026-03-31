from fastapi import APIRouter
from pydantic import BaseModel, Field

router = APIRouter()


class ExperimentCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=255)
    description: str = Field("", max_length=2000)
    hypothesis: str = Field("", max_length=2000)
    baseline_rate: float = Field(0.0, ge=0, le=1)
    min_detectable_effect: float = Field(0.0, ge=0)
    confidence_level: float = Field(0.95, gt=0, lt=1)


# In-memory store (replaced by DB when PostgreSQL is connected)
_experiments: dict[int, dict] = {}
_next_id = 1


@router.post("/experiments")
def create_experiment(request: ExperimentCreate):
    """Create a new experiment (in-memory)."""
    global _next_id
    experiment = {
        "id": _next_id,
        "name": request.name,
        "description": request.description,
        "hypothesis": request.hypothesis,
        "baseline_rate": request.baseline_rate,
        "min_detectable_effect": request.min_detectable_effect,
        "confidence_level": request.confidence_level,
        "status": "draft",
        "variants": [],
    }
    _experiments[_next_id] = experiment
    _next_id += 1
    return experiment


@router.get("/experiments")
def list_experiments():
    """List all experiments."""
    return list(_experiments.values())


@router.get("/experiments/{experiment_id}")
def get_experiment(experiment_id: int):
    """Get a specific experiment by ID."""
    if experiment_id not in _experiments:
        return {"error": "Experiment not found"}
    return _experiments[experiment_id]
