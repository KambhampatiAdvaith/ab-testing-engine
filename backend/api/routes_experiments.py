import uuid
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from backend.database import get_db
from backend.models import Experiment, Variant

router = APIRouter()


class VariantCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=100)
    is_control: bool = False


class ExperimentCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=255)
    description: Optional[str] = Field(None, max_length=2000)
    hypothesis: Optional[str] = Field(None, max_length=2000)
    baseline_rate: float = Field(0.0, ge=0, le=1)
    min_detectable_effect: Optional[float] = Field(None, ge=0)
    confidence_level: float = Field(0.95, gt=0, lt=1)
    variants: list[VariantCreate] = Field(default_factory=list)


def _experiment_to_dict(exp: Experiment) -> dict:
    return {
        "id": exp.id,
        "name": exp.name,
        "description": exp.description,
        "status": exp.status if isinstance(exp.status, str) else exp.status.value,
        "hypothesis": exp.hypothesis,
        "baseline_rate": exp.baseline_rate,
        "min_detectable_effect": exp.min_detectable_effect,
        "confidence_level": exp.confidence_level,
        "created_at": exp.created_at.isoformat() if exp.created_at else None,
        "updated_at": exp.updated_at.isoformat() if exp.updated_at else None,
        "variants": [
            {
                "id": v.id,
                "name": v.name,
                "is_control": v.is_control,
                "clicks": v.clicks,
                "impressions": v.impressions,
                "conversion_rate": v.conversion_rate,
            }
            for v in (exp.variants or [])
        ],
    }


@router.post("/experiments")
async def create_experiment(
    request: ExperimentCreate, db: AsyncSession = Depends(get_db)
):
    """Create a new experiment with optional variants."""
    experiment = Experiment(
        id=str(uuid.uuid4()),
        name=request.name,
        description=request.description,
        hypothesis=request.hypothesis,
        baseline_rate=request.baseline_rate,
        min_detectable_effect=request.min_detectable_effect,
        confidence_level=request.confidence_level,
    )
    db.add(experiment)
    await db.flush()

    for v in request.variants:
        variant = Variant(
            id=str(uuid.uuid4()),
            experiment_id=experiment.id,
            name=v.name,
            is_control=v.is_control,
        )
        db.add(variant)

    await db.flush()

    result = await db.execute(
        select(Experiment)
        .options(selectinload(Experiment.variants))
        .where(Experiment.id == experiment.id)
    )
    exp = result.scalar_one()
    return _experiment_to_dict(exp)


@router.get("/experiments")
async def list_experiments(db: AsyncSession = Depends(get_db)):
    """List all experiments."""
    result = await db.execute(
        select(Experiment).options(selectinload(Experiment.variants))
    )
    experiments = result.scalars().all()
    return [_experiment_to_dict(e) for e in experiments]


@router.get("/experiments/{experiment_id}")
async def get_experiment(experiment_id: str, db: AsyncSession = Depends(get_db)):
    """Get a specific experiment by ID."""
    result = await db.execute(
        select(Experiment)
        .options(selectinload(Experiment.variants))
        .where(Experiment.id == experiment_id)
    )
    exp = result.scalar_one_or_none()
    if exp is None:
        raise HTTPException(status_code=404, detail="Experiment not found")
    return _experiment_to_dict(exp)
