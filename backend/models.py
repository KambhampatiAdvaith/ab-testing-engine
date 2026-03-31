import uuid
from datetime import datetime, timezone
from enum import Enum as PyEnum
from typing import Optional

from sqlalchemy import (
    Boolean, DateTime, Enum, Float, ForeignKey, Index, Integer, JSON,
    String, Text,
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from backend.database import Base


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _uuid() -> str:
    return str(uuid.uuid4())


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class ExperimentStatus(str, PyEnum):
    draft = "draft"
    running = "running"
    paused = "paused"
    completed = "completed"


class EventType(str, PyEnum):
    click = "click"
    page_view = "page_view"
    conversion = "conversion"
    bounce = "bounce"


# ---------------------------------------------------------------------------
# ORM Models
# ---------------------------------------------------------------------------

class Experiment(Base):
    """Stores A/B test experiment metadata."""
    __tablename__ = "experiments"

    id: Mapped[str] = mapped_column(
        UUID(as_uuid=False), primary_key=True, default=_uuid
    )
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    status: Mapped[str] = mapped_column(
        Enum(ExperimentStatus), default=ExperimentStatus.draft, nullable=False
    )
    hypothesis: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    baseline_rate: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    min_detectable_effect: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    confidence_level: Mapped[float] = mapped_column(Float, default=0.95)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_now
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_now, onupdate=_now
    )

    variants: Mapped[list["Variant"]] = relationship(
        "Variant", back_populates="experiment", cascade="all, delete-orphan"
    )


class Variant(Base):
    """Stores variant data (Version A vs Version B) for an experiment."""
    __tablename__ = "variants"
    __table_args__ = (Index("ix_variants_experiment_id", "experiment_id"),)

    id: Mapped[str] = mapped_column(
        UUID(as_uuid=False), primary_key=True, default=_uuid
    )
    experiment_id: Mapped[str] = mapped_column(
        UUID(as_uuid=False), ForeignKey("experiments.id"), nullable=False
    )
    name: Mapped[str] = mapped_column(String(100), nullable=False)
    is_control: Mapped[bool] = mapped_column(Boolean, default=False)
    clicks: Mapped[int] = mapped_column(Integer, default=0)
    impressions: Mapped[int] = mapped_column(Integer, default=0)
    conversion_rate: Mapped[float] = mapped_column(Float, default=0.0)

    experiment: Mapped["Experiment"] = relationship(
        "Experiment", back_populates="variants"
    )
    events: Mapped[list["UserEvent"]] = relationship(
        "UserEvent", back_populates="variant", cascade="all, delete-orphan"
    )


class UserEvent(Base):
    """Stores individual user interaction events."""
    __tablename__ = "user_events"
    __table_args__ = (Index("ix_user_events_variant_id", "variant_id"),)

    id: Mapped[str] = mapped_column(
        UUID(as_uuid=False), primary_key=True, default=_uuid
    )
    variant_id: Mapped[str] = mapped_column(
        UUID(as_uuid=False), ForeignKey("variants.id"), nullable=False
    )
    user_identifier: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    event_type: Mapped[str] = mapped_column(
        Enum(EventType), nullable=False
    )
    value: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    metadata_: Mapped[Optional[dict]] = mapped_column(
        "metadata", JSON, nullable=True
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_now
    )

    variant: Mapped["Variant"] = relationship("Variant", back_populates="events")


class ClusterPersona(Base):
    """Stores K-Means clustering results for user segmentation."""
    __tablename__ = "cluster_personas"

    id: Mapped[str] = mapped_column(
        UUID(as_uuid=False), primary_key=True, default=_uuid
    )
    run_id: Mapped[str] = mapped_column(String(100), nullable=False)
    cluster_id: Mapped[int] = mapped_column(Integer, nullable=False)
    label: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    size: Mapped[int] = mapped_column(Integer, default=0)
    centroid: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    feature_means: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    silhouette_score: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_now
    )
    run_parameters: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
