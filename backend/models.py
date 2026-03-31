from datetime import datetime
from sqlalchemy import (
    Column, Integer, String, Float, DateTime, Boolean, ForeignKey, Text, JSON
)
from sqlalchemy.orm import relationship
from backend.database import Base


class Experiment(Base):
    """Stores A/B test experiment metadata."""
    __tablename__ = "experiments"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(255), nullable=False)
    description = Column(Text, nullable=True)
    status = Column(String(50), default="draft")
    hypothesis = Column(Text, nullable=True)
    baseline_rate = Column(Float, nullable=True)
    min_detectable_effect = Column(Float, nullable=True)
    confidence_level = Column(Float, default=0.95)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    variants = relationship("Variant", back_populates="experiment", cascade="all, delete-orphan")


class Variant(Base):
    """Stores variant data (Version A vs Version B) for an experiment."""
    __tablename__ = "variants"

    id = Column(Integer, primary_key=True, index=True)
    experiment_id = Column(Integer, ForeignKey("experiments.id"), nullable=False)
    name = Column(String(100), nullable=False)
    is_control = Column(Boolean, default=False)
    clicks = Column(Integer, default=0)
    total = Column(Integer, default=0)
    conversion_rate = Column(Float, default=0.0)

    experiment = relationship("Experiment", back_populates="variants")
    events = relationship("UserEvent", back_populates="variant", cascade="all, delete-orphan")


class UserEvent(Base):
    """Stores individual user interaction events (clicks, time-on-page)."""
    __tablename__ = "user_events"

    id = Column(Integer, primary_key=True, index=True)
    variant_id = Column(Integer, ForeignKey("variants.id"), nullable=False)
    user_identifier = Column(String(255), nullable=True)
    event_type = Column(String(50), nullable=False)
    value = Column(Float, nullable=True)
    metadata_ = Column("metadata", JSON, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    variant = relationship("Variant", back_populates="events")


class UserPersona(Base):
    """Stores K-Means clustering results for user segmentation."""
    __tablename__ = "user_personas"

    id = Column(Integer, primary_key=True, index=True)
    cluster_id = Column(Integer, nullable=False)
    label = Column(String(100), nullable=True)
    size = Column(Integer, default=0)
    feature_means = Column(JSON, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    run_parameters = Column(JSON, nullable=True)
