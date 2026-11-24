from __future__ import annotations

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str
    service: str
    device: str


class ModelStatus(BaseModel):
    key: str
    display_name: str
    technical_name: str
    checkpoint: str
    checkpoint_available: bool
    loaded: bool
    runtime_available: bool
    device: str | None = None
    error: str | None = None


class ModelsStatusResponse(BaseModel):
    models: list[ModelStatus]


class ProbabilityItem(BaseModel):
    class_id: str
    label_fr: str
    probability: float = Field(ge=0.0, le=1.0)


class AnalysisResponse(BaseModel):
    specialist: str
    specialist_display_name: str
    technical_model: str
    predicted_class: str
    predicted_label_fr: str
    confidence: float = Field(ge=0.0, le=1.0)
    probabilities: list[ProbabilityItem]
    device: str
    filename: str
