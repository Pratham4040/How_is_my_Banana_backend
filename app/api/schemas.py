from __future__ import annotations

from pydantic import BaseModel, Field


class PredictionResponse(BaseModel):
    prediction: str = Field(..., description="Legacy prediction label for backward compatibility")
    label: str = Field(..., description="Predicted label")
    class_id: int = Field(..., ge=0, description="Zero-based class index")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score")


class HealthResponse(BaseModel):
    message: str
