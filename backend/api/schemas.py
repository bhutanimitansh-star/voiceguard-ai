"""
VoiceGuard AI - Pydantic Schemas
==================================
Request/response contracts for the FastAPI layer.
"""

from __future__ import annotations

import datetime as dt
from typing import List, Dict, Optional

from pydantic import BaseModel, Field


class SuspiciousRegion(BaseModel):
    start_sec: float
    end_sec: float
    intensity: float


class PredictResponse(BaseModel):
    prediction: str = Field(..., example="AI Voice")
    confidence: float = Field(..., example=96.4)
    human_probability: float = Field(..., example=3.6)
    ai_probability: float = Field(..., example=96.4)
    processing_time: str = Field(..., example="1.42 s")
    suspicious_regions: List[SuspiciousRegion] = []
    waveform_png_base64: str
    melspectrogram_png_base64: str
    gradcam_png_base64: str
    acoustic_features: Dict[str, float]
    duration_sec: float

    class Config:
        json_schema_extra = {
            "example": {
                "prediction": "AI Voice",
                "confidence": 96.4,
                "human_probability": 3.6,
                "ai_probability": 96.4,
                "processing_time": "1.42 s",
            }
        }


class HistoryItem(BaseModel):
    id: int
    filename: str
    prediction: str
    confidence: float
    human_probability: float
    ai_probability: float
    processing_time: str
    duration_sec: Optional[float]
    created_at: dt.datetime

    class Config:
        from_attributes = True


class HistoryResponse(BaseModel):
    count: int
    items: List[HistoryItem]


class DeleteHistoryResponse(BaseModel):
    deleted_count: int
    message: str


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    device: str
    version: str
