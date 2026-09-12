"""
VoiceGuard AI - API Routes
============================
Endpoints:
    POST   /api/predict   - upload an audio file, get Human/AI verdict
    GET    /api/history   - list past predictions
    DELETE /api/history   - clear prediction history
    GET    /api/health    - service/model health check
"""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, UploadFile, File, HTTPException, Depends
from sqlalchemy.orm import Session

import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))

from config import ALLOWED_EXTENSIONS, MAX_UPLOAD_MB, DEVICE
from database import SessionLocal, save_prediction, get_all_predictions, clear_predictions
from models.inference import get_inference_engine
from api.schemas import (
    PredictResponse, HistoryResponse, HistoryItem,
    DeleteHistoryResponse, HealthResponse,
)

router = APIRouter(prefix="/api", tags=["VoiceGuard"])


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def _validate_upload(file: UploadFile, contents: bytes):
    ext = Path(file.filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{ext}'. Allowed: {sorted(ALLOWED_EXTENSIONS)}",
        )
    size_mb = len(contents) / (1024 * 1024)
    if size_mb > MAX_UPLOAD_MB:
        raise HTTPException(
            status_code=400,
            detail=f"File too large ({size_mb:.1f}MB). Max allowed is {MAX_UPLOAD_MB}MB.",
        )


@router.post("/predict", response_model=PredictResponse)
async def predict(file: UploadFile = File(...), db: Session = Depends(get_db)):
    """Run Human vs AI-generated voice detection on an uploaded audio clip."""
    contents = await file.read()
    _validate_upload(file, contents)

    try:
        engine = get_inference_engine()
        result = engine.predict(contents, file.filename)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Inference failed: {exc}")

    result_dict = result.__dict__
    save_prediction(db, filename=file.filename, result_dict=result_dict)

    return PredictResponse(**result_dict)


@router.get("/history", response_model=HistoryResponse)
def history(limit: int = 100, db: Session = Depends(get_db)):
    """Return the most recent predictions (newest first)."""
    records = get_all_predictions(db, limit=limit)
    items = [HistoryItem.model_validate(r) for r in records]
    return HistoryResponse(count=len(items), items=items)


@router.delete("/history", response_model=DeleteHistoryResponse)
def delete_history(db: Session = Depends(get_db)):
    """Wipe all stored prediction history."""
    deleted = clear_predictions(db)
    return DeleteHistoryResponse(deleted_count=deleted, message="History cleared successfully.")


@router.get("/health", response_model=HealthResponse)
def health():
    """Lightweight liveness/readiness probe, also reports whether a real
    trained checkpoint is loaded vs. randomly-initialized weights."""
    from config import CHECKPOINT_PATH
    return HealthResponse(
        status="ok",
        model_loaded=CHECKPOINT_PATH.exists(),
        device=str(DEVICE),
        version="1.0.0",
    )
