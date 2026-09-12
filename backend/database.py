"""
VoiceGuard AI - Database Layer (SQLite via SQLAlchemy)
=========================================================
Stores every prediction made by the system for the History page and for
future model-monitoring / drift-analysis purposes.
"""

from __future__ import annotations

import datetime as dt
from typing import List, Optional

from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, JSON
from sqlalchemy.orm import declarative_base, sessionmaker, Session

from config import DB_PATH

DATABASE_URL = f"sqlite:///{DB_PATH}"

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


class PredictionHistory(Base):
    __tablename__ = "prediction_history"

    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String, nullable=False)
    prediction = Column(String, nullable=False)          # "Human" | "AI Voice"
    confidence = Column(Float, nullable=False)
    human_probability = Column(Float, nullable=False)
    ai_probability = Column(Float, nullable=False)
    processing_time = Column(String, nullable=False)
    duration_sec = Column(Float, nullable=True)
    acoustic_features = Column(JSON, nullable=True)
    created_at = Column(DateTime, default=dt.datetime.utcnow)


def init_db():
    Base.metadata.create_all(bind=engine)


def get_db_session() -> Session:
    db = SessionLocal()
    try:
        return db
    finally:
        pass  # closed explicitly by caller / FastAPI dependency


def save_prediction(db: Session, filename: str, result_dict: dict) -> PredictionHistory:
    record = PredictionHistory(
        filename=filename,
        prediction=result_dict["prediction"],
        confidence=result_dict["confidence"],
        human_probability=result_dict["human_probability"],
        ai_probability=result_dict["ai_probability"],
        processing_time=result_dict["processing_time"],
        duration_sec=result_dict.get("duration_sec"),
        acoustic_features=result_dict.get("acoustic_features", {}),
    )
    db.add(record)
    db.commit()
    db.refresh(record)
    return record


def get_all_predictions(db: Session, limit: int = 100) -> List[PredictionHistory]:
    return (
        db.query(PredictionHistory)
        .order_by(PredictionHistory.created_at.desc())
        .limit(limit)
        .all()
    )


def clear_predictions(db: Session) -> int:
    count = db.query(PredictionHistory).delete()
    db.commit()
    return count
