"""
VoiceGuard AI - FastAPI Application Entrypoint
=================================================
Run with:
    uvicorn app:app --host 0.0.0.0 --port 8000 --reload

Docs available at:
    http://localhost:8000/docs        (Swagger UI)
    http://localhost:8000/redoc       (ReDoc)
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

from config import CORS_ORIGINS
from database import init_db
from api.routes import router
from models.inference import get_inference_engine


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting VoiceGuard AI backend...")
    init_db()
    logger.info("SQLite database initialized.")
    get_inference_engine()  # warm up / load model weights once at startup
    logger.info("Inference engine ready.")
    yield
    logger.info("Shutting down VoiceGuard AI backend.")


app = FastAPI(
    title="VoiceGuard AI",
    description="Deep learning system for detecting Human vs AI-generated voice audio.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)


@app.get("/", tags=["Root"])
def root():
    return {
        "service": "VoiceGuard AI",
        "status": "running",
        "docs": "/docs",
    }
