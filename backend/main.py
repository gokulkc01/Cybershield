"""
CyberShield v2 Backend - FastAPI Application

Behavioral Adversarial Robustness Intelligence Platform

This backend operationalizes:
- Session-centric Transformer detector
- Deterministic mutation engine
- Robustness evaluation pipeline
- Failure analysis system
"""

import os
import sys
from pathlib import Path
from contextlib import asynccontextmanager

# Add project root to Python path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import logging

from app.core.config import settings
from app.api import detection, mutation, robustness, artifacts, predict

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan context manager.
    Handles startup and shutdown events.
    """
    logger.info("CyberShield v2 Backend Starting...")
    yield
    logger.info("CyberShield v2 Backend Shutting Down...")


# Create FastAPI application
app = FastAPI(
    title="CyberShield v2 API",
    description="Behavioral Adversarial Robustness Intelligence Platform",
    version="2.0.0",
    lifespan=lifespan,
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "CyberShield v2 Backend",
        "version": "2.0.0"
    }


# Root endpoint
@app.get("/")
async def root():
    """Root endpoint - API information."""
    return {
        "service": "CyberShield v2 Backend",
        "version": "2.0.0",
        "description": "Behavioral Adversarial Robustness Intelligence Platform",
        "docs": "/docs",
        "endpoints": {
            "detection": "/api/v1/detection",
            "mutation": "/api/v1/mutation",
            "robustness": "/api/v1/robustness",
            "artifacts": "/api/v1/artifacts",
        }
    }


# Include API routers
app.include_router(
    detection.router,
    prefix="/api/v1/detection",
    tags=["Detection"]
)

app.include_router(
    mutation.router,
    prefix="/api/v1/mutation",
    tags=["Mutation"]
)

app.include_router(
    robustness.router,
    prefix="/api/v1/robustness",
    tags=["Robustness"]
)

app.include_router(
    artifacts.router,
    prefix="/api/v1/artifacts",
    tags=["Artifacts"]
)

app.include_router(
    predict.router,
    prefix="/api/v1/predict",
    tags=["Predict"]
)


# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler."""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "message": str(exc)
        }
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=settings.DEBUG
    )
