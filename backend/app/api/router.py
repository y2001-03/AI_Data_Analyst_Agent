"""API router configuration."""

from fastapi import APIRouter

from app.api.v1.endpoints import analysis, files, health


api_router = APIRouter()
api_router.include_router(health.router, prefix="/v1/health", tags=["health"])
api_router.include_router(analysis.router, prefix="/v1/analysis", tags=["analysis"])
api_router.include_router(files.router, prefix="/v1/datasets", tags=["datasets"])
