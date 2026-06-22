"""Health check endpoints."""

from fastapi import APIRouter

from app.schemas.common import HealthResponse


router = APIRouter()


@router.get("", response_model=HealthResponse)
def health_check() -> HealthResponse:
    """Return backend health status."""
    return HealthResponse(status="ok")

