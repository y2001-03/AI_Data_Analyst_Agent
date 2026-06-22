"""Analysis endpoints."""

from fastapi import APIRouter

from app.schemas.analysis import ChatAnalysisRequest
from app.schemas.file import DatasetExecutionResponse
from app.services.analysis_service import AnalysisService


router = APIRouter()
service = AnalysisService()


@router.post("/chat", response_model=DatasetExecutionResponse)
async def analyze_dataset(request: ChatAnalysisRequest) -> DatasetExecutionResponse:
    """Run question-driven analysis against a previously uploaded dataset."""
    return await service.run_chat_analysis(request)
