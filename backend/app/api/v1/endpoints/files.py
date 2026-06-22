"""File ingestion endpoints."""

from fastapi import APIRouter, File, Form, Query, UploadFile
from fastapi.responses import StreamingResponse

from app.schemas.file import DatasetExecutionResponse
from app.services.dataset_flow_service import DatasetFlowService


router = APIRouter()
service = DatasetFlowService()


@router.post("/upload", response_model=DatasetExecutionResponse)
async def upload_file(
    file: UploadFile = File(...),
    session_id: str | None = Form(default=None),
    stream: bool = Query(False),
):
    """Accept a dataset file and run the full analysis flow."""
    if not stream:
        return await service.normal_upload_flow(file, session_id=session_id)
    return StreamingResponse(
        service.stream_upload_flow(file),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
