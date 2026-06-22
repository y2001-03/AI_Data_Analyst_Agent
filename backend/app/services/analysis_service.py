"""Analysis service layer."""

from uuid import uuid4

from app.core.exceptions import AppException
from app.schemas.analysis import ChatAnalysisRequest
from app.schemas.file import DatasetExecutionResponse
from app.services.dataset_flow_service import DatasetFlowService
from app.services.memory_service import MemoryService


class AnalysisService:
    """Coordinate question-driven dataset analysis."""

    def __init__(self) -> None:
        self.dataset_flow_service = DatasetFlowService()
        self.memory_service = MemoryService()

    async def run_chat_analysis(self, request: ChatAnalysisRequest) -> DatasetExecutionResponse:
        """Execute the dataset workflow for a question and stored dataset."""
        dataset_id = request.dataset_id or request.file_name
        session_id = request.session_id or str(uuid4())
        memory_context = self.memory_service.get(session_id)
        response = await self.dataset_flow_service.chat_upload_flow(
            dataset_id=dataset_id,
            question=request.question,
            session_id=session_id,
            memory_context=memory_context,
        )
        self.memory_service.update_context(
            session_id=session_id,
            question=request.question,
            dataset_info={
                "dataset_id": response.dataset_id,
                **response.file_info.model_dump(mode="json"),
            },
            last_execution={
                "execution_results": [item.model_dump(mode="json") for item in response.execution_results],
                "debug": response.debug.model_dump(mode="json"),
            },
        )
        return response
