"""Orchestration layer for the dataset analysis flow."""

from __future__ import annotations

import asyncio
import json
from io import BytesIO
from pathlib import Path
from typing import AsyncIterator, Awaitable, Callable
from uuid import uuid4

from fastapi import HTTPException, UploadFile
import pandas as pd

from app.core.context import trace_id_var
from app.core.database import create_async_session
from app.core.logging import get_logger
from app.graph.dataset_workflow import build_dataset_upload_graph
from app.models.analysis import AnalysisResult, AnalysisSession
from app.models.dataset import Dataset
from app.repositories.analysis_repository import (
    AnalysisResultRepository,
    AnalysisSessionRepository,
)
from app.repositories.dataset_repository import DatasetRepository
from app.schemas.file import (
    DatasetExecutionResponse,
    DebugFallbackSummary,
    DebugResponse,
    TraceEntry,
    TraceResponse,
)
from app.services.dataset_context_service import DatasetContextService

logger = get_logger(__name__)


class DatasetFlowService:
    """Compose parsing, understanding, planning, and execution."""

    def __init__(self) -> None:
        self.graph = build_dataset_upload_graph()
        self.dataset_context_service = DatasetContextService()

    async def run_upload_flow(self, file: UploadFile) -> DatasetExecutionResponse:
        """Execute the full upload flow and return the final JSON payload."""
        return await self.normal_upload_flow(file)

    async def normal_upload_flow(
        self,
        file: UploadFile,
        session_id: str | None = None,
    ) -> DatasetExecutionResponse:
        """Execute the full upload flow without streaming."""
        result, dataset_id = await self.unified_run_graph(file, session_id=session_id)
        response = self._build_response(result, dataset_id=dataset_id)
        # Persist to database (fire-and-forget — never fails the response).
        await self._persist_dataset(file, result, dataset_id)
        return response

    async def chat_upload_flow(
        self,
        dataset_id: str | None,
        question: str,
        session_id: str | None = None,
        memory_context: dict[str, object] | None = None,
    ) -> DatasetExecutionResponse:
        """Execute the dataset workflow for a stored dataset and user question."""
        resolved_dataset_id = dataset_id
        if not resolved_dataset_id and session_id:
            resolved_dataset_id = self.dataset_context_service.get_latest_dataset_id(session_id)
        if not resolved_dataset_id:
            self._raise_stage_error("understand", "No dataset_id was provided and no recent dataset exists for this session.", 400)
        dataset = self.dataset_context_service.get_dataset(resolved_dataset_id)

        # Create analysis session record before running the graph.
        analysis_session = AnalysisSession(
            dataset_id=resolved_dataset_id,
            question=question,
            status="running",
        )
        await self._persist_session(analysis_session)

        result = await self._run_graph(
            file_name=dataset.file_name,
            content=dataset.content,
            question=question,
            memory_context=memory_context,
            dataframe=dataset.dataframe.copy(),
        )
        response = self._build_response(result, dataset_id=resolved_dataset_id)

        # Persist analysis result and mark session completed.
        await self._persist_result(analysis_session.id, response)
        return response

    async def stream_upload_flow(self, file: UploadFile) -> AsyncIterator[str]:
        """Stream node progress and emit the final JSON payload."""
        try:
            progress_queue: asyncio.Queue[dict[str, object] | None] = asyncio.Queue()

            async def enqueue_progress(payload: dict[str, object]) -> None:
                await progress_queue.put(payload)

            graph_task = asyncio.create_task(
                self.unified_run_graph(file, on_node_complete=enqueue_progress)
            )
            yield self._format_sse(
                "progress",
                {
                    "node": "understanding",
                    "status": "started",
                    "file_name": file.filename or "unknown",
                },
            )
            while True:
                if graph_task.done() and progress_queue.empty():
                    break
                payload = await progress_queue.get()
                if payload is None:
                    break
                yield self._format_sse("progress", payload)
            result = await graph_task
            response = self._build_response(result)
            yield self._format_sse("progress", {"node": "finished", "status": "completed"})
            yield self._format_sse(
                "result",
                response.model_dump(mode="json"),
            )
        except HTTPException as exc:
            detail = exc.detail if isinstance(exc.detail, dict) else {"reason": str(exc.detail)}
            logger.warning("SSE stream error | error=%s", detail)
            yield self._format_sse(
                "error",
                {
                    "node": "finished",
                    "status": "failed",
                    "error": detail,
                },
            )
        except Exception as exc:
            logger.exception("SSE stream unexpected error")
            yield self._format_sse(
                "error",
                {
                    "node": "finished",
                    "status": "failed",
                    "error": {"reason": str(exc)},
                },
            )

    async def unified_run_graph(
        self,
        file: UploadFile,
        session_id: str | None = None,
        on_node_complete: Callable[[dict[str, object] | None], Awaitable[None]] | None = None,
    ) -> tuple[dict[str, object], str]:
        """Execute LangGraph once and return the final graph state."""
        content = await file.read()
        file_name = file.filename or "unknown"
        dataframe = self._load_dataframe(file_name, content)
        result = await self._run_graph(
            file_name=file_name,
            content=content,
            question=None,
            memory_context=None,
            dataframe=dataframe.copy(),
            on_node_complete=on_node_complete,
        )
        dataset_id = str(uuid4())
        file_info = result["file_info"]
        schema = [column.model_dump(mode="json") for column in file_info.columns] if file_info is not None else []
        preview = file_info.preview if file_info is not None else []
        self.dataset_context_service.save_dataset(
            dataset_id=dataset_id,
            file_name=file_name,
            content=content,
            dataframe=dataframe,
            schema=schema,
            preview=preview,
            session_id=session_id,
        )
        return result, dataset_id

    async def _run_graph(
        self,
        file_name: str,
        content: bytes,
        question: str | None,
        memory_context: dict[str, object] | None = None,
        dataframe: pd.DataFrame | None = None,
        on_node_complete: Callable[[dict[str, object] | None], Awaitable[None]] | None = None,
    ) -> dict[str, object]:
        """Execute LangGraph once and return the final graph state."""
        trace_id = str(uuid4())
        trace_id_var.set(trace_id)
        logger.info(
            "Graph run started | file_name=%s question=%s trace_id=%s",
            file_name,
            question is not None,
            trace_id,
        )
        initial_state = self._build_initial_state(
            file_name,
            content,
            question,
            memory_context,
            dataframe=dataframe,
        )
        last_values: dict[str, object] | None = None
        async for mode, chunk in self.graph.astream(
            initial_state,
            stream_mode=["updates", "values"],
        ):
            if mode == "updates":
                node_name, node_state = self._extract_update(chunk)
                if node_name is None or on_node_complete is None:
                    continue
                await on_node_complete(self._build_progress_payload(node_name, node_state))
                continue
            if mode == "values" and isinstance(chunk, dict):
                last_values = chunk
        if on_node_complete is not None:
            await on_node_complete(None)
        if last_values is None:
            logger.error("Graph stream completed without final state | trace_id=%s", trace_id)
            raise RuntimeError("Graph stream completed without final state.")
        logger.info("Graph run completed | trace_id=%s", trace_id)
        return last_values

    def _raise_stage_error(self, stage: str, reason: str, status_code: int) -> None:
        """Raise a structured stage error."""
        raise HTTPException(
            status_code=status_code,
            detail={"stage": stage, "reason": reason},
        )

    # ------------------------------------------------------------------
    # Persistence helpers (fire-and-forget — never fail the response)
    # ------------------------------------------------------------------

    async def _persist_dataset(
        self,
        file: UploadFile,
        result: dict[str, object],
        dataset_id: str,
    ) -> None:
        """Save dataset metadata to PostgreSQL."""
        try:
            file_info = result.get("file_info")
            db = create_async_session()
            try:
                repo = DatasetRepository(db)
                dataset = Dataset(
                    id=dataset_id,
                    filename=file.filename or "unknown",
                    file_size=getattr(file, "size", 0) or 0,
                    file_type=(file.filename or "unknown").rsplit(".", 1)[-1]
                    if file.filename
                    else "csv",
                    row_count=file_info.row_count if file_info is not None else 0,
                    column_count=file_info.column_count if file_info is not None else 0,
                    schema_info=[c.model_dump(mode="json") for c in file_info.columns]
                    if file_info is not None and file_info.columns
                    else None,
                )
                await repo.create(dataset)
                await db.commit()
                logger.info("Dataset persisted | id=%s filename=%s", dataset_id, dataset.filename)
            finally:
                await db.close()
        except Exception:
            logger.exception("Failed to persist dataset | id=%s", dataset_id)

    async def _persist_session(self, session: AnalysisSession) -> None:
        """Create an analysis session record."""
        try:
            db = create_async_session()
            try:
                repo = AnalysisSessionRepository(db)
                await repo.create(session)
                await db.commit()
                logger.info(
                    "Analysis session created | id=%s dataset_id=%s status=%s",
                    session.id,
                    session.dataset_id,
                    session.status,
                )
            finally:
                await db.close()
        except Exception:
            logger.exception("Failed to persist analysis session | id=%s", session.id)

    async def _persist_result(
        self,
        session_id: str,
        response: DatasetExecutionResponse,
    ) -> None:
        """Save analysis result and mark the session as completed."""
        try:
            db = create_async_session()
            try:
                session_repo = AnalysisSessionRepository(db)
                result_repo = AnalysisResultRepository(db)

                analysis_result = AnalysisResult(
                    session_id=session_id,
                    result=response.model_dump(mode="json"),
                )
                await result_repo.create(analysis_result)
                await session_repo.mark_completed(session_id)
                await db.commit()
                logger.info(
                    "Analysis result persisted | session_id=%s", session_id,
                )
            finally:
                await db.close()
        except Exception:
            logger.exception("Failed to persist analysis result | session_id=%s", session_id)

    def _build_initial_state(
        self,
        file_name: str,
        content: bytes,
        question: str | None,
        memory_context: dict[str, object] | None,
        dataframe: pd.DataFrame | None,
    ) -> dict[str, object]:
        """Build the standard initial graph state."""
        return {
            "file_name": file_name,
            "content": content,
            "question": question,
            "memory_context": memory_context,
            "dataframe": dataframe,
            "file_info": None,
            "schema": [],
            "ai_analysis": None,
            "tasks": [],
            "planner_plan": None,
            "planner_status": None,
            "planner_issues": [],
            "normalized_intent": None,
            "execution_stages": [],
            "execution_order": [],
            "dependency_count": 0,
            "execution_supported": True,
            "execution_results": [],
            "planner_failed": False,
            "execution_failed": False,
            "error_stage": None,
            "error_reason": None,
            "node_status": {},
            "trace_log": [],
            "execution_path": [],
            "fallback_used": False,
            "fallback_reason": None,
        }

    def _build_response(
        self,
        result: dict[str, object],
        dataset_id: str | None = None,
    ) -> DatasetExecutionResponse:
        """Convert the final graph state into the public response model."""
        if result["file_info"] is None:
            self._raise_stage_error(
                str(result["error_stage"] or "understand"),
                str(result["error_reason"] or "Graph did not produce file info."),
                500,
            )
        if result["ai_analysis"] is None:
            self._raise_stage_error("understand", "Graph did not produce required outputs.", 500)
        return DatasetExecutionResponse(
            dataset_id=dataset_id,
            file_info=result["file_info"],
            ai_analysis=result["ai_analysis"],
            execution_results=result["execution_results"],
            trace=TraceResponse(
                execution_path=result["execution_path"],
                node_status=result["node_status"],
                trace_log=[TraceEntry(**entry) for entry in result["trace_log"]],
                fallback_used=bool(result["fallback_used"]),
                fallback_reason=result["fallback_reason"],
            ),
            debug=DebugResponse(
                execution_path=result["execution_path"],
                node_status=result["node_status"],
                trace_log=[TraceEntry(**entry) for entry in result["trace_log"]],
                fallback_summary=DebugFallbackSummary(
                    understand=result["node_status"].get("understand") == "fallback",
                    plan=result["node_status"].get("plan") == "fallback",
                ),
            ),
        )

    def _extract_update(
        self,
        chunk: object,
    ) -> tuple[str | None, dict[str, object]]:
        """Extract one node update from a LangGraph updates chunk."""
        if not isinstance(chunk, dict) or not chunk:
            return None, {}
        node_name = next(iter(chunk))
        node_state = chunk[node_name]
        if not isinstance(node_state, dict):
            return node_name, {}
        return node_name, node_state

    def _build_progress_payload(
        self,
        node_name: str,
        node_state: dict[str, object],
    ) -> dict[str, object]:
        """Build a public progress event from one completed node."""
        node_alias = {
            "understand": "understanding",
            "plan": "planning",
            "execute": "executing",
        }.get(node_name, node_name)
        trace_log = node_state.get("trace_log")
        latest_trace = trace_log[-1] if isinstance(trace_log, list) and trace_log else None
        if not isinstance(latest_trace, dict):
            latest_trace = None
        node_status = node_state.get("node_status", {})
        status = "completed"
        if isinstance(node_status, dict):
            status = str(node_status.get(node_name, "completed"))
        payload: dict[str, object] = {
            "node": node_alias,
            "status": status,
        }
        if latest_trace is not None:
            payload["trace"] = latest_trace
        return payload

    def _format_sse(self, event: str, payload: dict[str, object]) -> str:
        """Serialize one SSE event payload."""
        body = json.dumps(payload, ensure_ascii=False, default=str)
        return f"event: {event}\ndata: {body}\n\n"

    def _load_dataframe(self, file_name: str, content: bytes) -> pd.DataFrame:
        """Load raw file bytes into a dataframe for caching and execution."""
        extension = Path(file_name).suffix.lower()
        buffer = BytesIO(content)
        if extension == ".csv":
            return pd.read_csv(buffer)
        return pd.read_excel(buffer)
