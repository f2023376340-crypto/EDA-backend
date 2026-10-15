import asyncio
import json
from collections.abc import AsyncIterator
from typing import Any

import structlog
from fastapi import APIRouter, Query
from fastapi.responses import StreamingResponse

from app.core.errors import AppError
from app.models.api import AnalysisResult, AnalyzeRequest
from app.pipeline.orchestrator import run_analysis

router = APIRouter(tags=["analyze"])
log = structlog.get_logger(__name__)

_KEEPALIVE_S = 15.0


@router.post("/analyze", response_model=AnalysisResult)
async def analyze(body: AnalyzeRequest) -> AnalysisResult:
    return await run_analysis(body.product_url)


@router.get("/analyze/stream")
async def analyze_stream(product_url: str = Query(min_length=1)) -> StreamingResponse:
    """Runs the same pipeline as POST /analyze, streaming progress as
    Server-Sent Events (event names in app/pipeline/progress.py) and ending with
    `done` (the AnalysisResult) or `failed` ({code, message}). Not "error":
    EventSource reserves that name for connection errors. Closing the
    connection cancels the analysis.
    """
    loop = asyncio.get_running_loop()
    queue: asyncio.Queue[tuple[str, dict[str, Any]] | None] = asyncio.Queue()

    def emit(event: str, data: dict[str, Any]) -> None:
        # Called from the request loop and from the scraper's Playwright thread.
        loop.call_soon_threadsafe(queue.put_nowait, (event, data))

    async def work() -> None:
        try:
            result = await run_analysis(product_url, emit=emit)
            emit("done", result.model_dump(mode="json"))
        except AppError as exc:
            emit("failed", {"code": exc.code, "message": exc.message})
        except Exception:
            log.exception("analyze_stream_failed")
            emit(
                "failed",
                {
                    "code": "INTERNAL_ERROR",
                    "message": "The analysis failed unexpectedly. The backend log has the details.",
                },
            )
        finally:
            loop.call_soon_threadsafe(queue.put_nowait, None)

    async def events() -> AsyncIterator[str]:
        task = asyncio.create_task(work())
        try:
            while True:
                try:
                    item = await asyncio.wait_for(queue.get(), timeout=_KEEPALIVE_S)
                except TimeoutError:
                    yield ": keep-alive\n\n"
                    continue
                if item is None:
                    break
                name, data = item
                yield f"event: {name}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"
        finally:
            if not task.done():
                task.cancel()

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
