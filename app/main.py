"""FastAPI app factory."""

from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager

import structlog
from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api.v1.analyze import router as analyze_router
from app.api.v1.runs import router as runs_router
from app.core.config import settings
from app.core.errors import AppError
from app.scraper.client import close_browser

log = structlog.get_logger(__name__)


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    yield
    await close_browser()


def create_app() -> FastAPI:
    app = FastAPI(title="Daraz AI Review Analyzer", lifespan=lifespan)

    # Registered before CORS so CORS wraps it: an unexpected crash still returns
    # JSON with CORS headers, instead of a bare 500 the browser hides from the
    # frontend (which then misreports it as "backend unreachable").
    @app.middleware("http")
    async def unexpected_errors_as_json(
        request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        try:
            return await call_next(request)
        except Exception:
            log.exception("unhandled_error", path=request.url.path)
            return JSONResponse(
                status_code=500,
                content={
                    "error": {
                        "code": "INTERNAL_ERROR",
                        "message": "The analysis failed unexpectedly. The backend log has the details.",
                        "details": {},
                    }
                },
            )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins_list,
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )

    @app.exception_handler(AppError)
    async def handle_app_error(_: Request, exc: AppError) -> JSONResponse:
        return JSONResponse(status_code=exc.http_status, content=exc.to_body())

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    app.include_router(analyze_router, prefix="/api/v1")
    app.include_router(runs_router, prefix="/api/v1")
    return app


app = create_app()
