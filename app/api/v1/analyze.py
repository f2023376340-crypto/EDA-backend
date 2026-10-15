from fastapi import APIRouter

from app.models.api import AnalyzeRequest, AnalyzeResponse
from app.pipeline.orchestrator import run_analysis

router = APIRouter(tags=["analyze"])


@router.post("/analyze", response_model=AnalyzeResponse)
async def analyze(body: AnalyzeRequest) -> AnalyzeResponse:
    return await run_analysis(body.product_url)
