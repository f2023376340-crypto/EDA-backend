from fastapi import APIRouter

from app.models.api import AnalysisResult, AnalyzeRequest
from app.pipeline.orchestrator import run_analysis

router = APIRouter(tags=["analyze"])


@router.post("/analyze", response_model=AnalysisResult)
async def analyze(body: AnalyzeRequest) -> AnalysisResult:
    return await run_analysis(body.product_url)
