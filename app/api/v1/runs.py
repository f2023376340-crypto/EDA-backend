from fastapi import APIRouter

from app.models.api import AnalysisResult, RunSummary
from app.pipeline.runs import get_run, list_runs

router = APIRouter(tags=["runs"])


@router.get("/runs", response_model=list[RunSummary])
def runs() -> list[RunSummary]:
    return list_runs()


@router.get("/runs/{run_id}", response_model=AnalysisResult)
def run(run_id: str) -> AnalysisResult:
    return get_run(run_id)
