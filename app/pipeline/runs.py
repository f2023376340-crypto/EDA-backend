"""Saved analyses (run folders with an analysis.json), for re-opening past runs."""

import structlog
from pydantic import ValidationError

from app.core.errors import RunNotFoundError
from app.models.api import AnalysisResult, RunSummary
from app.pipeline import io

log = structlog.get_logger(__name__)


def list_runs(limit: int = 50) -> list[RunSummary]:
    """Most recent first. Runs saved by an older AnalysisResult shape are skipped."""
    summaries: list[RunSummary] = []
    for run_dir in io.analysed_run_dirs():
        try:
            result = io.read_analysis(run_dir)
        except ValidationError:
            log.info("run_skipped_old_format", run_id=run_dir.name)
            continue
        summaries.append(
            RunSummary(
                run_id=result.run_id,
                item_id=result.product.item_id,
                product_name=result.product.name,
                product_category=result.product.product_category,
                url=result.product.url,
                scraped_at=result.scraped_at,
                reviews_scored=result.stats.reviews_scored,
                avg_score=result.avg_score,
                max_level=result.scale.max_level,
                net_sentiment=result.net_sentiment,
            )
        )
        if len(summaries) == limit:
            break
    return summaries


def get_run(run_id: str) -> AnalysisResult:
    run_dir = io.find_run_dir(run_id)
    if run_dir is None:
        raise RunNotFoundError("No saved analysis with that id.", details={"run_id": run_id})
    try:
        return io.read_analysis(run_dir)
    except ValidationError as exc:
        raise RunNotFoundError(
            "This analysis was saved in an older format; analyse the product again.",
            details={"run_id": run_id},
        ) from exc
