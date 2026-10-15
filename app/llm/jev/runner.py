"""Scores every clean review with Jev (stage 5): one Score request per review,
bounded by a semaphore (CLAUDE.md §5.5 — never an unbounded gather).

No retries: a request that fails leaves that review unscored (score NaN in
jev_scores.csv, counted as unprocessed) rather than failing the job or being
replaced by a star-rating guess. If the key is rejected, or no review could
be scored at all, the job raises LLM_UNAVAILABLE.
"""

import asyncio
import math
import time
from dataclasses import dataclass, field
from typing import Any

import pandas as pd
import structlog

from app.core.config import settings
from app.core.errors import LLMUnavailableError
from app.llm.jev.client import JEVAuthError, JEVClient, JEVRequestError, ScoreRubric, jev_client, load_rubric
from app.models.jev import JEVReviewInput
from app.pipeline.io import SCORE_COLUMNS

log = structlog.get_logger(__name__)


@dataclass
class JEVStats:
    requests: int = 0
    scored: int = 0
    failed: int = 0
    input_tokens: int = 0
    elapsed_s: float = 0.0


@dataclass
class JEVRunResult:
    scores: pd.DataFrame
    unprocessed_review_ids: list[str]
    stats: JEVStats = field(default_factory=JEVStats)


def nearest_level(score: float, max_level: int) -> int:
    """Round half up (2.5 -> 3), clamped to the rubric."""
    return min(max(math.floor(score + 0.5), 0), max_level)


async def _score_all(
    client: JEVClient, reviews: list[JEVReviewInput], rubric: ScoreRubric, max_concurrency: int
) -> JEVRunResult:
    started = time.perf_counter()
    stats = JEVStats(requests=len(reviews))
    rows: dict[str, dict[str, Any]] = {}
    failed: list[str] = []
    auth_error: JEVAuthError | None = None
    semaphore = asyncio.Semaphore(max_concurrency)

    async def score_one(review: JEVReviewInput) -> None:
        nonlocal auth_error
        row: dict[str, Any] = {
            "review_id": review.review_id,
            "score": math.nan,
            "level": pd.NA,
            "confidence": math.nan,
            "model": client.model,
            "rubric_version": rubric.version,
        }
        async with semaphore:
            try:
                result = await client.score(review.text, rubric)
            except JEVAuthError as exc:
                # Recorded, not raised: a raise inside a task surfaces as an ExceptionGroup.
                auth_error = exc
                failed.append(review.review_id)
            except JEVRequestError as exc:
                failed.append(review.review_id)
                log.warning(
                    "jev_score_failed", review_id=review.review_id, status=exc.status_code, error=str(exc)
                )
            else:
                row |= {
                    "score": round(result.score, 4),
                    "level": nearest_level(result.score, rubric.max_level),
                    "confidence": round(result.confidence, 4),
                    "model": result.model,
                }
                stats.scored += 1
                stats.input_tokens += result.input_tokens
        rows[review.review_id] = row

    async with asyncio.TaskGroup() as tg:
        for review in reviews:
            tg.create_task(score_one(review))

    if auth_error is not None:
        raise LLMUnavailableError("Jev rejected the API key.") from auth_error
    if reviews and stats.scored == 0:
        raise LLMUnavailableError("Jev could not score any review.", details={"failed": len(failed)})

    stats.failed = len(failed)
    stats.elapsed_s = time.perf_counter() - started
    scores = pd.DataFrame([rows[r.review_id] for r in reviews], columns=SCORE_COLUMNS)
    log.info("jev_done", rubric_version=rubric.version, model=client.model, **stats.__dict__)
    return JEVRunResult(scores=scores, unprocessed_review_ids=failed, stats=stats)


async def run_jev(
    reviews: list[JEVReviewInput],
    *,
    client: JEVClient | None = None,
    rubric: ScoreRubric | None = None,
    max_concurrency: int | None = None,
) -> JEVRunResult:
    rubric = rubric or load_rubric()
    concurrency = max_concurrency or settings.jev_max_concurrency
    if client is not None:
        return await _score_all(client, reviews, rubric, concurrency)
    async with jev_client() as owned:
        return await _score_all(owned, reviews, rubric, concurrency)
