"""Evaluates every clean review with Jev (stage 5): one request per review,
bounded by a semaphore (CLAUDE.md §5.5 — never an unbounded gather). Each
request carries the overall sentiment Score plus, for every generated category
except "Overall", a Noul (does the review talk about it?) and a Score.

Outputs:
- scores: one row per review (io.SCORE_COLUMNS)
- category_scores: one row per (review, category) (io.CATEGORY_SCORE_COLUMNS).
  score/level/confidence are filled only when the mention probability is >=
  MENTION_THRESHOLD; Jev still answers the category Score when the review never
  mentions it, and that answer is meaningless.

No retries: a request that fails leaves that review unscored (NaN in
jev_scores.csv, no category rows, counted as unprocessed) rather than failing
the job or being replaced by a star-rating guess. If the key is rejected, or
no review could be scored at all, the job raises LLM_UNAVAILABLE.
"""

import asyncio
import math
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import pandas as pd
import structlog

from app.core.config import settings
from app.core.errors import LLMUnavailableError
from app.llm.jev.client import (
    JEVAuthError,
    JEVClient,
    JEVRequestError,
    ReviewEvaluation,
    ScoreRubric,
    jev_client,
    load_rubric,
)
from app.models.jev import OVERALL_FEATURE_NAME, Feature, JEVReviewInput
from app.pipeline.io import CATEGORY_SCORE_COLUMNS, SCORE_COLUMNS

log = structlog.get_logger(__name__)

MENTION_THRESHOLD = 0.5

# (review, its jev_scores row, its jev_category_scores rows), as in io.SCORE_COLUMNS / CATEGORY_SCORE_COLUMNS.
ReviewCallback = Callable[[JEVReviewInput, dict[str, Any], list[dict[str, Any]]], None]


@dataclass
class JEVStats:
    requests: int = 0
    scored: int = 0
    failed: int = 0
    categories: int = 0
    input_tokens: int = 0
    elapsed_s: float = 0.0


@dataclass
class JEVRunResult:
    scores: pd.DataFrame
    category_scores: pd.DataFrame
    unprocessed_review_ids: list[str]
    stats: JEVStats = field(default_factory=JEVStats)


def nearest_level(score: float, max_level: int) -> int:
    """Round half up (2.5 -> 3), clamped to the rubric."""
    return min(max(math.floor(score + 0.5), 0), max_level)


def _category_rows(review_id: str, result: ReviewEvaluation, max_level: int) -> list[dict[str, Any]]:
    rows = []
    for c in result.categories:
        mentioned = c.mention_probability >= MENTION_THRESHOLD
        rows.append(
            {
                "review_id": review_id,
                "feature": c.name,
                "mention_probability": round(c.mention_probability, 4),
                "mentioned": mentioned,
                "score": round(c.score, 4) if mentioned else math.nan,
                "level": nearest_level(c.score, max_level) if mentioned else pd.NA,
                "confidence": round(c.confidence, 4) if mentioned else math.nan,
            }
        )
    return rows


async def _evaluate_all(
    client: JEVClient,
    reviews: list[JEVReviewInput],
    categories: list[Feature],
    rubric: ScoreRubric,
    max_concurrency: int,
    on_review: ReviewCallback | None,
) -> JEVRunResult:
    started = time.perf_counter()
    stats = JEVStats(requests=len(reviews), categories=len(categories))
    score_rows: dict[str, dict[str, Any]] = {}
    category_rows: dict[str, list[dict[str, Any]]] = {}
    failed: list[str] = []
    auth_error: JEVAuthError | None = None
    semaphore = asyncio.Semaphore(max_concurrency)

    async def evaluate_one(review: JEVReviewInput) -> None:
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
                result = await client.evaluate(review.text, rubric, categories)
            except JEVAuthError as exc:
                # Recorded, not raised: a raise inside a task surfaces as an ExceptionGroup.
                auth_error = exc
                failed.append(review.review_id)
            except JEVRequestError as exc:
                failed.append(review.review_id)
                log.warning(
                    "jev_review_failed", review_id=review.review_id, status=exc.status_code, error=str(exc)
                )
            else:
                row |= {
                    "score": round(result.score, 4),
                    "level": nearest_level(result.score, rubric.max_level),
                    "confidence": round(result.confidence, 4),
                    "model": result.model,
                }
                category_rows[review.review_id] = _category_rows(review.review_id, result, rubric.max_level)
                stats.scored += 1
                stats.input_tokens += result.input_tokens
        score_rows[review.review_id] = row
        if on_review:
            on_review(review, row, category_rows.get(review.review_id, []))

    async with asyncio.TaskGroup() as tg:
        for review in reviews:
            tg.create_task(evaluate_one(review))

    if auth_error is not None:
        raise LLMUnavailableError("Jev rejected the API key.") from auth_error
    if reviews and stats.scored == 0:
        raise LLMUnavailableError("Jev could not score any review.", details={"failed": len(failed)})

    stats.failed = len(failed)
    stats.elapsed_s = time.perf_counter() - started
    log.info("jev_done", rubric_version=rubric.version, model=client.model, **stats.__dict__)
    return JEVRunResult(
        scores=pd.DataFrame([score_rows[r.review_id] for r in reviews], columns=SCORE_COLUMNS),
        category_scores=pd.DataFrame(
            [row for r in reviews for row in category_rows.get(r.review_id, [])],
            columns=CATEGORY_SCORE_COLUMNS,
        ),
        unprocessed_review_ids=failed,
        stats=stats,
    )


async def run_jev(
    reviews: list[JEVReviewInput],
    features: list[Feature],
    *,
    client: JEVClient | None = None,
    rubric: ScoreRubric | None = None,
    max_concurrency: int | None = None,
    on_review: ReviewCallback | None = None,
) -> JEVRunResult:
    """`features` is the generated FeatureSet; "Overall" is never asked about —
    aggregation derives it from reviews that mention no other category.
    `on_review(review, score_row, category_rows)` is called as each review finishes.
    """
    rubric = rubric or load_rubric()
    categories = [f for f in features if f.name != OVERALL_FEATURE_NAME]
    concurrency = max_concurrency or settings.jev_max_concurrency
    if client is not None:
        return await _evaluate_all(client, reviews, categories, rubric, concurrency, on_review)
    async with jev_client() as owned:
        return await _evaluate_all(owned, reviews, categories, rubric, concurrency, on_review)
