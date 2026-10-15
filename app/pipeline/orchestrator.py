"""Composes the pipeline (CLAUDE.md §2). Every stage reads its inputs from and
writes its outputs to the run folder via app/pipeline/io.py, so each one can
also be re-run on its own from scripts/.

    run_analysis: validate URL -> scrape -> analyse_run
    analyse_run:  stage_clean -> stage_features (Groq) -> stage_jev (Jev) -> stage_aggregate

Every stage reports progress through `emit` (see app/pipeline/progress.py);
the streaming endpoint turns those into Server-Sent Events.
"""

import asyncio
import math
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from app.core.url_validator import validate_daraz_url
from app.llm.feature_generator import generate_features
from app.llm.jev.client import load_rubric
from app.llm.jev.runner import JEVStats, ReviewCallback, run_jev
from app.models.api import AnalysisResult, RunStats
from app.models.jev import FeatureGenerationOutcome, JEVReviewInput
from app.pipeline import io
from app.pipeline.aggregation import build_analysis_result, level_sentiment
from app.pipeline.cleaning import clean_reviews
from app.pipeline.jev_inputs import reviews_to_jev_inputs
from app.pipeline.progress import Emit, no_emit
from app.scraper import client as scraper

# §5.4: at most one concurrent product scrape per process.
_scrape_slot = asyncio.Semaphore(1)
_SCORED_TEXT_CHARS = 280


async def scrape_to_run(product_url: str, *, max_pages: int | None = None, emit: Emit = no_emit) -> Path:
    emit("stage", {"stage": "validating", "message": "Checking the link"})
    ref = validate_daraz_url(product_url)
    emit("stage", {"stage": "scraping", "message": "Opening the product page and collecting reviews"})
    async with _scrape_slot:
        scraped = await scraper.scrape(ref, max_pages=max_pages, emit=emit)
    run_dir = io.new_run_dir(ref.item_id, now=datetime.now())
    io.write_product(run_dir, scraped.product)
    io.write_raw_reviews(run_dir, scraped.reviews)
    io.write_run_meta(
        run_dir,
        io.RunMeta(
            scraped_at=datetime.now(UTC), reviews_scraped=len(scraped.reviews), partial=scraped.partial
        ),
    )
    return run_dir


def stage_clean(run_dir: Path, *, emit: Emit = no_emit) -> pd.DataFrame:
    emit("stage", {"stage": "cleaning", "message": "Cleaning review text and removing duplicates"})
    raw = io.read_raw_reviews(run_dir)
    clean = clean_reviews(raw)
    io.write_clean_reviews(run_dir, clean)
    emit("cleaned", {"clean": len(clean), "dropped": len(raw) - len(clean)})
    return clean


async def stage_features(run_dir: Path, *, emit: Emit = no_emit) -> FeatureGenerationOutcome | None:
    inputs = reviews_to_jev_inputs(io.read_clean_reviews(run_dir))
    if not inputs:
        return None
    emit("stage", {"stage": "categorizing", "message": "Working out which features matter for this product"})
    outcome = await generate_features(io.read_product(run_dir), inputs)
    io.write_features(run_dir, outcome)
    emit(
        "categories",
        {
            "product_category": outcome.feature_set.product_category,
            "used_fallback": outcome.used_fallback,
            "features": [f.model_dump() for f in outcome.feature_set.features],
        },
    )
    return outcome


def _scored_event_emitter(emit: Emit, total: int) -> ReviewCallback:
    max_level = load_rubric().max_level
    done = 0

    def on_review(review: JEVReviewInput, row: dict[str, Any], categories: list[dict[str, Any]]) -> None:
        nonlocal done
        done += 1
        scored = not (isinstance(row["score"], float) and math.isnan(row["score"]))
        emit(
            "scored",
            {
                "done": done,
                "total": total,
                "review_id": review.review_id,
                "rating": review.star_rating,
                "text": review.text[:_SCORED_TEXT_CHARS],
                "score": row["score"] if scored else None,
                "level": int(row["level"]) if scored else None,
                "sentiment": level_sentiment(int(row["level"]), max_level) if scored else None,
                "features": [
                    {"name": c["feature"], "score": c["score"]} for c in categories if c["mentioned"]
                ],
            },
        )

    return on_review


async def stage_jev(run_dir: Path, *, limit: int | None = None, emit: Emit = no_emit) -> JEVStats | None:
    inputs = reviews_to_jev_inputs(io.read_clean_reviews(run_dir))[:limit]
    if not inputs:
        io.write_scores(run_dir, pd.DataFrame(columns=io.SCORE_COLUMNS))
        io.write_category_scores(run_dir, pd.DataFrame(columns=io.CATEGORY_SCORE_COLUMNS))
        return None
    rubric = load_rubric()
    emit("scale", {"rubric_version": rubric.version, "max_level": rubric.max_level, "labels": rubric.labels})
    emit("stage", {"stage": "scoring", "message": f"Reading {len(inputs)} reviews"})
    result = await run_jev(
        inputs,
        io.read_features(run_dir).feature_set.features,
        on_review=_scored_event_emitter(emit, len(inputs)),
    )
    io.write_scores(run_dir, result.scores)
    io.write_category_scores(run_dir, result.category_scores)
    return result.stats


def stage_aggregate(
    run_dir: Path, *, jev_stats: JEVStats | None = None, emit: Emit = no_emit
) -> AnalysisResult:
    emit("stage", {"stage": "compiling", "message": "Compiling trends, features and terms"})
    meta = io.read_run_meta(run_dir)
    scores = io.read_scores(run_dir)
    # Only reviews sent to Jev (all of them, unless stage_jev ran with --limit).
    clean = io.read_clean_reviews(run_dir)
    analysed = clean[clean["review_id"].isin(scores["review_id"])].reset_index(drop=True)
    features = io.read_features(run_dir) if (run_dir / "features.json").exists() else None
    scored = scores["score"].notna()

    result = build_analysis_result(
        product=io.read_product(run_dir),
        features=features,
        clean=analysed,
        scores=scores,
        category_scores=io.read_category_scores(run_dir),
        rubric=load_rubric(),
        run_id=run_dir.name,
        scraped_at=meta.scraped_at,
        generated_at=datetime.now(UTC),
        stats=RunStats(
            reviews_scraped=meta.reviews_scraped,
            reviews_analysed=len(analysed),
            reviews_scored=int(scored.sum()),
            unprocessed_reviews=int((~scored).sum()),
            partial=meta.partial,
            used_fallback_features=bool(features and features.used_fallback),
            jev_model=str(scores.loc[scored, "model"].iloc[0]) if scored.any() else None,
            input_tokens=jev_stats.input_tokens if jev_stats else None,
        ),
    )
    io.write_analysis(run_dir, result)
    return result


async def analyse_run(run_dir: Path, *, emit: Emit = no_emit) -> AnalysisResult:
    stage_clean(run_dir, emit=emit)
    await stage_features(run_dir, emit=emit)
    jev_stats = await stage_jev(run_dir, emit=emit)
    return stage_aggregate(run_dir, jev_stats=jev_stats, emit=emit)


async def run_analysis(product_url: str, *, emit: Emit = no_emit) -> AnalysisResult:
    return await analyse_run(await scrape_to_run(product_url, emit=emit), emit=emit)
