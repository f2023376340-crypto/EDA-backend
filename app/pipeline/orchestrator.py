"""Composes the pipeline (CLAUDE.md §2). Every stage reads its inputs from and
writes its outputs to the run folder via app/pipeline/io.py, so each one can
also be re-run on its own from scripts/.

    run_analysis: validate URL -> scrape -> analyse_run
    analyse_run:  stage_clean -> stage_features (Groq) -> stage_jev (Jev) -> stage_aggregate
"""

import asyncio
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from app.core.url_validator import validate_daraz_url
from app.llm.feature_generator import generate_features
from app.llm.jev.client import load_rubric
from app.llm.jev.runner import JEVStats, run_jev
from app.models.api import AnalysisResult, RunStats
from app.models.jev import FeatureGenerationOutcome
from app.pipeline import io
from app.pipeline.aggregation import build_analysis_result
from app.pipeline.cleaning import clean_reviews
from app.pipeline.jev_inputs import reviews_to_jev_inputs
from app.scraper import client as scraper

# §5.4: at most one concurrent product scrape per process.
_scrape_slot = asyncio.Semaphore(1)


async def scrape_to_run(product_url: str, *, max_pages: int | None = None) -> Path:
    ref = validate_daraz_url(product_url)
    async with _scrape_slot:
        scraped = await scraper.scrape(ref, max_pages=max_pages)
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


def stage_clean(run_dir: Path) -> pd.DataFrame:
    clean = clean_reviews(io.read_raw_reviews(run_dir))
    io.write_clean_reviews(run_dir, clean)
    return clean


async def stage_features(run_dir: Path) -> FeatureGenerationOutcome | None:
    inputs = reviews_to_jev_inputs(io.read_clean_reviews(run_dir))
    if not inputs:
        return None
    outcome = await generate_features(io.read_product(run_dir), inputs)
    io.write_features(run_dir, outcome)
    return outcome


async def stage_jev(run_dir: Path, *, limit: int | None = None) -> JEVStats | None:
    inputs = reviews_to_jev_inputs(io.read_clean_reviews(run_dir))[:limit]
    if not inputs:
        io.write_scores(run_dir, pd.DataFrame(columns=io.SCORE_COLUMNS))
        return None
    result = await run_jev(inputs)
    io.write_scores(run_dir, result.scores)
    return result.stats


def stage_aggregate(run_dir: Path, *, jev_stats: JEVStats | None = None) -> AnalysisResult:
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


async def analyse_run(run_dir: Path) -> AnalysisResult:
    stage_clean(run_dir)
    await stage_features(run_dir)
    jev_stats = await stage_jev(run_dir)
    return stage_aggregate(run_dir, jev_stats=jev_stats)


async def run_analysis(product_url: str) -> AnalysisResult:
    return await analyse_run(await scrape_to_run(product_url))
