"""Composes pipeline stages. Currently: validate URL -> scrape -> generate
features. Cleaning, JEV and aggregation (§2 stages 3, 5, 6) slot in between
scrape and response once they are built.
"""

import asyncio

from app.core.url_validator import validate_daraz_url
from app.llm.feature_generator import generate_features
from app.models.api import AnalyzeResponse, ReviewOut
from app.scraper import client as scraper

# §5.4: at most one concurrent product scrape per process.
_scrape_slot = asyncio.Semaphore(1)


async def run_analysis(product_url: str) -> AnalyzeResponse:
    ref = validate_daraz_url(product_url)

    async with _scrape_slot:
        scraped = await scraper.scrape(ref)

    outcome = await generate_features(scraped.product, scraped.reviews)

    return AnalyzeResponse(
        product_name=scraped.product.product_name,
        overall_rating=scraped.product.overall_rating,
        generated_categories=outcome.feature_set,
        used_fallback_categories=outcome.used_fallback,
        reviews_list=[
            ReviewOut(review_id=r.review_id, rating=r.rating, text=r.text, review_date=r.review_date)
            for r in scraped.reviews
        ],
        reviews_scraped=len(scraped.reviews),
        partial=scraped.partial,
    )
