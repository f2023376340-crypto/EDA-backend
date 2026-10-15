"""Progress events for the streaming endpoint (GET /api/v1/analyze/stream).

Pipeline code reports progress through an `Emit` callback: synchronous,
non-blocking, and safe to call from any thread (the scraper runs on its own
Playwright thread). The endpoint supplies one that forwards to its SSE queue;
everything else gets `no_emit`.

Event names and payloads (the frontend mirrors these in src/lib/stream.ts):
    stage       {stage, message}         stage in STAGES
    product     {name, category_breadcrumb, price, overall_rating, rating_count, url}
    reviews     {collected}              after each page of reviews
    cleaned     {clean, dropped}
    categories  {product_category, used_fallback, features: [{name, description}]}
    scale       {rubric_version, max_level, labels}   before scoring; scores run 0..max_level
    scored      {done, total, review_id, rating, text, score, level, sentiment, features}
    done        AnalysisResult
    failed      {code, message}   ("error" is reserved by EventSource for connection errors)
"""

from collections.abc import Callable
from typing import Any, Literal

Stage = Literal["validating", "scraping", "cleaning", "categorizing", "scoring", "compiling"]
STAGES: tuple[Stage, ...] = ("validating", "scraping", "cleaning", "categorizing", "scoring", "compiling")

Emit = Callable[[str, dict[str, Any]], None]


def no_emit(_event: str, _data: dict[str, Any]) -> None:
    return None
