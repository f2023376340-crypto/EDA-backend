"""Scraper output models — the boundary between app/scraper and everything
downstream. Field names mirror the raw_reviews.csv contract in CLAUDE.md §2
so a later pandas stage can load these rows without renaming columns.
"""

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class RatingDistribution(BaseModel):
    model_config = ConfigDict(extra="forbid")

    one_star: int = Field(ge=0)
    two_star: int = Field(ge=0)
    three_star: int = Field(ge=0)
    four_star: int = Field(ge=0)
    five_star: int = Field(ge=0)


class RawProduct(BaseModel):
    model_config = ConfigDict(extra="forbid")

    item_id: str
    sku_id: str | None = None
    url: str
    product_name: str
    category_breadcrumb: str
    price: float | None = None
    overall_rating: float | None = Field(default=None, ge=0, le=5)
    total_rating_count: int = Field(ge=0, default=0)
    rating_distribution: RatingDistribution | None = None


class RawReview(BaseModel):
    model_config = ConfigDict(extra="forbid")

    review_id: str
    rating: int = Field(ge=1, le=5)
    text: str
    raw_text: str
    review_date: datetime | None = None
    sku_info: str | None = None
    helpful_count: int = Field(ge=0, default=0)


class ScrapeResult(BaseModel):
    """What app/scraper hands back to the orchestrator for one product."""

    model_config = ConfigDict(extra="forbid")

    product: RawProduct
    reviews: list[RawReview]
    partial: bool = False
