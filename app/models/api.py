"""Request/response models for app/api/v1. The authoritative URL check
(app/core/url_validator) runs explicitly in the orchestrator, not as a
pydantic field validator here — that keeps INVALID_URL errors flowing
through the AppError handler with the §6.1 body shape instead of FastAPI's
default RequestValidationError shape.

AnalysisResult is the contract the frontend mirrors (zod) — keep field names
stable. Metric definitions: docs/decisions/0001-jev-sentiment-metrics.md.
"""

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field

from app.models.domain import RatingDistribution
from app.models.jev import Sentiment


class AnalyzeRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    product_url: str = Field(min_length=1)


class _Out(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ProductInfo(_Out):
    item_id: str
    url: str
    name: str
    category_breadcrumb: str
    product_category: str
    price: float | None
    overall_rating: float | None
    rating_count: int
    rating_distribution: RatingDistribution | None


class SentimentCounts(_Out):
    positive: int
    neutral: int
    negative: int


class ScoreScale(_Out):
    """The rubric reviews were scored against; `score` values run 0..max_level."""

    rubric_version: str
    max_level: int
    labels: list[str]  # index = level, e.g. ["Negative", "Neutral", "Positive"]
    descriptions: list[str]


class LevelCount(_Out):
    level: int
    label: str
    count: int


class RecencyWindow(_Out):
    days: int | None  # None = all time
    review_count: int
    scored_count: int
    avg_score: float | None
    net_sentiment: float | None
    avg_rating: float | None


class TrendPoint(_Out):
    month: str  # YYYY-MM
    review_count: int
    scored_count: int
    avg_score: float | None
    avg_rating: float | None
    net_sentiment: float | None
    sentiment: SentimentCounts


class TermFrequency(_Out):
    term: str
    review_count: int
    share: float
    avg_rating: float
    avg_score: float | None
    net_sentiment: float | None


class FeatureQuote(_Out):
    review_id: str
    rating: int
    score: float  # the feature's score in this review
    sentiment: Sentiment
    text: str


class FeatureInsight(_Out):
    """Per generated category. "Overall" covers scored reviews that mention no
    other category, using their overall score.
    """

    name: str
    description: str
    mention_count: int
    mention_share: float  # 0-1, of scored reviews
    avg_score: float | None
    net_sentiment: float | None
    sentiment: SentimentCounts
    avg_rating: float | None  # stars of the reviews that mention it
    quotes: list[FeatureQuote]


class ReviewFeature(_Out):
    name: str
    score: float
    level: int
    sentiment: Sentiment


class ReviewOut(_Out):
    review_id: str
    rating: int
    text: str
    review_date: datetime | None
    helpful_count: int
    score: float | None  # None when Jev could not score this review
    level: int | None
    confidence: float | None
    sentiment: Sentiment | None
    features: list[ReviewFeature]  # categories this review talks about


class RunStats(_Out):
    reviews_scraped: int
    reviews_analysed: int
    reviews_scored: int
    unprocessed_reviews: int
    partial: bool
    used_fallback_features: bool
    jev_model: str | None
    input_tokens: int | None


class AnalysisResult(_Out):
    run_id: str
    generated_at: datetime
    scraped_at: datetime
    product: ProductInfo
    features: list[FeatureInsight]  # sorted by mention_count, "Overall" last
    scale: ScoreScale
    avg_score: float | None
    net_sentiment: float | None
    sentiment: SentimentCounts
    levels: list[LevelCount]
    recency: list[RecencyWindow]
    trend: list[TrendPoint]
    rating_distribution: RatingDistribution  # of the analysed reviews
    terms: list[TermFrequency]
    reviews: list[ReviewOut]
    stats: RunStats


class RunSummary(_Out):
    """One saved analysis, for the recent-analyses list."""

    run_id: str
    item_id: str
    product_name: str
    product_category: str
    url: str
    scraped_at: datetime
    reviews_scored: int
    avg_score: float | None
    max_level: int  # avg_score runs 0..max_level
    net_sentiment: float | None
