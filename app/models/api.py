"""Request/response models for app/api/v1. The authoritative URL check
(app/core/url_validator) runs explicitly in the orchestrator, not as a
pydantic field validator here — that keeps INVALID_URL errors flowing
through the AppError handler with the §6.1 body shape instead of FastAPI's
default RequestValidationError shape.
"""

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field

from app.models.jev import FeatureSet


class AnalyzeRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    product_url: str = Field(min_length=1)


class ReviewOut(BaseModel):
    model_config = ConfigDict(extra="forbid")

    review_id: str
    rating: int = Field(ge=1, le=5)
    text: str
    review_date: datetime | None = None


class AnalyzeResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    product_name: str
    overall_rating: float | None = Field(default=None, ge=0, le=5)
    generated_categories: FeatureSet
    used_fallback_categories: bool = False
    reviews_list: list[ReviewOut]
    reviews_scraped: int
    partial: bool = False
