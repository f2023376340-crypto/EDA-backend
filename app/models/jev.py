"""Models for feature generation (Groq) and the review input sent to Jev.
LLM output models forbid extra keys so hallucinated fields fail validation.
"""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

OVERALL_FEATURE_NAME = "Overall"

Sentiment = Literal["positive", "neutral", "negative"]


class Feature(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1, max_length=40)
    description: str


class FeatureSet(BaseModel):
    model_config = ConfigDict(extra="forbid")

    product_category: str
    features: list[Feature] = Field(min_length=1, max_length=9)


class FeatureGenerationOutcome(BaseModel):
    """Wraps the FeatureSet with whether the §6.4 generic fallback was used,
    so callers can surface that to the client instead of silently losing it.
    """

    model_config = ConfigDict(extra="forbid")

    feature_set: FeatureSet
    used_fallback: bool = False


class JEVReviewInput(BaseModel):
    review_id: str
    text: str
    star_rating: int = Field(ge=1, le=5)
