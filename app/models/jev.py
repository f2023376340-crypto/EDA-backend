"""LLM-facing models shared by feature generation and (later) JEV batch
processing. Only Feature/FeatureSet are used this session — the JEV batch
models (JEVReviewInput, FeatureMention, ...) land with the batch-processing
work per CLAUDE.md §5.2.
"""

from pydantic import BaseModel, ConfigDict, Field

OVERALL_FEATURE_NAME = "Overall"


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
