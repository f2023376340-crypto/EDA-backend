"""Stage 4: product-specific feature categories via Groq (§2, §5.5, §6.4).

One structured-output call, one repair attempt that feeds the validation
error back, then the generic fallback set. `Overall` is always appended
here in code — the prompt tells the model not to produce it, and any copy
it produces anyway is dropped.
"""

import time
from pathlib import Path

import structlog
from groq import APIConnectionError, AuthenticationError, RateLimitError
from langchain_core.exceptions import OutputParserException
from langchain_core.prompts import ChatPromptTemplate
from pydantic import ValidationError

from app.core.config import settings
from app.core.errors import LLMUnavailableError
from app.llm.groq_client import get_chat_model
from app.models.domain import RawProduct, RawReview
from app.models.jev import OVERALL_FEATURE_NAME, Feature, FeatureGenerationOutcome, FeatureSet

log = structlog.get_logger(__name__)

PROMPT_VERSION = "features_v1"
_PROMPT_PATH = Path(__file__).parent / "prompts" / f"{PROMPT_VERSION}.md"

_TEMPERATURE = 0.3
_SAMPLE_SIZE = 20
_SAMPLE_REVIEW_MAX_CHARS = 300
_MIN_FEATURES = 4
_MAX_FEATURES = 8
_CACHE_TTL_S = 24 * 60 * 60

_OVERALL = Feature(name=OVERALL_FEATURE_NAME, description="General impressions not tied to one feature.")

_FALLBACK_FEATURES = [
    Feature(name="Quality", description="Build and material quality of the product."),
    Feature(name="Value for Money", description="Whether the product is worth its price."),
    Feature(name="Delivery & Packaging", description="Delivery speed and packaging condition."),
    Feature(name="Seller Service", description="Seller communication and after-sales support."),
]

_cache: dict[str, tuple[float, FeatureGenerationOutcome]] = {}


class _InvalidFeatureSet(ValueError):
    pass


def _build_review_sample(reviews: list[RawReview]) -> str:
    lines = [f"- ({r.rating}/5) {r.text[:_SAMPLE_REVIEW_MAX_CHARS]}" for r in reviews if r.text.strip()][
        :_SAMPLE_SIZE
    ]
    return "\n".join(lines) if lines else "(no reviews available)"


def _finalize(raw: FeatureSet) -> FeatureSet:
    """Drop any model-produced Overall and duplicate names, enforce the
    4–8 range, then append our own Overall.
    """
    seen: set[str] = set()
    features: list[Feature] = []
    for feature in raw.features:
        key = feature.name.strip().casefold()
        if key == OVERALL_FEATURE_NAME.casefold() or key in seen:
            continue
        seen.add(key)
        features.append(Feature(name=feature.name.strip(), description=feature.description.strip()))

    if not _MIN_FEATURES <= len(features) <= _MAX_FEATURES:
        raise _InvalidFeatureSet(
            f"Expected {_MIN_FEATURES}-{_MAX_FEATURES} distinct non-Overall features, got {len(features)}."
        )
    return FeatureSet(product_category=raw.product_category.strip(), features=[*features, _OVERALL])


def _fallback(product: RawProduct) -> FeatureGenerationOutcome:
    category = product.category_breadcrumb.split(">")[-1].strip() or "General"
    return FeatureGenerationOutcome(
        feature_set=FeatureSet(product_category=category, features=[*_FALLBACK_FEATURES, _OVERALL]),
        used_fallback=True,
    )


async def generate_features(product: RawProduct, reviews: list[RawReview]) -> FeatureGenerationOutcome:
    cached = _cache.get(product.item_id)
    if cached and cached[0] > time.monotonic():
        return cached[1]

    llm = get_chat_model(settings.groq_model_features, _TEMPERATURE)
    structured = llm.with_structured_output(FeatureSet)
    template = _PROMPT_PATH.read_text(encoding="utf-8")
    variables = {
        "product_name": product.product_name,
        "category": product.category_breadcrumb or "unknown",
        "review_sample": _build_review_sample(reviews),
    }

    repair_note = ""
    started = time.perf_counter()
    for attempt in (1, 2):
        prompt = ChatPromptTemplate.from_messages([("human", template + repair_note)])
        try:
            raw = await (prompt | structured).ainvoke(variables)
            if not isinstance(raw, FeatureSet):
                raw = FeatureSet.model_validate(raw)
            feature_set = _finalize(raw)
        except (RateLimitError, APIConnectionError, AuthenticationError) as exc:
            raise LLMUnavailableError(
                "Groq is unavailable for feature generation.", details={"reason": type(exc).__name__}
            ) from exc
        except (OutputParserException, ValidationError, _InvalidFeatureSet) as exc:
            log.warning(
                "feature_generation_invalid",
                item_id=product.item_id,
                attempt=attempt,
                prompt_version=PROMPT_VERSION,
                error=str(exc)[:500],
            )
            # Braces would be read as template variables by ChatPromptTemplate.
            safe_error = str(exc)[:500].replace("{", "(").replace("}", ")")
            repair_note = (
                "\n\nYour previous answer was invalid. Fix it to satisfy every rule above. "
                f"Validation errors: {safe_error}"
            )
            continue

        outcome = FeatureGenerationOutcome(feature_set=feature_set)
        _cache[product.item_id] = (time.monotonic() + _CACHE_TTL_S, outcome)
        log.info(
            "feature_generation_ok",
            item_id=product.item_id,
            attempt=attempt,
            prompt_version=PROMPT_VERSION,
            feature_count=len(feature_set.features),
            latency_ms=round((time.perf_counter() - started) * 1000),
        )
        return outcome

    log.warning("feature_generation_fallback", item_id=product.item_id, prompt_version=PROMPT_VERSION)
    return _fallback(product)
