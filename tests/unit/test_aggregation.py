"""Aggregation math against hand-computed numbers.

Five reviews scored against the 5-level rubric (0 Hostile .. 4 Enthusiastic),
as of 2026-10-16:

    id  stars  date        score  level  sentiment  text
    r1  5      2026-10-14  3.2    3      positive   good packaging
    r2  4      2026-10-01  3.9    4      positive   good taste
    r3  2      2026-08-20  1.1    1      negative   bad packaging
    r4  3      2026-06-10  2.0    2      neutral    packaging okay
    r5  1      (none)      -      -      unscored   fraud seller
"""

import math
from datetime import UTC, datetime

import pandas as pd
import pytest

from app.llm.jev.client import load_rubric
from app.models.api import RunStats, SentimentCounts
from app.models.domain import RawProduct
from app.models.jev import Feature, FeatureGenerationOutcome, FeatureSet
from app.pipeline import aggregation as agg

AS_OF = datetime(2026, 10, 16, tzinfo=UTC)


def _clean() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "review_id": ["r1", "r2", "r3", "r4", "r5"],
            "rating": pd.array([5, 4, 2, 3, 1], dtype="Int64"),
            "text": ["good packaging", "good taste", "bad packaging", "packaging okay", "fraud seller"],
            "raw_text": ["good packaging", "good taste", "bad packaging", "packaging okay", "fraud seller"],
            "review_date": pd.to_datetime(
                ["2026-10-14", "2026-10-01", "2026-08-20", "2026-06-10", None], utc=True
            ),
            "sku_info": [None] * 5,
            "helpful_count": pd.array([0, 3, 1, 0, 0], dtype="Int64"),
        }
    )


def _scores() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "review_id": ["r1", "r2", "r3", "r4", "r5"],
            "score": [3.2, 3.9, 1.1, 2.0, math.nan],
            "level": pd.array([3, 4, 1, 2, None], dtype="Int64"),
            "confidence": [0.9, 0.95, 0.8, 0.7, math.nan],
            "model": ["jev-1.14.0"] * 5,
            "rubric_version": ["jev_sentiment_v1"] * 5,
        }
    )


def _reviews() -> pd.DataFrame:
    return agg.with_scores(_clean(), _scores(), max_level=4)


@pytest.mark.parametrize(
    ("level", "sentiment"),
    [(0, "negative"), (1, "negative"), (2, "neutral"), (3, "positive"), (4, "positive")],
)
def test_level_sentiment(level: int, sentiment: str) -> None:
    assert agg.level_sentiment(level, 4) == sentiment


def test_with_scores_leaves_unscored_reviews_without_sentiment() -> None:
    reviews = _reviews()
    assert reviews["sentiment"].tolist() == ["positive", "positive", "negative", "neutral", None]


def test_recency_windows() -> None:
    windows = {w.days: w for w in agg.recency_windows(_reviews(), AS_OF)}

    assert (windows[7].review_count, windows[7].avg_score, windows[7].net_sentiment) == (1, 3.2, 100.0)
    assert (windows[30].review_count, windows[30].avg_score, windows[30].avg_rating) == (2, 3.55, 4.5)
    assert (windows[90].avg_score, windows[90].net_sentiment, windows[90].avg_rating) == (2.73, 33.3, 3.67)
    all_time = windows[None]
    assert (all_time.review_count, all_time.scored_count) == (5, 4)
    assert (all_time.avg_score, all_time.net_sentiment, all_time.avg_rating) == (2.55, 25.0, 3.0)


def test_monthly_trend_keeps_empty_months() -> None:
    trend = agg.monthly_trend(_reviews())

    assert [p.month for p in trend] == ["2026-06", "2026-07", "2026-08", "2026-09", "2026-10"]
    assert [p.review_count for p in trend] == [1, 0, 1, 0, 2]
    assert [p.avg_score for p in trend] == [2.0, None, 1.1, None, 3.55]
    assert [p.net_sentiment for p in trend] == [0.0, None, -100.0, None, 100.0]
    assert trend[-1].sentiment == SentimentCounts(positive=2, neutral=0, negative=0)


def test_term_frequencies() -> None:
    terms = agg.term_frequencies(_reviews())

    assert [(t.term, t.review_count) for t in terms] == [("packaging", 3), ("good", 2)]
    packaging, good = terms
    assert (packaging.share, packaging.avg_rating, packaging.avg_score, packaging.net_sentiment) == (
        0.6,
        3.33,
        2.1,
        0.0,
    )
    assert (good.share, good.avg_rating, good.avg_score, good.net_sentiment) == (0.4, 4.5, 3.55, 100.0)


@pytest.mark.parametrize(
    ("text", "tokens"),
    [
        ("Fluoride Content:wow Freshness: Cleaning:", {"wow"}),
        ("Freshness: wow Cleaning: bohat acha hai, packaging achi", {"wow", "acha", "packaging", "achi"}),
        ("taste: bad, sab ko leni chahiye", {"taste", "bad"}),
        ("بہت اچھا ہے", {"اچھا"}),
    ],
)
def test_tokenize_drops_template_labels_and_filler(text: str, tokens: set[str]) -> None:
    assert agg.tokenize(text) == tokens


def test_build_analysis_result() -> None:
    product = RawProduct(
        item_id="1",
        url="u",
        product_name="Sensodyne",
        category_breadcrumb="Health & Beauty > Oral Care > Toothpaste",
        total_rating_count=169,
    )
    features = FeatureGenerationOutcome(
        feature_set=FeatureSet(
            product_category="Toothpaste",
            features=[Feature(name="Taste", description="t"), Feature(name="Overall", description="o")],
        )
    )
    stats = RunStats(
        reviews_scraped=5,
        reviews_analysed=5,
        reviews_scored=4,
        unprocessed_reviews=1,
        partial=False,
        used_fallback_features=False,
        jev_model="jev-1.14.0",
        input_tokens=1600,
    )

    result = agg.build_analysis_result(
        product=product,
        features=features,
        clean=_clean(),
        scores=_scores(),
        rubric=load_rubric(),
        run_id="1_x",
        scraped_at=AS_OF,
        generated_at=AS_OF,
        stats=stats,
    )

    assert result.sentiment == SentimentCounts(positive=2, neutral=1, negative=1)
    assert (result.avg_score, result.net_sentiment) == (2.55, 25.0)
    assert [(lv.label, lv.count) for lv in result.levels] == [
        ("Hostile", 0),
        ("Negative", 1),
        ("Neutral", 1),
        ("Positive", 1),
        ("Enthusiastic", 1),
    ]
    assert result.scale.max_level == 4
    assert result.product.product_category == "Toothpaste"
    assert [c.name for c in result.categories] == ["Taste", "Overall"]
    assert result.rating_distribution.one_star == 1
    unscored = result.reviews[-1]
    assert (unscored.review_id, unscored.score, unscored.level, unscored.sentiment) == (
        "r5",
        None,
        None,
        None,
    )
    assert result.reviews[1].helpful_count == 3
