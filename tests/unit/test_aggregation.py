"""Aggregation math against hand-computed numbers.

Five reviews scored against the 3-level rubric (0 Negative, 1 Neutral,
2 Positive), as of 2026-10-16:

    id  stars  date        score  level  sentiment  text
    r1  5      2026-10-14  1.6    2      positive   good packaging
    r2  4      2026-10-01  1.9    2      positive   good taste
    r3  2      2026-08-20  0.3    0      negative   bad packaging
    r4  3      2026-06-10  1.0    1      neutral    packaging okay
    r5  1      (none)      -      -      unscored   fraud seller

Category mentions (features Taste, Packaging, Overall):

    r1  Packaging 1.8 (level 2, positive)
    r2  Taste     1.9 (level 2, positive)
    r3  Packaging 0.2 (level 0, negative)
    r4  nothing mentioned (Packaging probability 0.3) -> counts toward Overall
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
MAX_LEVEL = 2


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
            "score": [1.6, 1.9, 0.3, 1.0, math.nan],
            "level": pd.array([2, 2, 0, 1, None], dtype="Int64"),
            "confidence": [0.9, 0.95, 0.8, 0.7, math.nan],
            "model": ["jev-1.14.0"] * 5,
            "rubric_version": ["jev_sentiment_v3"] * 5,
        }
    )


def _reviews() -> pd.DataFrame:
    return agg.with_scores(_clean(), _scores(), max_level=MAX_LEVEL)


FEATURES = [
    Feature(name="Taste", description="flavour"),
    Feature(name="Packaging", description="box"),
    Feature(name="Overall", description="general"),
]


def _category_scores() -> pd.DataFrame:
    # (review, feature) -> (probability, score, level); r5 failed, so it has no rows.
    mentioned = {
        ("r1", "Packaging"): (0.97, 1.8, 2),
        ("r2", "Taste"): (0.95, 1.9, 2),
        ("r3", "Packaging"): (0.9, 0.2, 0),
    }
    rows = []
    for review_id in ("r1", "r2", "r3", "r4"):
        for feature in ("Taste", "Packaging"):
            probability, score, level = mentioned.get(
                (review_id, feature), (0.3 if review_id == "r4" else 0.02, None, None)
            )
            rows.append(
                {
                    "review_id": review_id,
                    "feature": feature,
                    "mention_probability": probability,
                    "mentioned": score is not None,
                    "score": math.nan if score is None else score,
                    "level": level,
                    "confidence": math.nan if score is None else 0.9,
                }
            )
    df = pd.DataFrame(rows)
    df["level"] = df["level"].astype("Int64")
    return df


def test_feature_insights() -> None:
    reviews = _reviews()
    mentions = agg.category_mentions(_category_scores(), reviews, max_level=MAX_LEVEL)

    insights = agg.feature_insights(reviews, mentions, FEATURES)

    assert [i.name for i in insights] == ["Packaging", "Taste", "Overall"]
    packaging, taste, overall = insights
    assert (packaging.mention_count, packaging.mention_share, packaging.avg_score) == (2, 0.5, 1.0)
    assert (packaging.net_sentiment, packaging.avg_rating) == (0.0, 3.5)
    assert packaging.sentiment == SentimentCounts(positive=1, neutral=0, negative=1)
    assert [(q.review_id, q.score, q.sentiment) for q in packaging.quotes] == [
        ("r3", 0.2, "negative"),
        ("r1", 1.8, "positive"),
    ]
    assert (taste.mention_count, taste.mention_share, taste.avg_score, taste.net_sentiment) == (
        1,
        0.25,
        1.9,
        100.0,
    )
    # Overall: scored reviews that mention no category (r4), scored by their overall score.
    assert (overall.mention_count, overall.mention_share, overall.avg_score, overall.avg_rating) == (
        1,
        0.25,
        1.0,
        3.0,
    )
    assert [q.review_id for q in overall.quotes] == ["r4"]


def test_feature_with_no_mentions() -> None:
    reviews = _reviews()
    empty = agg.category_mentions(_category_scores().iloc[0:0], reviews, max_level=MAX_LEVEL)
    taste = agg.feature_insights(reviews, empty, FEATURES[:1])[0]
    assert (taste.mention_count, taste.avg_score, taste.net_sentiment, taste.quotes) == (0, None, None, [])


@pytest.mark.parametrize(
    ("level", "max_level", "sentiment"),
    [
        (0, 2, "negative"),
        (1, 2, "neutral"),
        (2, 2, "positive"),
        # Runs saved with the earlier 5-level rubric still aggregate correctly.
        (1, 4, "negative"),
        (2, 4, "neutral"),
        (3, 4, "positive"),
    ],
)
def test_level_sentiment(level: int, max_level: int, sentiment: str) -> None:
    assert agg.level_sentiment(level, max_level) == sentiment


def test_with_scores_leaves_unscored_reviews_without_sentiment() -> None:
    reviews = _reviews()
    assert reviews["sentiment"].tolist() == ["positive", "positive", "negative", "neutral", None]


def test_recency_windows() -> None:
    windows = {w.days: w for w in agg.recency_windows(_reviews(), AS_OF)}

    assert (windows[7].review_count, windows[7].avg_score, windows[7].net_sentiment) == (1, 1.6, 100.0)
    assert (windows[30].review_count, windows[30].avg_score, windows[30].avg_rating) == (2, 1.75, 4.5)
    assert (windows[90].avg_score, windows[90].net_sentiment, windows[90].avg_rating) == (1.27, 33.3, 3.67)
    all_time = windows[None]
    assert (all_time.review_count, all_time.scored_count) == (5, 4)
    assert (all_time.avg_score, all_time.net_sentiment, all_time.avg_rating) == (1.2, 25.0, 3.0)


def test_monthly_trend_keeps_empty_months() -> None:
    trend = agg.monthly_trend(_reviews())

    assert [p.month for p in trend] == ["2026-06", "2026-07", "2026-08", "2026-09", "2026-10"]
    assert [p.review_count for p in trend] == [1, 0, 1, 0, 2]
    assert [p.avg_score for p in trend] == [1.0, None, 0.3, None, 1.75]
    assert [p.net_sentiment for p in trend] == [0.0, None, -100.0, None, 100.0]
    assert trend[-1].sentiment == SentimentCounts(positive=2, neutral=0, negative=0)


def test_term_frequencies() -> None:
    terms = agg.term_frequencies(_reviews())

    assert [(t.term, t.review_count) for t in terms] == [("packaging", 3), ("good", 2)]
    packaging, good = terms
    assert (packaging.share, packaging.avg_rating, packaging.avg_score, packaging.net_sentiment) == (
        0.6,
        3.33,
        0.97,
        0.0,
    )
    assert (good.share, good.avg_rating, good.avg_score, good.net_sentiment) == (0.4, 4.5, 1.75, 100.0)


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
            features=FEATURES,
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
        category_scores=_category_scores(),
        rubric=load_rubric(),
        run_id="1_x",
        scraped_at=AS_OF,
        generated_at=AS_OF,
        stats=stats,
    )

    assert result.sentiment == SentimentCounts(positive=2, neutral=1, negative=1)
    assert (result.avg_score, result.net_sentiment) == (1.2, 25.0)
    assert [(lv.label, lv.count) for lv in result.levels] == [
        ("Negative", 1),
        ("Neutral", 1),
        ("Positive", 2),
    ]
    assert (result.scale.max_level, result.scale.rubric_version) == (2, "jev_sentiment_v3")
    assert result.product.product_category == "Toothpaste"
    assert [f.name for f in result.features] == ["Packaging", "Taste", "Overall"]
    assert result.rating_distribution.one_star == 1
    by_id = {r.review_id: r for r in result.reviews}
    assert [(f.name, f.score, f.level, f.sentiment) for f in by_id["r1"].features] == [
        ("Packaging", 1.8, 2, "positive")
    ]
    assert by_id["r4"].features == []
    unscored = result.reviews[-1]
    assert (unscored.review_id, unscored.score, unscored.level, unscored.sentiment) == (
        "r5",
        None,
        None,
        None,
    )
    assert result.reviews[1].helpful_count == 3
