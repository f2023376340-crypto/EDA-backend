"""Stage 6 (CLAUDE.md §2): jev_scores + jev_category_scores + clean_reviews ->
AnalysisResult. Pure
and deterministic: no model calls, no I/O, inputs untouched.

Every sentiment metric comes from Jev's score; star ratings are reported
alongside but never substituted for a missing score. Definitions are in
docs/decisions/0001-jev-sentiment-metrics.md.
"""

import re
from datetime import datetime, timedelta
from typing import Any, cast

import pandas as pd

from app.llm.jev.client import ScoreRubric
from app.models.api import (
    AnalysisResult,
    FeatureInsight,
    FeatureQuote,
    LevelCount,
    ProductInfo,
    RecencyWindow,
    ReviewFeature,
    ReviewOut,
    RunStats,
    ScoreScale,
    SentimentCounts,
    TermFrequency,
    TrendPoint,
)
from app.models.domain import RatingDistribution, RawProduct
from app.models.jev import OVERALL_FEATURE_NAME, Feature, FeatureGenerationOutcome, Sentiment

RECENCY_WINDOWS_DAYS = (7, 30, 90)
TOP_TERMS = 15
MIN_TERM_REVIEWS = 2
QUOTE_LIMIT = 3
QUOTE_MAX_CHARS = 280

# Daraz review-form labels ("Fluoride Content:", "Freshness:") appear even when
# left empty. They are Title Case, which keeps the customer's own words: in
# "Content:wow Freshness:" only the labels go, and "taste: bad" stays.
_TEMPLATE_LABEL_RE = re.compile(r"\b[A-Z][a-z]+(?:\s[A-Z][a-z]+)?\s*:")
_TOKEN_RE = re.compile(r"[^\W\d_]{3,}")
_STOPWORDS = frozenset(
    """
    the and for are but not you all any can had her was one our out has have this that with they from
    what when were will would there their them then than been being into very just also only some such
    its it's i'm im ive dont didnt doesnt cant wont isnt wasnt aren werent your yours mine myself him his
    she hers who whom which while where how why too more most much many other same each few both again
    further once here off over under above below about after before between through during because
    until should could does did doing get got getting make made use used using one two three
    product products item items daraz thanks thank shukriya
    hai hain hy hay hey hen hn tha thi thy ka ki ke kay ko ne se sy main mein mai bhi bhe aur
    yeh wo woh wah hi jo jab tak liye lye liya kya kiya kia kar karo kr krna karna raha rahi rhi rha
    sab sub apna apni aap app ap bohat bahut boht bht bhot buhat nahi nhi nai nahin
    chahiye chahye chahiay chahyee chahitye leni lena lene lia
    ہے ہیں کا کی کے کو نے سے میں بھی اور یہ وہ تھا تھی نہیں بہت ہی جو کیا
    """.split()
)


# -- helpers -----------------------------------------------------------------------


def _records(df: pd.DataFrame) -> list[dict[str, Any]]:
    return cast(list[dict[str, Any]], df.to_dict("records"))


def _mean(series: pd.Series, ndigits: int = 2) -> float | None:
    values = pd.to_numeric(series, errors="coerce").dropna()
    return None if values.empty else round(float(values.mean()), ndigits)


def net_sentiment(positive: int, negative: int, total: int) -> float | None:
    """(% positive - % negative) of scored reviews, -100..100; None if none scored."""
    return None if total == 0 else round(100 * (positive - negative) / total, 1)


def level_sentiment(level: int, max_level: int) -> Sentiment:
    """Levels below the rubric's midpoint are negative, above it positive. For
    the 3-level rubric (v3): 0 -> negative, 1 -> neutral, 2 -> positive; for the
    5-level one (v1/v2): 0-1 negative, 2 neutral, 3-4 positive.
    """
    midpoint = max_level / 2
    if level < midpoint:
        return "negative"
    return "neutral" if level == midpoint else "positive"


def _counts(reviews: pd.DataFrame) -> SentimentCounts:
    values = reviews["sentiment"].dropna().tolist()
    return SentimentCounts(
        positive=values.count("positive"), neutral=values.count("neutral"), negative=values.count("negative")
    )


def _net(counts: SentimentCounts) -> float | None:
    return net_sentiment(counts.positive, counts.negative, counts.positive + counts.neutral + counts.negative)


# -- joining ----------------------------------------------------------------------------


def with_scores(clean: pd.DataFrame, scores: pd.DataFrame, max_level: int) -> pd.DataFrame:
    """clean reviews + score/level/confidence/sentiment (NaN/None when unscored)."""
    out = clean.merge(scores[["review_id", "score", "level", "confidence"]], on="review_id", how="left")
    # Built as an object column so unscored reviews keep None (Series.map would turn it into NaN).
    out["sentiment"] = pd.Series(
        [None if pd.isna(lv) else level_sentiment(int(lv), max_level) for lv in out["level"]],
        index=out.index,
        dtype=object,
    )
    return out


# -- views ------------------------------------------------------------------------------


def level_counts(reviews: pd.DataFrame, rubric: ScoreRubric) -> list[LevelCount]:
    counts = reviews["level"].dropna().astype(int).value_counts()
    return [
        LevelCount(level=i, label=label, count=int(counts.get(i, 0))) for i, label in enumerate(rubric.labels)
    ]


def recency_windows(reviews: pd.DataFrame, as_of: datetime) -> list[RecencyWindow]:
    """Last 7/30/90 days relative to the scrape time, then all time."""
    windows: list[RecencyWindow] = []
    for days in (*RECENCY_WINDOWS_DAYS, None):
        subset = reviews if days is None else reviews[reviews["review_date"] >= as_of - timedelta(days=days)]
        windows.append(
            RecencyWindow(
                days=days,
                review_count=len(subset),
                scored_count=int(subset["score"].notna().sum()),
                avg_score=_mean(subset["score"]),
                net_sentiment=_net(_counts(subset)),
                avg_rating=_mean(subset["rating"]),
            )
        )
    return windows


def monthly_trend(reviews: pd.DataFrame) -> list[TrendPoint]:
    """One point per calendar month (UTC) from the first to the last dated
    review; empty months are kept with count 0 so gaps stay visible.
    """
    dated = reviews.dropna(subset=["review_date"])
    if dated.empty:
        return []
    months = dated["review_date"].dt.tz_convert(None).dt.to_period("M")
    points: list[TrendPoint] = []
    for period in pd.period_range(months.min(), months.max(), freq="M"):
        subset = dated[months == period]
        counts = _counts(subset)
        points.append(
            TrendPoint(
                month=str(period),
                review_count=len(subset),
                scored_count=int(subset["score"].notna().sum()),
                avg_score=_mean(subset["score"]),
                avg_rating=_mean(subset["rating"]),
                net_sentiment=_net(counts),
                sentiment=counts,
            )
        )
    return points


def tokenize(text: str) -> set[str]:
    stripped = _TEMPLATE_LABEL_RE.sub(" ", text)
    return {t for t in _TOKEN_RE.findall(stripped.casefold()) if t not in _STOPWORDS}


def term_frequencies(reviews: pd.DataFrame, top_n: int = TOP_TERMS) -> list[TermFrequency]:
    """Most common terms by number of reviews using them (once per review), with
    the avg stars, avg Jev score and net sentiment of those reviews.
    """
    total = len(reviews)
    if total == 0:
        return []
    exploded = (
        reviews.assign(term=reviews["text"].fillna("").map(lambda t: sorted(tokenize(t))))
        .explode("term")
        .dropna(subset=["term"])
    )
    if exploded.empty:
        return []
    terms: list[TermFrequency] = []
    for term, group in exploded.groupby("term"):
        if len(group) < MIN_TERM_REVIEWS:
            continue
        counts = _counts(group)
        terms.append(
            TermFrequency(
                term=str(term),
                review_count=len(group),
                share=round(len(group) / total, 4),
                avg_rating=cast(float, _mean(group["rating"])),
                avg_score=_mean(group["score"]),
                net_sentiment=_net(counts),
            )
        )
    terms.sort(key=lambda t: (-t.review_count, t.term))
    return terms[:top_n]


def rating_distribution(reviews: pd.DataFrame) -> RatingDistribution:
    counts = reviews["rating"].value_counts()
    return RatingDistribution(
        one_star=int(counts.get(1, 0)),
        two_star=int(counts.get(2, 0)),
        three_star=int(counts.get(3, 0)),
        four_star=int(counts.get(4, 0)),
        five_star=int(counts.get(5, 0)),
    )


def _optional_float(value: Any) -> float | None:
    return None if pd.isna(value) else float(value)


# -- features -------------------------------------------------------------------------


def category_mentions(category_scores: pd.DataFrame, reviews: pd.DataFrame, max_level: int) -> pd.DataFrame:
    """Mentioned (review, category) rows joined with the review's stars and
    text, plus a sentiment from the category's own level.
    """
    mentioned = category_scores[category_scores["mentioned"].fillna(False).astype(bool)]
    rows = mentioned.merge(reviews[["review_id", "rating", "text"]], on="review_id", how="inner")
    rows["sentiment"] = pd.Series(
        [level_sentiment(int(lv), max_level) for lv in rows["level"]], index=rows.index, dtype=object
    )
    return rows


def representative_quotes(rows: pd.DataFrame, limit: int = QUOTE_LIMIT) -> list[FeatureQuote]:
    """The lowest-scored mention (sharpest friction) first, then the highest,
    then the next lowest. `rows` needs review_id, rating, text, score, sentiment.
    """
    ordered = rows.sort_values(["score", "review_id"], kind="stable")
    if ordered.empty:
        return []
    picked = [ordered.index[0]]
    if len(ordered) > 1:
        picked.append(ordered.index[-1])
    picked.extend(i for i in ordered.index[1:-1] if len(picked) < limit)
    return [
        FeatureQuote(
            review_id=str(r["review_id"]),
            rating=int(r["rating"]),
            score=round(float(r["score"]), 2),
            sentiment=r["sentiment"],
            text=_shorten(str(r["text"])),
        )
        for r in _records(rows.loc[picked[:limit]])
    ]


def _shorten(text: str) -> str:
    return text if len(text) <= QUOTE_MAX_CHARS else text[: QUOTE_MAX_CHARS - 1].rstrip() + "…"


def _insight(name: str, description: str, rows: pd.DataFrame, scored_total: int) -> FeatureInsight:
    counts = _counts(rows)
    return FeatureInsight(
        name=name,
        description=description,
        mention_count=len(rows),
        mention_share=round(len(rows) / scored_total, 4) if scored_total else 0.0,
        avg_score=_mean(rows["score"]),
        net_sentiment=_net(counts),
        sentiment=counts,
        avg_rating=_mean(rows["rating"]),
        quotes=representative_quotes(rows),
    )


def feature_insights(
    reviews: pd.DataFrame, mentions: pd.DataFrame, features: list[Feature]
) -> list[FeatureInsight]:
    """One insight per generated category, by mention_count (ties keep the
    generated order). "Overall" comes last and covers scored reviews that mention
    no other category, using their overall score.
    """
    scored = reviews[reviews["score"].notna()]
    insights = [
        _insight(f.name, f.description, mentions[mentions["feature"] == f.name], len(scored))
        for f in features
        if f.name != OVERALL_FEATURE_NAME
    ]
    insights.sort(key=lambda i: -i.mention_count)
    overall = next((f for f in features if f.name == OVERALL_FEATURE_NAME), None)
    if overall is not None:
        generic = scored[~scored["review_id"].isin(mentions["review_id"])]
        insights.append(_insight(overall.name, overall.description, generic, len(scored)))
    return insights


# -- review feed ----------------------------------------------------------------------


def _review_feed(reviews: pd.DataFrame, mentions: pd.DataFrame) -> list[ReviewOut]:
    by_review: dict[str, list[ReviewFeature]] = {}
    for m in _records(mentions):
        by_review.setdefault(str(m["review_id"]), []).append(
            ReviewFeature(
                name=str(m["feature"]),
                score=round(float(m["score"]), 2),
                level=int(m["level"]),
                sentiment=m["sentiment"],
            )
        )
    return [
        ReviewOut(
            review_id=str(r["review_id"]),
            rating=int(r["rating"]),
            text=str(r["text"]),
            review_date=None if pd.isna(r["review_date"]) else r["review_date"].to_pydatetime(),
            helpful_count=0 if pd.isna(r["helpful_count"]) else int(r["helpful_count"]),
            score=_optional_float(r["score"]),
            level=None if pd.isna(r["level"]) else int(r["level"]),
            confidence=_optional_float(r["confidence"]),
            sentiment=r["sentiment"],
            features=by_review.get(str(r["review_id"]), []),
        )
        for r in _records(reviews)
    ]


# -- assembly ---------------------------------------------------------------------------


def build_analysis_result(
    *,
    product: RawProduct,
    features: FeatureGenerationOutcome | None,
    clean: pd.DataFrame,
    scores: pd.DataFrame,
    category_scores: pd.DataFrame,
    rubric: ScoreRubric,
    run_id: str,
    scraped_at: datetime,
    generated_at: datetime,
    stats: RunStats,
) -> AnalysisResult:
    reviews = with_scores(clean, scores, rubric.max_level)
    mentions = category_mentions(category_scores, reviews, rubric.max_level)
    counts = _counts(reviews)
    category = (
        features.feature_set.product_category
        if features
        else product.category_breadcrumb.split(">")[-1].strip() or "Product"
    )
    return AnalysisResult(
        run_id=run_id,
        generated_at=generated_at,
        scraped_at=scraped_at,
        product=ProductInfo(
            item_id=product.item_id,
            url=product.url,
            name=product.product_name,
            category_breadcrumb=product.category_breadcrumb,
            product_category=category,
            price=product.price,
            overall_rating=product.overall_rating,
            rating_count=product.total_rating_count,
            rating_distribution=product.rating_distribution,
        ),
        features=feature_insights(reviews, mentions, features.feature_set.features) if features else [],
        scale=ScoreScale(
            rubric_version=rubric.version,
            max_level=rubric.max_level,
            labels=rubric.labels,
            descriptions=rubric.criteria,
        ),
        avg_score=_mean(reviews["score"]),
        net_sentiment=_net(counts),
        sentiment=counts,
        levels=level_counts(reviews, rubric),
        recency=recency_windows(reviews, scraped_at),
        trend=monthly_trend(reviews),
        rating_distribution=rating_distribution(reviews),
        terms=term_frequencies(reviews),
        reviews=_review_feed(reviews, mentions),
        stats=stats,
    )
