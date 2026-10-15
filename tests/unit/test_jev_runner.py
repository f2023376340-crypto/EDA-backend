import asyncio
import math
from typing import Any

import httpx
import pytest

from app.core.errors import LLMUnavailableError
from app.llm.jev.client import JEVClient, ReviewEvaluation, ScoreRubric
from app.llm.jev.runner import nearest_level, run_jev
from app.models.jev import Feature, JEVReviewInput
from tests.jev_fakes import answers, fake_jev_client

FEATURES = [
    Feature(name="Taste", description="t"),
    Feature(name="Packaging", description="p"),
    Feature(name="Overall", description="o"),
]


def _reviews(*texts: str) -> list[JEVReviewInput]:
    return [JEVReviewInput(review_id=f"r{i}", text=t, star_rating=5) for i, t in enumerate(texts)]


async def test_scores_reviews_and_categories_in_input_order() -> None:
    overall = {"bad taste": 0.2, "ok": 1.0, "great box": 1.8}
    categories = {"bad taste": {0: (0.97, 0.1)}, "great box": {1: (0.9, 1.9)}}
    result = await run_jev(
        _reviews("great box", "bad taste", "ok"),
        FEATURES,
        client=fake_jev_client(answers(lambda s: overall[s], categories)),
    )

    assert result.scores["review_id"].tolist() == ["r0", "r1", "r2"]
    assert result.scores["score"].tolist() == [1.8, 0.2, 1.0]
    assert result.scores["level"].tolist() == [2, 0, 1]  # 3-level rubric: 0 Negative .. 2 Positive
    assert set(result.scores["rubric_version"]) == {"jev_sentiment_v3"}
    assert result.stats.scored == 3
    assert result.stats.categories == 2  # "Overall" is never asked about

    cats = result.category_scores
    assert len(cats) == 6  # 3 reviews x 2 categories
    mentioned = cats[cats["mentioned"]]
    assert mentioned[["review_id", "feature", "score", "level"]].values.tolist() == [
        ["r0", "Packaging", 1.9, 2],
        ["r1", "Taste", 0.1, 0],
    ]


async def test_unmentioned_category_scores_are_not_stored() -> None:
    result = await run_jev(_reviews("ok"), FEATURES, client=fake_jev_client(answers(lambda _: 2.0)))
    row = result.category_scores.iloc[0]
    assert not row["mentioned"]
    assert row["mention_probability"] == 0.02
    assert math.isnan(row["score"])


async def test_failed_request_leaves_review_unscored_without_retrying() -> None:
    calls: list[dict[str, Any]] = []
    client = fake_jev_client(answers(lambda s: 500 if s == "boom" else 3.0), calls)

    result = await run_jev(_reviews("fine", "boom"), FEATURES, client=client)

    assert math.isnan(result.scores.set_index("review_id").loc["r1", "score"])
    assert set(result.category_scores["review_id"]) == {"r0"}
    assert result.unprocessed_review_ids == ["r1"]
    assert len(calls) == 2  # one request per review, no retry


async def test_every_request_failing_raises_unavailable() -> None:
    with pytest.raises(LLMUnavailableError):
        await run_jev(_reviews("a", "b"), FEATURES, client=fake_jev_client(answers(lambda _: 503)))


async def test_rejected_key_raises_unavailable() -> None:
    client = fake_jev_client(answers(lambda s: 401 if s == "b" else 3.0))
    with pytest.raises(LLMUnavailableError, match="API key"):
        await run_jev(_reviews("a", "b"), FEATURES, client=client)


async def test_concurrency_is_bounded() -> None:
    in_flight = 0
    peak = 0

    class SlowClient(JEVClient):
        async def evaluate(
            self, state: str, rubric: ScoreRubric, categories: list[Feature]
        ) -> ReviewEvaluation:
            nonlocal in_flight, peak
            in_flight += 1
            peak = max(peak, in_flight)
            await asyncio.sleep(0.01)
            in_flight -= 1
            return ReviewEvaluation(
                score=3.0, confidence=1.0, model="jev-1.14.0", input_tokens=1, categories=[]
            )

    client = SlowClient(httpx.AsyncClient(), model="jev-1.14.0")
    await run_jev(_reviews(*[f"t{i}" for i in range(12)]), FEATURES, client=client, max_concurrency=3)
    assert peak == 3


@pytest.mark.parametrize(
    ("score", "max_level", "level"),
    [
        (0.0, 2, 0),
        (0.49, 2, 0),
        (0.5, 2, 1),
        (1.5, 2, 2),
        (2.0, 2, 2),
        (2.7, 2, 2),
        (2.5, 4, 3),
        (3.99, 4, 4),
    ],
)
def test_nearest_level_rounds_half_up(score: float, max_level: int, level: int) -> None:
    assert nearest_level(score, max_level) == level
