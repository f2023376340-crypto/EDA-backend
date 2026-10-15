import asyncio
import math

import httpx
import pytest

from app.core.errors import LLMUnavailableError
from app.llm.jev.client import JEVClient, ScoredReview, ScoreRubric
from app.llm.jev.runner import nearest_level, run_jev
from app.models.jev import JEVReviewInput
from tests.jev_fakes import fake_jev_client


def _reviews(*texts: str) -> list[JEVReviewInput]:
    return [JEVReviewInput(review_id=f"r{i}", text=t, star_rating=5) for i, t in enumerate(texts)]


async def test_scores_every_review_in_input_order() -> None:
    scores = {"bad": 1.02, "ok": 2.0, "great": 3.6}
    result = await run_jev(_reviews("great", "bad", "ok"), client=fake_jev_client(lambda s: scores[s]))

    assert result.scores["review_id"].tolist() == ["r0", "r1", "r2"]
    assert result.scores["score"].tolist() == [3.6, 1.02, 2.0]
    assert result.scores["level"].tolist() == [4, 1, 2]
    assert set(result.scores["rubric_version"]) == {"jev_sentiment_v1"}
    assert result.unprocessed_review_ids == []
    assert result.stats.scored == 3
    assert result.stats.input_tokens == 1200


async def test_failed_request_leaves_review_unscored_without_retrying() -> None:
    calls: list[dict[str, object]] = []
    client = fake_jev_client(lambda s: 500 if s == "boom" else 3.0, calls)

    result = await run_jev(_reviews("fine", "boom"), client=client)

    row = result.scores.set_index("review_id").loc["r1"]
    assert math.isnan(row["score"])
    assert result.unprocessed_review_ids == ["r1"]
    assert len(calls) == 2  # one request per review, no retry


async def test_every_request_failing_raises_unavailable() -> None:
    with pytest.raises(LLMUnavailableError):
        await run_jev(_reviews("a", "b"), client=fake_jev_client(lambda _: 503))


async def test_rejected_key_raises_unavailable() -> None:
    with pytest.raises(LLMUnavailableError, match="API key"):
        await run_jev(_reviews("a", "b"), client=fake_jev_client(lambda s: 401 if s == "b" else 3.0))


async def test_concurrency_is_bounded() -> None:
    in_flight = 0
    peak = 0

    class SlowClient(JEVClient):
        async def score(self, state: str, rubric: ScoreRubric) -> ScoredReview:
            nonlocal in_flight, peak
            in_flight += 1
            peak = max(peak, in_flight)
            await asyncio.sleep(0.01)
            in_flight -= 1
            return ScoredReview(score=3.0, confidence=1.0, model="jev-1.14.0", input_tokens=1)

    client = SlowClient(httpx.AsyncClient(), model="jev-1.14.0")
    await run_jev(_reviews(*[f"t{i}" for i in range(12)]), client=client, max_concurrency=3)
    assert peak == 3


@pytest.mark.parametrize(("score", "level"), [(0.0, 0), (0.49, 0), (0.5, 1), (2.5, 3), (3.99, 4), (4.0, 4)])
def test_nearest_level_rounds_half_up(score: float, level: int) -> None:
    assert nearest_level(score, 4) == level
