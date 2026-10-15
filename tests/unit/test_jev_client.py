from typing import Any

import httpx
import pytest

from app.llm.jev.client import JEVAuthError, JEVClient, JEVRequestError, load_rubric
from app.models.jev import Feature
from tests.jev_fakes import answers, fake_jev_client

CATEGORIES = [
    Feature(name="Taste", description="flavour {mint}"),
    Feature(name="Packaging", description="box"),
]


async def test_sends_one_request_with_overall_and_per_category_questions() -> None:
    requests: list[dict[str, Any]] = []
    client = fake_jev_client(answers(lambda _: 3.82, {"x": {0: (0.97, 1.01)}}), requests)
    rubric = load_rubric()

    result = await client.evaluate("x", rubric, CATEGORIES)

    [body] = requests
    assert body["model"] == "jev-1.14.0"
    assert body["state"] == "x"
    questions = body["questions"]
    assert list(questions) == ["sentiment", "c0_mentioned", "c0_score", "c1_mentioned", "c1_score"]
    assert questions["sentiment"] == {
        "type": "score",
        "instructions": rubric.instructions,
        "criteria": rubric.criteria,
    }
    assert questions["c0_mentioned"] == {
        "type": "noul",
        "instructions": "Does this review say anything about Taste (flavour {mint})?",
    }
    assert questions["c1_score"]["type"] == "score"
    assert questions["c1_score"]["criteria"] == rubric.criteria
    assert "Packaging" in questions["c1_score"]["instructions"]

    assert (result.score, result.model, result.input_tokens) == (3.82, "jev-1.14.0", 400)
    taste, packaging = result.categories
    assert (taste.name, taste.mention_probability, taste.score) == ("Taste", 0.97, 1.01)
    assert (packaging.name, packaging.mention_probability) == ("Packaging", 0.02)


async def test_no_categories_sends_only_the_sentiment_question() -> None:
    requests: list[dict[str, Any]] = []
    result = await fake_jev_client(answers(lambda _: 2.0), requests).evaluate("x", load_rubric(), [])
    assert list(requests[0]["questions"]) == ["sentiment"]
    assert result.categories == []


def test_rubric() -> None:
    rubric = load_rubric()
    assert rubric.version == "jev_sentiment_v3"
    assert rubric.labels == ["Negative", "Neutral", "Positive"]
    assert rubric.max_level == 2
    assert "kaha kuch bheja kuch" in rubric.instructions


@pytest.mark.parametrize(
    ("status", "error"), [(401, JEVAuthError), (403, JEVAuthError), (500, JEVRequestError)]
)
async def test_http_errors(status: int, error: type[Exception]) -> None:
    client = fake_jev_client(answers(lambda _: status))
    with pytest.raises(error):
        await client.evaluate("x", load_rubric(), CATEGORIES)


@pytest.mark.parametrize(
    "answers_json",
    [
        {},  # nothing answered
        {"sentiment": {"type": "noul", "noul": 0.5}},  # wrong answer type
    ],
)
async def test_unexpected_body_is_a_request_error(answers_json: dict[str, Any]) -> None:
    body = {"model": "jev-1.14.0", "answers": answers_json}
    http = httpx.AsyncClient(
        base_url="https://jev.test/v1",
        transport=httpx.MockTransport(lambda _: httpx.Response(200, json=body)),
    )
    with pytest.raises(JEVRequestError):
        await JEVClient(http, model="jev-1.14.0").evaluate("x", load_rubric(), [])


async def test_connection_error_is_a_request_error() -> None:
    def boom(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("down", request=request)

    http = httpx.AsyncClient(base_url="https://jev.test/v1", transport=httpx.MockTransport(boom))
    with pytest.raises(JEVRequestError):
        await JEVClient(http, model="jev-1.14.0").evaluate("x", load_rubric(), [])
