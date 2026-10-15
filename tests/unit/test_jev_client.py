import httpx
import pytest

from app.llm.jev.client import JEVAuthError, JEVClient, JEVRequestError, load_rubric
from tests.jev_fakes import fake_jev_client


async def test_sends_documented_score_payload_and_parses_answer() -> None:
    requests: list[dict[str, object]] = []
    client = fake_jev_client(lambda _: 3.82, requests)
    rubric = load_rubric()

    result = await client.score("works great", rubric)

    assert result.score == 3.82
    assert result.model == "jev-1.14.0"
    assert result.input_tokens == 400
    [body] = requests
    assert body["model"] == "jev-1.14.0"
    assert body["state"] == "works great"
    assert body["questions"] == {
        "sentiment": {"type": "score", "instructions": rubric.instructions, "criteria": rubric.criteria}
    }


def test_rubric_is_the_five_level_scale() -> None:
    rubric = load_rubric()
    assert rubric.labels == ["Hostile", "Negative", "Neutral", "Positive", "Enthusiastic"]
    assert rubric.max_level == 4


@pytest.mark.parametrize(
    ("status", "error"), [(401, JEVAuthError), (403, JEVAuthError), (500, JEVRequestError)]
)
async def test_http_errors(status: int, error: type[Exception]) -> None:
    client = fake_jev_client(lambda _: status)
    with pytest.raises(error):
        await client.score("x", load_rubric())


async def test_unexpected_body_is_a_request_error() -> None:
    http = httpx.AsyncClient(
        base_url="https://jev.test/v1",
        transport=httpx.MockTransport(lambda _: httpx.Response(200, json={"model": "jev", "answers": {}})),
    )
    with pytest.raises(JEVRequestError):
        await JEVClient(http, model="jev-1.14.0").score("x", load_rubric())


async def test_connection_error_is_a_request_error() -> None:
    def boom(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("down", request=request)

    http = httpx.AsyncClient(base_url="https://jev.test/v1", transport=httpx.MockTransport(boom))
    with pytest.raises(JEVRequestError):
        await JEVClient(http, model="jev-1.14.0").score("x", load_rubric())
