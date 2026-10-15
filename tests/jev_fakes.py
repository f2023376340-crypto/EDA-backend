"""A Jev System One API stand-in on httpx.MockTransport, shared by tests."""

import json
from collections.abc import Callable

import httpx

from app.llm.jev.client import JEVClient

ScoreFor = Callable[[str], float | int]  # state text -> score, or an HTTP status to fail with


def score_response(score: float, *, question_id: str = "sentiment", levels: int = 5) -> dict[str, object]:
    lower = int(score)
    probabilities = {str(i): 0.0 for i in range(levels)}
    probabilities[str(lower)] = round(1 - (score - lower), 4)
    if lower + 1 < levels:
        probabilities[str(lower + 1)] = round(score - lower, 4)
    return {
        "model": "jev-1.14.0",
        "answers": {
            question_id: {
                "type": "score",
                "score": score,
                "confidence": 0.9,
                "legend": {str(i): f"level {i}" for i in range(levels)},
                "probabilities": probabilities,
            }
        },
        "usage": {"input_tokens": 400, "output_tokens": 17},
    }


def fake_jev_client(score_for: ScoreFor, requests: list[dict[str, object]] | None = None) -> JEVClient:
    """`score_for` returns a float score, or an int HTTP status (e.g. 500) to fail with."""

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        if requests is not None:
            requests.append(body)
        outcome = score_for(str(body["state"]))
        if isinstance(outcome, int) and not isinstance(outcome, bool) and outcome >= 100:
            return httpx.Response(outcome, json={"detail": {"message": "fail"}})
        return httpx.Response(200, json=score_response(float(outcome)))

    http = httpx.AsyncClient(base_url="https://jev.test/v1", transport=httpx.MockTransport(handler))
    return JEVClient(http, model="jev-1.14.0")
