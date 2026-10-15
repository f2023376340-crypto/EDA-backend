"""A Jev System One API stand-in on httpx.MockTransport, shared by tests."""

import json
import re
from collections.abc import Callable, Mapping
from typing import Any

import httpx

from app.llm.jev.client import JEVClient

# (state, question_id, question) -> score / noul probability, or an int >= 100 = HTTP status to fail with.
AnswerFor = Callable[[str, str, dict[str, Any]], float | int]
_CATEGORY_QID = re.compile(r"^c(\d+)_(mentioned|score)$")


def _score_answer(score: float, levels: int) -> dict[str, Any]:
    lower = min(int(score), levels - 1)
    probabilities = {str(i): 0.0 for i in range(levels)}
    probabilities[str(lower)] = round(1 - (score - lower), 4)
    if lower + 1 < levels:
        probabilities[str(lower + 1)] = round(score - lower, 4)
    return {
        "type": "score",
        "score": score,
        "confidence": 0.9,
        "legend": {str(i): f"level {i}" for i in range(levels)},
        "probabilities": probabilities,
    }


def answers(
    overall: Callable[[str], float | int],
    categories: Mapping[str, Mapping[int, tuple[float, float]]] | None = None,
) -> AnswerFor:
    """`overall(state)` answers the sentiment question; `categories[state][i]` is
    (mention probability, score) for category i. Unlisted categories are not mentioned.
    """

    def answer_for(state: str, question_id: str, _question: dict[str, Any]) -> float | int:
        match = _CATEGORY_QID.match(question_id)
        if match is None:
            return overall(state)
        probability, score = (categories or {}).get(state, {}).get(int(match.group(1)), (0.02, 2.0))
        return probability if match.group(2) == "mentioned" else score

    return answer_for


def fake_jev_client(answer_for: AnswerFor, requests: list[dict[str, Any]] | None = None) -> JEVClient:
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        if requests is not None:
            requests.append(body)
        state = str(body["state"])
        out: dict[str, Any] = {}
        for qid, question in body["questions"].items():
            value = answer_for(state, qid, question)
            if isinstance(value, int) and not isinstance(value, bool) and value >= 100:
                return httpx.Response(value, json={"detail": {"message": "fail"}})
            if question["type"] == "noul":
                out[qid] = {"type": "noul", "noul": float(value)}
            else:
                out[qid] = _score_answer(float(value), len(question["criteria"]))
        return httpx.Response(
            200,
            json={"model": "jev-1.14.0", "answers": out, "usage": {"input_tokens": 400, "output_tokens": 17}},
        )

    http = httpx.AsyncClient(base_url="https://jev.test/v1", transport=httpx.MockTransport(handler))
    return JEVClient(http, model="jev-1.14.0")
