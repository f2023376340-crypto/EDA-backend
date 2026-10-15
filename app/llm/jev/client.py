"""TypeSafe Jev client (System One API). Jev is a decision model, not a
generative LLM: each call sends a review as `state` plus one Score question,
and gets back a probability-weighted position on the rubric's levels. There
is no prompt-and-parse step; the response is typed JSON validated below.

API: POST {JEV_BASE_URL}/systemone, Bearer auth. Level indices are 0-based, so
a 5-level rubric scores 0.0-4.0.
"""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from functools import lru_cache
from pathlib import Path
from typing import Literal

import httpx
from pydantic import BaseModel, Field, ValidationError

from app.core.config import settings
from app.core.errors import LLMUnavailableError

RUBRIC_PATH = Path(__file__).parents[1] / "prompts" / "jev_sentiment_v1.json"
_TIMEOUT_S = 30.0


class ScoreRubric(BaseModel):
    version: str
    question_id: str
    instructions: str
    criteria: list[str] = Field(min_length=2, max_length=10)

    @property
    def max_level(self) -> int:
        return len(self.criteria) - 1

    @property
    def labels(self) -> list[str]:
        """Short names: the text before the first period ("Hostile", ...)."""
        return [c.split(".", 1)[0].strip() for c in self.criteria]


class ScoreAnswer(BaseModel):
    type: Literal["score"]
    score: float
    confidence: float
    probabilities: dict[str, float]


class Usage(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0


class SystemOneResponse(BaseModel):
    model: str
    answers: dict[str, ScoreAnswer]
    usage: Usage = Usage()


class JEVRequestError(Exception):
    def __init__(self, message: str, *, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


class JEVAuthError(JEVRequestError):
    pass


class ScoredReview(BaseModel):
    score: float
    confidence: float
    model: str
    input_tokens: int


@lru_cache(maxsize=1)
def load_rubric() -> ScoreRubric:
    return ScoreRubric.model_validate_json(RUBRIC_PATH.read_text(encoding="utf-8"))


class JEVClient:
    def __init__(self, http: httpx.AsyncClient, *, model: str) -> None:
        self._http = http
        self.model = model

    async def score(self, state: str, rubric: ScoreRubric) -> ScoredReview:
        body = {
            "model": self.model,
            "state": state,
            "questions": {
                rubric.question_id: {
                    "type": "score",
                    "instructions": rubric.instructions,
                    "criteria": rubric.criteria,
                }
            },
        }
        try:
            response = await self._http.post("/systemone", json=body)
        except httpx.HTTPError as exc:
            raise JEVRequestError(f"{type(exc).__name__}: {exc}") from exc

        if response.status_code in (401, 403):
            raise JEVAuthError("Jev rejected the API key.", status_code=response.status_code)
        if response.status_code != 200:
            raise JEVRequestError(response.text[:300], status_code=response.status_code)

        try:
            parsed = SystemOneResponse.model_validate_json(response.content)
            answer = parsed.answers[rubric.question_id]
        except (ValidationError, KeyError) as exc:
            raise JEVRequestError(f"Unexpected Jev response: {response.text[:300]}") from exc
        return ScoredReview(
            score=answer.score,
            confidence=answer.confidence,
            model=parsed.model,
            input_tokens=parsed.usage.input_tokens,
        )


@asynccontextmanager
async def jev_client() -> AsyncIterator[JEVClient]:
    if not settings.jev_api_key:
        raise LLMUnavailableError("JEV_API_KEY is not configured.")
    async with httpx.AsyncClient(
        base_url=settings.jev_base_url,
        headers={"Authorization": f"Bearer {settings.jev_api_key}"},
        timeout=_TIMEOUT_S,
    ) as http:
        yield JEVClient(http, model=settings.jev_model)
