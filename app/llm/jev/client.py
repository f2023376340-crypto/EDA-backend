"""TypeSafe Jev client (System One API). Jev is a decision model, not a
generative LLM: each call sends one review as `state` plus a map of typed
questions, and Jev answers all of them in one pass. There is no
prompt-and-parse step; the response is typed JSON validated below.

Per review we ask:
- `sentiment`: Score on the rubric's levels (0-based: the 3-level rubric scores 0.0-2.0)
- per generated category i: `c{i}_mentioned` (Noul: probability the review talks
  about it) and `c{i}_score` (Score for that category on the same levels)

API: POST {JEV_BASE_URL}/systemone, Bearer auth.
"""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from functools import lru_cache
from pathlib import Path
from typing import Annotated, Any, Literal

import httpx
from pydantic import BaseModel, Field, ValidationError

from app.core.config import settings
from app.core.errors import LLMUnavailableError
from app.models.jev import Feature

RUBRIC_PATH = Path(__file__).parents[1] / "prompts" / "jev_sentiment_v3.json"
_TIMEOUT_S = 30.0


def mention_question_id(index: int) -> str:
    return f"c{index}_mentioned"


def category_score_question_id(index: int) -> str:
    return f"c{index}_score"


class ScoreRubric(BaseModel):
    version: str
    question_id: str
    instructions: str
    category_mention_instructions: str  # placeholders: {name}, {description}
    category_score_instructions: str  # placeholder: {name}
    criteria: list[str] = Field(min_length=2, max_length=10)

    @property
    def max_level(self) -> int:
        return len(self.criteria) - 1

    @property
    def labels(self) -> list[str]:
        """Short names: the text before the first period ("Negative", ...)."""
        return [c.split(".", 1)[0].strip() for c in self.criteria]

    def questions(self, categories: list[Feature]) -> dict[str, dict[str, Any]]:
        def fill(template: str, category: Feature) -> str:
            # str.replace, not str.format: generated names/descriptions may contain braces.
            return template.replace("{name}", category.name).replace("{description}", category.description)

        questions: dict[str, dict[str, Any]] = {
            self.question_id: {"type": "score", "instructions": self.instructions, "criteria": self.criteria}
        }
        for i, category in enumerate(categories):
            questions[mention_question_id(i)] = {
                "type": "noul",
                "instructions": fill(self.category_mention_instructions, category),
            }
            questions[category_score_question_id(i)] = {
                "type": "score",
                "instructions": fill(self.category_score_instructions, category),
                "criteria": self.criteria,
            }
        return questions


class ScoreAnswer(BaseModel):
    type: Literal["score"]
    score: float
    confidence: float
    probabilities: dict[str, float]


class NoulAnswer(BaseModel):
    type: Literal["noul"]
    noul: float  # probability that the answer is "yes"


Answer = Annotated[ScoreAnswer | NoulAnswer, Field(discriminator="type")]


class Usage(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0


class SystemOneResponse(BaseModel):
    model: str
    answers: dict[str, Answer]
    usage: Usage = Usage()


class JEVRequestError(Exception):
    def __init__(self, message: str, *, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


class JEVAuthError(JEVRequestError):
    pass


class CategoryEvaluation(BaseModel):
    name: str
    mention_probability: float
    score: float
    confidence: float


class ReviewEvaluation(BaseModel):
    score: float
    confidence: float
    model: str
    input_tokens: int
    categories: list[CategoryEvaluation]


@lru_cache(maxsize=1)
def load_rubric() -> ScoreRubric:
    return ScoreRubric.model_validate_json(RUBRIC_PATH.read_text(encoding="utf-8"))


def _answer[T: (ScoreAnswer, NoulAnswer)](
    answers: dict[str, ScoreAnswer | NoulAnswer], key: str, kind: type[T]
) -> T:
    answer = answers[key]
    if not isinstance(answer, kind):
        raise TypeError(f"{key}: expected {kind.__name__}, got {type(answer).__name__}")
    return answer


class JEVClient:
    def __init__(self, http: httpx.AsyncClient, *, model: str) -> None:
        self._http = http
        self.model = model

    async def evaluate(self, state: str, rubric: ScoreRubric, categories: list[Feature]) -> ReviewEvaluation:
        body = {"model": self.model, "state": state, "questions": rubric.questions(categories)}
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
            overall = _answer(parsed.answers, rubric.question_id, ScoreAnswer)
            evaluations = []
            for i, category in enumerate(categories):
                mention = _answer(parsed.answers, mention_question_id(i), NoulAnswer)
                score = _answer(parsed.answers, category_score_question_id(i), ScoreAnswer)
                evaluations.append(
                    CategoryEvaluation(
                        name=category.name,
                        mention_probability=mention.noul,
                        score=score.score,
                        confidence=score.confidence,
                    )
                )
        except (ValidationError, KeyError, TypeError) as exc:
            raise JEVRequestError(f"Unexpected Jev response: {response.text[:300]}") from exc
        return ReviewEvaluation(
            score=overall.score,
            confidence=overall.confidence,
            model=parsed.model,
            input_tokens=parsed.usage.input_tokens,
            categories=evaluations,
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
