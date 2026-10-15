"""Domain error taxonomy. All raised errors that should reach the client as a
structured JSON body subclass AppError; app/main.py registers the handler
that renders them per the §6.1 contract.
"""

from typing import Any, Literal

ErrorCode = Literal[
    "INVALID_URL",
    "PRODUCT_NOT_FOUND",
    "NO_REVIEWS",
    "SCRAPE_BLOCKED",
    "SCRAPE_TIMEOUT",
    "LLM_UNAVAILABLE",
    "LLM_OUTPUT_INVALID",
    "RUN_NOT_FOUND",
]

_STATUS_BY_CODE: dict[ErrorCode, int] = {
    "RUN_NOT_FOUND": 404,
    "INVALID_URL": 422,
    "PRODUCT_NOT_FOUND": 404,
    "NO_REVIEWS": 200,
    "SCRAPE_BLOCKED": 503,
    "SCRAPE_TIMEOUT": 504,
    "LLM_UNAVAILABLE": 503,
    "LLM_OUTPUT_INVALID": 502,
}


class AppError(Exception):
    code: ErrorCode

    def __init__(self, message: str, *, details: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.details = details or {}

    @property
    def http_status(self) -> int:
        return _STATUS_BY_CODE[self.code]

    def to_body(self) -> dict[str, Any]:
        return {"error": {"code": self.code, "message": self.message, "details": self.details}}


class InvalidUrlError(AppError):
    code: ErrorCode = "INVALID_URL"


class ProductNotFoundError(AppError):
    code: ErrorCode = "PRODUCT_NOT_FOUND"


class NoReviewsError(AppError):
    code: ErrorCode = "NO_REVIEWS"


class ScrapeBlockedError(AppError):
    code: ErrorCode = "SCRAPE_BLOCKED"


class ScrapeTimeoutError(AppError):
    code: ErrorCode = "SCRAPE_TIMEOUT"


class LLMUnavailableError(AppError):
    code: ErrorCode = "LLM_UNAVAILABLE"


class LLMOutputInvalidError(AppError):
    code: ErrorCode = "LLM_OUTPUT_INVALID"


class RunNotFoundError(AppError):
    code: ErrorCode = "RUN_NOT_FOUND"
