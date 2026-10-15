"""Pages through Daraz's getReviewList JSON endpoint with httpx (§5.4).

Bounded and idempotent: stops on an empty page, a page with no new review ids,
the reported last page, or `max_pages`. Each page is retried on timeouts, 5xx
and 429 (honouring Retry-After, else backing off >= 10s).
"""

import asyncio
from collections.abc import Callable
from datetime import datetime
from typing import Any, NamedTuple

import httpx
from tenacity import AsyncRetrying, RetryCallState, retry_if_exception_type, stop_after_attempt
from tenacity.wait import wait_random_exponential

from app.core.errors import ScrapeBlockedError
from app.models.domain import RawReview
from app.scraper import parser
from app.scraper import selectors as sel

PAGE_SIZE = 50
_ATTEMPTS = 3
_MIN_RATE_LIMIT_BACKOFF_S = 10.0
_MAX_BACKOFF_S = 10.0


class ReviewApiUnavailable(Exception):
    """The endpoint answered with something other than review JSON on page 1;
    callers should fall back to the DOM.
    """


class ReviewFetchResult(NamedTuple):
    reviews: list[RawReview]
    partial: bool


class _RateLimited(Exception):
    def __init__(self, retry_after: float | None) -> None:
        super().__init__(f"429 (retry-after={retry_after})")
        self.retry_after = retry_after


class _ServerError(Exception):
    pass


class _Forbidden(Exception):
    pass


def _retry_after_s(response: httpx.Response) -> float | None:
    try:
        return float(response.headers["retry-after"])
    except (KeyError, ValueError):
        return None


def _wait(state: RetryCallState) -> float:
    exc = state.outcome.exception() if state.outcome else None
    if isinstance(exc, _RateLimited):
        return exc.retry_after if exc.retry_after is not None else _MIN_RATE_LIMIT_BACKOFF_S
    return float(wait_random_exponential(multiplier=1, max=_MAX_BACKOFF_S)(state))


async def _get_page(client: httpx.AsyncClient, item_id: str, page_no: int, page_size: int) -> dict[str, Any]:
    params: dict[str, str | int] = {
        "itemId": item_id,
        "pageSize": page_size,
        "pageNo": page_no,
        "filter": 0,
        "sort": 0,
    }
    async for attempt in AsyncRetrying(
        retry=retry_if_exception_type((httpx.TransportError, _RateLimited, _ServerError)),
        stop=stop_after_attempt(_ATTEMPTS),
        wait=_wait,
        reraise=True,
    ):
        with attempt:
            response = await client.get(sel.REVIEW_LIST_URL, params=params)
            if response.status_code == 429:
                raise _RateLimited(_retry_after_s(response))
            if response.status_code == 403:
                raise _Forbidden()
            if response.status_code >= 500:
                raise _ServerError(str(response.status_code))
            response.raise_for_status()
            payload: dict[str, Any] = response.json()
            return payload
    raise AssertionError("unreachable: AsyncRetrying reraises")


async def fetch_reviews(
    client: httpx.AsyncClient,
    item_id: str,
    *,
    max_pages: int,
    now: datetime,
    delay_s: Callable[[], float],
    page_size: int = PAGE_SIZE,
) -> ReviewFetchResult:
    seen: set[str] = set()
    reviews: list[RawReview] = []
    total_items = 0
    total_pages = 0
    last_page_no = 0
    last_page_full = False

    for page_no in range(1, max_pages + 1):
        if page_no > 1:
            await asyncio.sleep(delay_s())
        try:
            payload = await _get_page(client, item_id, page_no, page_size)
        except (_RateLimited, _Forbidden) as exc:
            if not reviews:
                raise ScrapeBlockedError(
                    "Daraz is rate-limiting or blocking review requests.",
                    details={"item_id": item_id, "reason": str(exc) or type(exc).__name__},
                ) from exc
            return ReviewFetchResult(reviews, partial=True)
        except (httpx.HTTPError, _ServerError, ValueError) as exc:
            if page_no == 1:
                raise ReviewApiUnavailable(str(exc)) from exc
            return ReviewFetchResult(reviews, partial=True)

        if not payload.get("success"):
            if page_no == 1:
                raise ReviewApiUnavailable(f"success=false: {str(payload)[:200]}")
            return ReviewFetchResult(reviews, partial=True)

        page = parser.parse_review_list_page(payload, now=now)
        if page_no == 1:
            total_items, total_pages = page.total_items, page.total_pages

        new = [r for r in page.reviews if r.review_id not in seen]
        if not new:
            break
        seen.update(r.review_id for r in new)
        reviews.extend(new)
        last_page_no = page_no
        last_page_full = page.item_count >= page_size

        # totalItems/totalPages count reviews Daraz hides (ratings.hiddenCount),
        # so the real end is usually a short page well before totalPages.
        if not last_page_full or page_no >= total_pages or (total_items and len(seen) >= total_items):
            break

    partial = last_page_no == max_pages and last_page_full and last_page_no < total_pages
    return ReviewFetchResult(reviews, partial=partial)
