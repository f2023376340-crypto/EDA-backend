from datetime import UTC, datetime
from typing import Any

import httpx
import pytest

from app.core.errors import ScrapeBlockedError
from app.scraper import review_api

NOW = datetime(2026, 10, 16, tzinfo=UTC)


@pytest.fixture(autouse=True)
def _no_backoff(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(review_api, "_MIN_RATE_LIMIT_BACKOFF_S", 0.0)
    monkeypatch.setattr(review_api, "_MAX_BACKOFF_S", 0.0)


def _items(start: int, count: int) -> list[dict[str, Any]]:
    return [
        {
            "reviewRateId": start + i,
            "rating": 5,
            "reviewContent": f"r{start + i}",
            "reviewTime": "02 Jun 2026",
        }
        for i in range(count)
    ]


def _page(items: list[dict[str, Any]], total_items: int = 169, total_pages: int = 4) -> dict[str, Any]:
    paging = (
        {"totalItems": total_items, "totalPages": total_pages}
        if items
        else {"totalItems": 0, "totalPages": 0}
    )
    return {"success": True, "model": {"items": items, "paging": paging}}


def _client(responses: dict[int, list[httpx.Response]], calls: list[int]) -> httpx.AsyncClient:
    def handler(request: httpx.Request) -> httpx.Response:
        page_no = int(request.url.params["pageNo"])
        calls.append(page_no)
        return responses[page_no].pop(0)

    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


async def _fetch(
    client: httpx.AsyncClient, max_pages: int = 10, page_size: int = 3
) -> review_api.ReviewFetchResult:
    return await review_api.fetch_reviews(
        client, "1", max_pages=max_pages, now=NOW, delay_s=lambda: 0.0, page_size=page_size
    )


async def test_stops_on_short_page_even_though_total_pages_says_more() -> None:
    calls: list[int] = []
    responses = {
        1: [httpx.Response(200, json=_page(_items(0, 3)))],
        2: [httpx.Response(200, json=_page(_items(3, 2)))],
    }
    result = await _fetch(_client(responses, calls))
    assert [r.review_id for r in result.reviews] == ["0", "1", "2", "3", "4"]
    assert result.partial is False
    assert calls == [1, 2]


async def test_stops_on_empty_page() -> None:
    calls: list[int] = []
    responses = {1: [httpx.Response(200, json=_page(_items(0, 3)))], 2: [httpx.Response(200, json=_page([]))]}
    result = await _fetch(_client(responses, calls))
    assert len(result.reviews) == 3
    assert result.partial is False


async def test_stops_when_a_page_repeats() -> None:
    calls: list[int] = []
    same = _page(_items(0, 3))
    responses = {1: [httpx.Response(200, json=same)], 2: [httpx.Response(200, json=same)]}
    result = await _fetch(_client(responses, calls))
    assert len(result.reviews) == 3
    assert calls == [1, 2]


async def test_max_pages_with_full_last_page_is_partial() -> None:
    calls: list[int] = []
    responses = {
        1: [httpx.Response(200, json=_page(_items(0, 3)))],
        2: [httpx.Response(200, json=_page(_items(3, 3)))],
    }
    result = await _fetch(_client(responses, calls), max_pages=2)
    assert len(result.reviews) == 6
    assert result.partial is True


async def test_retries_5xx_then_succeeds() -> None:
    calls: list[int] = []
    responses = {1: [httpx.Response(503), httpx.Response(200, json=_page(_items(0, 2)))]}
    result = await _fetch(_client(responses, calls))
    assert len(result.reviews) == 2
    assert calls == [1, 1]


async def test_repeated_429_after_some_reviews_returns_partial() -> None:
    calls: list[int] = []
    limited = [httpx.Response(429, headers={"Retry-After": "0"}) for _ in range(3)]
    responses = {1: [httpx.Response(200, json=_page(_items(0, 3)))], 2: limited}
    result = await _fetch(_client(responses, calls))
    assert len(result.reviews) == 3
    assert result.partial is True
    assert calls == [1, 2, 2, 2]


async def test_repeated_429_on_first_page_is_blocked() -> None:
    calls: list[int] = []
    responses = {1: [httpx.Response(429) for _ in range(3)]}
    with pytest.raises(ScrapeBlockedError):
        await _fetch(_client(responses, calls))


async def test_non_json_first_page_signals_dom_fallback() -> None:
    calls: list[int] = []
    responses = {1: [httpx.Response(200, text="<html>login</html>")]}
    with pytest.raises(review_api.ReviewApiUnavailable):
        await _fetch(_client(responses, calls))
