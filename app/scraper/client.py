"""Playwright-driven scraper (§5.4): one browser per process, reused; a
fresh context per job; always closed in `finally`.

Review pagination: the review JSON endpoint the PDP itself calls is
intercepted via a response listener (preferred, per CLAUDE.md §2 stage 2),
but its exact query parameters are UNCONFIRMED this session (see
selectors.py / README.md) so we cannot yet hand-craft httpx pagination
requests against it. Until that's confirmed, pagination is driven by
clicking the page's own "next" control — each click naturally re-triggers
the same endpoint, so captured responses are still parsed as JSON when
present, falling back to BeautifulSoup over the rendered DOM otherwise.
Swap in direct httpx paging once the endpoint is confirmed; the capture
plumbing here does not change.
"""

import asyncio
import random
import time
from pathlib import Path
from typing import Any

from playwright.async_api import (
    Browser,
    Page,
    Playwright,
    async_playwright,
)
from playwright.async_api import (
    TimeoutError as PlaywrightTimeoutError,
)
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_random_exponential

from app.core.config import settings
from app.core.errors import ProductNotFoundError, ScrapeBlockedError, ScrapeTimeoutError
from app.core.url_validator import DarazProductRef
from app.models.domain import RawReview, ScrapeResult
from app.scraper import parser
from app.scraper import selectors as sel

_DEBUG_DIR = Path(__file__).resolve().parents[2] / "debug"

_NAV_TIMEOUT_MS = 30_000
_SELECTOR_TIMEOUT_MS = 10_000

_playwright: Playwright | None = None
_browser: Browser | None = None
_browser_lock = asyncio.Lock()


async def _get_browser() -> Browser:
    global _playwright, _browser
    async with _browser_lock:
        if _browser is None or not _browser.is_connected():
            _playwright = await async_playwright().start()
            _browser = await _playwright.chromium.launch(headless=settings.scraper_headless)
    return _browser


async def close_browser() -> None:
    """Call from the FastAPI shutdown hook so the process exits cleanly."""
    global _playwright, _browser
    if _browser is not None:
        await _browser.close()
        _browser = None
    if _playwright is not None:
        await _playwright.stop()
        _playwright = None


def _random_delay_s() -> float:
    return random.uniform(settings.scraper_min_delay_s, settings.scraper_max_delay_s)


async def _dump_debug(page: Page, label: str) -> None:
    _DEBUG_DIR.mkdir(parents=True, exist_ok=True)
    stamp = int(time.time())
    try:
        await page.screenshot(path=str(_DEBUG_DIR / f"{label}_{stamp}.png"), full_page=True)
        html = await page.content()
        (_DEBUG_DIR / f"{label}_{stamp}.html").write_text(html, encoding="utf-8")
    except Exception:  # noqa: BLE001 - best-effort diagnostics, must not mask the real error
        pass


class _ReviewJsonCapture:
    """Collects JSON bodies of responses that look like the review endpoint."""

    def __init__(self) -> None:
        self.pages: list[dict[str, Any]] = []

    async def on_response(self, response: Any) -> None:
        if sel.REVIEW_ENDPOINT_URL_HINT not in response.url:
            return
        content_type = response.headers.get("content-type", "")
        if "application/json" not in content_type:
            return
        try:
            payload = await response.json()
        except Exception:  # noqa: BLE001 - not every matching response is parseable JSON
            return
        self.pages.append(payload)


@retry(
    retry=retry_if_exception_type(PlaywrightTimeoutError),
    stop=stop_after_attempt(3),
    wait=wait_random_exponential(multiplier=1, max=10),
)
async def _goto(page: Page, url: str) -> None:
    await page.goto(url, wait_until="domcontentloaded", timeout=_NAV_TIMEOUT_MS)


async def _consume_captured_pages(
    capture: _ReviewJsonCapture, seen_ids: set[str], reviews: list[RawReview]
) -> bool:
    """Parses and appends any newly-captured JSON pages. Returns True if any
    review was new (used to detect an exhausted/idempotent pagination loop).
    """
    found_new = False
    for payload in capture.pages:
        for review in parser.parse_review_json_page(payload):
            if review.review_id not in seen_ids:
                seen_ids.add(review.review_id)
                reviews.append(review)
                found_new = True
    capture.pages.clear()
    return found_new


async def _collect_reviews(
    page: Page, capture: _ReviewJsonCapture, total_rating_count: int
) -> tuple[list[RawReview], bool]:
    seen_ids: set[str] = set()
    reviews: list[RawReview] = []

    # The initial PDP load may already have triggered the first page of reviews.
    await _consume_captured_pages(capture, seen_ids, reviews)

    page_num = 1
    while page_num < settings.max_review_pages:
        if total_rating_count and len(seen_ids) >= total_rating_count:
            break

        next_button = await page.query_selector(sel.REVIEW_NEXT_PAGE_BUTTON)
        if next_button is None:
            break
        class_attr = (await next_button.get_attribute("class")) or ""
        if sel.REVIEW_NEXT_PAGE_BUTTON_DISABLED_CLASS in class_attr:
            break

        try:
            await next_button.click(timeout=_SELECTOR_TIMEOUT_MS)
        except PlaywrightTimeoutError:
            break

        await asyncio.sleep(_random_delay_s())

        got_new_from_api = await _consume_captured_pages(capture, seen_ids, reviews)
        if not got_new_from_api:
            html = await page.content()
            for review in parser.parse_review_items_from_dom(html):
                if review.review_id not in seen_ids:
                    seen_ids.add(review.review_id)
                    reviews.append(review)

        page_num += 1

    partial = page_num >= settings.max_review_pages and (
        total_rating_count == 0 or len(seen_ids) < total_rating_count
    )
    return reviews, partial


async def scrape(ref: DarazProductRef) -> ScrapeResult:
    browser = await _get_browser()
    context = await browser.new_context(
        locale="en-PK",
        user_agent=(
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/129.0.0.0 Safari/537.36"
        ),
        extra_http_headers={"Accept-Language": "en-PK,en;q=0.9"},
    )
    try:
        page = await context.new_page()
        capture = _ReviewJsonCapture()
        page.on("response", capture.on_response)

        try:
            await _goto(page, ref.canonical_url)
        except PlaywrightTimeoutError as exc:
            await _dump_debug(page, f"timeout_{ref.item_id}")
            raise ScrapeTimeoutError(
                "Daraz did not respond in time.", details={"item_id": ref.item_id}
            ) from exc

        html = await page.content()
        if parser.is_blocked_page(html):
            await _dump_debug(page, f"blocked_{ref.item_id}")
            raise ScrapeBlockedError("Daraz returned a bot-check page.", details={"item_id": ref.item_id})

        try:
            await page.wait_for_selector(sel.PRODUCT_NAME, timeout=_SELECTOR_TIMEOUT_MS)
        except PlaywrightTimeoutError as exc:
            await _dump_debug(page, f"not_found_{ref.item_id}")
            raise ProductNotFoundError(
                "Could not find product details on the page.",
                details={"item_id": ref.item_id},
            ) from exc

        html = await page.content()
        product = parser.parse_product_page(
            html, item_id=ref.item_id, sku_id=ref.sku_id, url=ref.canonical_url
        )

        reviews: list[RawReview] = []
        partial = False
        if product.total_rating_count > 0:
            try:
                await page.wait_for_selector(sel.REVIEW_SECTION, timeout=_SELECTOR_TIMEOUT_MS)
                reviews, partial = await _collect_reviews(page, capture, product.total_rating_count)
            except PlaywrightTimeoutError:
                # Review section never rendered; return the product with no reviews
                # rather than failing the whole scrape.
                partial = True

        return ScrapeResult(product=product, reviews=reviews, partial=partial)
    finally:
        await context.close()
