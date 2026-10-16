"""Playwright-driven scraper (§5.4): one browser per process, reused; a fresh
context per job, always closed in `finally`.

Sources (details in README.md):
- product header: the PDP's own getDetailInfo response, captured by a
  response listener; DOM fallback after scrolling the lazy rating module in.
- reviews: getReviewList paged directly with httpx (review_api.py), reusing
  this context's cookies and headers; if that endpoint is unusable, fall back to
  scrolling the review module in and clicking "next" until it is disabled.
"""

import asyncio
import random
import sys
import threading
import time
from collections.abc import Coroutine
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import httpx
import structlog
from playwright.async_api import (
    Browser,
    BrowserContext,
    Page,
    Playwright,
    Response,
    Route,
    async_playwright,
)
from playwright.async_api import Error as PlaywrightError
from playwright.async_api import TimeoutError as PlaywrightTimeoutError

from app.core.config import settings
from app.core.errors import ProductNotFoundError, ScrapeBlockedError, ScrapeTimeoutError
from app.core.url_validator import DarazProductRef
from app.models.domain import RawProduct, RawReview, ScrapeResult
from app.pipeline.progress import Emit, no_emit
from app.scraper import parser, review_api
from app.scraper import selectors as sel

log = structlog.get_logger(__name__)

ReviewSource = Literal["auto", "dom"]

_DEBUG_DIR = Path(__file__).resolve().parents[2] / "debug"
_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/129.0.0.0 Safari/537.36"
)
_ACCEPT_LANGUAGE = "en-PK,en;q=0.9"

_SCROLL_STEP_PX = 900

# Keeps Chromium inside small hosts (Render free tier: 512 MB). /dev/shm there
# is tiny, so shared memory goes to /tmp; one renderer, no zygote, no GPU.
_CHROMIUM_ARGS = [
    "--disable-dev-shm-usage",
    "--disable-gpu",
    "--disable-extensions",
    "--no-zygote",
    "--renderer-process-limit=1",
]
# The parsers read JSON responses and DOM text only, so these are never needed.
_BLOCKED_RESOURCE_TYPES = frozenset({"image", "media", "font"})
_SCROLL_MAX_STEPS = 25


class _PlaywrightLoop:
    """All Playwright work runs on one dedicated thread whose event loop can
    start subprocesses. Playwright launches Chromium as a subprocess, and on
    Windows `fastapi dev` (uvicorn --reload) serves requests on a
    SelectorEventLoop, which cannot, so launching from the request loop raises
    NotImplementedError. The browser singleton lives on this loop too.
    """

    def __init__(self) -> None:
        self._loop: asyncio.AbstractEventLoop | None = None
        self._lock = threading.Lock()

    def _ensure(self) -> asyncio.AbstractEventLoop:
        with self._lock:
            if self._loop is None:
                loop = asyncio.ProactorEventLoop() if sys.platform == "win32" else asyncio.new_event_loop()
                threading.Thread(target=loop.run_forever, name="playwright-loop", daemon=True).start()
                self._loop = loop
            return self._loop

    async def run[T](self, coro: Coroutine[Any, Any, T]) -> T:
        return await asyncio.wrap_future(asyncio.run_coroutine_threadsafe(coro, self._ensure()))

    @property
    def started(self) -> bool:
        return self._loop is not None


_playwright_loop = _PlaywrightLoop()
_playwright: Playwright | None = None
_browser: Browser | None = None
_browser_lock = asyncio.Lock()


async def _get_browser() -> Browser:
    global _playwright, _browser
    async with _browser_lock:
        if _browser is None or not _browser.is_connected():
            _playwright = await async_playwright().start()
            _browser = await _playwright.chromium.launch(
                headless=settings.scraper_headless, args=_CHROMIUM_ARGS
            )
    return _browser


async def _skip_heavy_resources(route: Route) -> None:
    if route.request.resource_type in _BLOCKED_RESOURCE_TYPES:
        await route.abort()
    else:
        await route.continue_()


async def _close_browser() -> None:
    global _playwright, _browser
    if _browser is not None:
        await _browser.close()
        _browser = None
    if _playwright is not None:
        await _playwright.stop()
        _playwright = None


async def close_browser() -> None:
    """Called from the FastAPI shutdown hook so the process exits cleanly."""
    if _playwright_loop.started:
        await _playwright_loop.run(_close_browser())


def _wait_timeout_ms() -> float:
    return settings.scraper_wait_timeout_s * 1000


def _random_delay_s() -> float:
    return random.uniform(settings.scraper_min_delay_s, settings.scraper_max_delay_s)


async def _dump_debug(page: Page, label: str) -> None:
    _DEBUG_DIR.mkdir(parents=True, exist_ok=True)
    stamp = int(time.time())
    try:
        await page.screenshot(path=str(_DEBUG_DIR / f"{label}_{stamp}.png"), full_page=True)
        (_DEBUG_DIR / f"{label}_{stamp}.html").write_text(await page.content(), encoding="utf-8")
    except PlaywrightError:
        pass  # best-effort diagnostics; must not mask the real error


class _DetailInfoCapture:
    """Keeps the first getDetailInfo response that parses into a product."""

    def __init__(self, ref: DarazProductRef) -> None:
        self._ref = ref
        self.product: RawProduct | None = None
        self.ready = asyncio.Event()

    async def on_response(self, response: Response) -> None:
        if self.product is not None or sel.DETAIL_INFO_URL_HINT not in response.url.lower():
            return
        try:
            payload: Any = await response.json()
        except (PlaywrightError, ValueError):
            return
        if not isinstance(payload, dict):
            return
        product = parser.parse_detail_info(
            payload, item_id=self._ref.item_id, sku_id=self._ref.sku_id, url=self._ref.canonical_url
        )
        if product is not None:
            self.product = product
            self.ready.set()


async def _scroll_until_present(page: Page, selector: str) -> bool:
    """The rating and review modules are lazy-loaded when scrolled into view."""
    for _ in range(_SCROLL_MAX_STEPS):
        if await page.query_selector(selector) is not None:
            return True
        await page.mouse.wheel(0, _SCROLL_STEP_PX)
        await page.wait_for_timeout(500)
    return await page.query_selector(selector) is not None


async def _load_product(page: Page, capture: _DetailInfoCapture, ref: DarazProductRef) -> RawProduct:
    try:
        await asyncio.wait_for(capture.ready.wait(), timeout=settings.scraper_wait_timeout_s)
    except TimeoutError:
        pass
    if capture.product is not None:
        return capture.product

    log.info("detail_info_missing_using_dom", item_id=ref.item_id)
    await _scroll_until_present(page, sel.OVERALL_RATING)
    return parser.parse_product_page(
        await page.content(), item_id=ref.item_id, sku_id=ref.sku_id, url=ref.canonical_url
    )


async def _reviews_via_api(
    context: BrowserContext, ref: DarazProductRef, max_pages: int, emit: Emit
) -> review_api.ReviewFetchResult:
    cookies = {c["name"]: c["value"] for c in await context.cookies() if "name" in c and "value" in c}
    headers = {
        "User-Agent": _USER_AGENT,
        "Accept-Language": _ACCEPT_LANGUAGE,
        "Accept": "application/json, text/plain, */*",
        "Referer": ref.canonical_url,
    }
    async with httpx.AsyncClient(headers=headers, cookies=cookies, timeout=15.0) as client:
        return await review_api.fetch_reviews(
            client,
            ref.item_id,
            max_pages=max_pages,
            now=datetime.now(UTC),
            delay_s=_random_delay_s,
            on_page=lambda collected: emit("reviews", {"collected": collected}),
        )


def _add_new(new: list[RawReview], seen: set[str], reviews: list[RawReview]) -> int:
    added = 0
    for review in new:
        if review.review_id not in seen:
            seen.add(review.review_id)
            reviews.append(review)
            added += 1
    return added


async def _reviews_via_dom(
    page: Page, total_rating_count: int, max_pages: int, emit: Emit
) -> review_api.ReviewFetchResult:
    """Fallback: scroll the review module in, read page 1 from the DOM, then
    click "next" until it is disabled. Each click makes the page fetch
    getReviewList (pageSize=5); that JSON is read when it arrives, since it
    carries real review ids and dates, else the re-rendered DOM is parsed.
    """
    if not await _scroll_until_present(page, sel.REVIEW_ITEM):
        return review_api.ReviewFetchResult([], partial=True)

    seen: set[str] = set()
    reviews: list[RawReview] = []
    _add_new(parser.parse_review_items_from_dom(await page.content(), now=datetime.now(UTC)), seen, reviews)
    emit("reviews", {"collected": len(reviews)})

    # total_rating_count includes reviews Daraz hides, so it is only an upper
    # bound here; the loop normally ends on a disabled button or an empty page.
    pages_read = 1
    while pages_read < max_pages and not (total_rating_count and len(seen) >= total_rating_count):
        next_button = await page.query_selector(sel.REVIEW_NEXT_PAGE_BUTTON)
        if next_button is None or await next_button.is_disabled():
            break

        await asyncio.sleep(_random_delay_s())
        new: list[RawReview] = []
        try:
            async with page.expect_response(
                lambda r: sel.REVIEW_LIST_URL_HINT in r.url, timeout=_wait_timeout_ms()
            ) as response_info:
                await next_button.click(timeout=_wait_timeout_ms())
            payload = await (await response_info.value).json()
            new = parser.parse_review_list_page(payload, now=datetime.now(UTC)).reviews
        except (PlaywrightTimeoutError, PlaywrightError, ValueError):
            await page.wait_for_timeout(1500)
            new = parser.parse_review_items_from_dom(await page.content(), now=datetime.now(UTC))

        if _add_new(new, seen, reviews) == 0:
            break
        emit("reviews", {"collected": len(reviews)})
        pages_read += 1

    partial = pages_read >= max_pages and (total_rating_count == 0 or len(seen) < total_rating_count)
    return review_api.ReviewFetchResult(reviews, partial=partial)


async def scrape(
    ref: DarazProductRef,
    *,
    max_pages: int | None = None,
    review_source: ReviewSource = "auto",
    emit: Emit = no_emit,
) -> ScrapeResult:
    """`max_pages` bounds review pages: 50 reviews each via the API, 5 each via
    the DOM fallback. `emit` gets `product` once the header is parsed and
    `reviews` as pages come in; it is called from the Playwright thread.
    """
    return await _playwright_loop.run(
        _scrape(ref, max_pages=max_pages, review_source=review_source, emit=emit)
    )


async def _scrape(
    ref: DarazProductRef, *, max_pages: int | None, review_source: ReviewSource, emit: Emit
) -> ScrapeResult:
    max_pages = max_pages or settings.max_review_pages
    browser = await _get_browser()
    context = await browser.new_context(
        locale="en-PK",
        user_agent=_USER_AGENT,
        extra_http_headers={"Accept-Language": _ACCEPT_LANGUAGE},
        viewport={"width": 1366, "height": 900},
    )
    try:
        await context.route("**/*", _skip_heavy_resources)
        page = await context.new_page()
        capture = _DetailInfoCapture(ref)
        page.on("response", capture.on_response)

        try:
            response = await page.goto(
                ref.canonical_url,
                wait_until="domcontentloaded",
                timeout=settings.scraper_nav_timeout_s * 1000,
            )
        except PlaywrightTimeoutError as exc:
            await _dump_debug(page, f"timeout_{ref.item_id}")
            raise ScrapeTimeoutError(
                "Daraz did not respond in time.", details={"item_id": ref.item_id}
            ) from exc

        if response is not None and response.status == 404:
            raise ProductNotFoundError("Product page returned 404.", details={"item_id": ref.item_id})
        if (response is not None and response.status in (403, 429)) or parser.is_blocked_page(
            await page.content()
        ):
            await _dump_debug(page, f"blocked_{ref.item_id}")
            raise ScrapeBlockedError("Daraz returned a bot-check page.", details={"item_id": ref.item_id})

        try:
            await page.wait_for_selector(sel.PRODUCT_NAME, timeout=_wait_timeout_ms())
        except PlaywrightTimeoutError as exc:
            await _dump_debug(page, f"not_found_{ref.item_id}")
            raise ProductNotFoundError(
                "Could not find product details on the page.", details={"item_id": ref.item_id}
            ) from exc

        product = await _load_product(page, capture, ref)
        emit(
            "product",
            {
                "name": product.product_name,
                "category_breadcrumb": product.category_breadcrumb,
                "price": product.price,
                "overall_rating": product.overall_rating,
                "rating_count": product.total_rating_count,
                "url": product.url,
            },
        )
        if product.total_rating_count == 0:
            return ScrapeResult(product=product, reviews=[], partial=False)

        result: review_api.ReviewFetchResult | None = None
        if review_source == "auto":
            try:
                result = await _reviews_via_api(context, ref, max_pages, emit)
            except review_api.ReviewApiUnavailable as exc:
                log.warning("review_api_unavailable_using_dom", item_id=ref.item_id, error=str(exc)[:300])
            if result is not None and not result.reviews:
                log.warning("review_api_returned_nothing_using_dom", item_id=ref.item_id)
                result = None
        source = "api"
        if result is None:
            source = "dom"
            result = await _reviews_via_dom(page, product.total_rating_count, max_pages, emit)

        log.info(
            "scrape_done",
            item_id=ref.item_id,
            reviews=len(result.reviews),
            reported_total=product.total_rating_count,
            partial=result.partial,
            source=source,
        )
        return ScrapeResult(product=product, reviews=result.reviews, partial=result.partial)
    finally:
        await context.close()
