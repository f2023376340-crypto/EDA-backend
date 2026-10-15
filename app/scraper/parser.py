"""Pure HTML/JSON -> domain-model parsing. No I/O and no Playwright, so every
function here can be unit-tested against saved fixtures (§6.5).

Sources, in order of preference (see README.md):
- getDetailInfo mtop response -> product header (parse_detail_info)
- getReviewList JSON -> reviews (parse_review_list_page)
- rendered DOM -> both, as a fallback (parse_product_page, parse_review_items_from_dom)

Parsing is defensive: a missing optional field becomes None/0. A review with
no usable rating or id is skipped rather than given an invented value.
"""

import hashlib
import json
import re
from collections import Counter
from datetime import UTC, datetime, timedelta
from typing import Any, NamedTuple

from bs4 import BeautifulSoup
from bs4.element import Tag

from app.models.domain import RatingDistribution, RawProduct, RawReview
from app.scraper import selectors as sel

_MASK_FILL_RE = re.compile(r"half_(\d+(?:\.\d+)?)%")
_RELATIVE_DATE_RE = re.compile(r"^(\d+)\s+(minute|hour|day|week|month|year)s?\s+ago$", re.I)
_RELATIVE_UNITS = {
    "minute": timedelta(minutes=1),
    "hour": timedelta(hours=1),
    "day": timedelta(days=1),
    "week": timedelta(weeks=1),
    "month": timedelta(days=30),
    "year": timedelta(days=365),
}


class ReviewListPage(NamedTuple):
    reviews: list[RawReview]
    item_count: int  # raw items on the page, including any skipped as unusable
    total_items: int
    total_pages: int


# -- shared helpers --------------------------------------------------------


def _text_or_none(tag: Tag | None) -> str | None:
    if tag is None:
        return None
    text = tag.get_text(strip=True)
    return text or None


def _parse_float(text: str | None) -> float | None:
    if not text:
        return None
    match = re.search(r"\d+(?:\.\d+)?", text.replace(",", ""))
    return float(match.group()) if match else None


def _parse_int(text: str | None) -> int:
    if not text:
        return 0
    match = re.search(r"\d+", text.replace(",", ""))
    return int(match.group()) if match else 0


def _clamp_rating(value: Any) -> int | None:
    try:
        rating = round(float(value))
    except (TypeError, ValueError):
        return None
    return rating if 1 <= rating <= 5 else None


def _distribution(counts: list[int]) -> RatingDistribution | None:
    """`counts` is ordered 5-star down to 1-star, as Daraz renders it."""
    if len(counts) != 5:
        return None
    five, four, three, two, one = counts
    return RatingDistribution(five_star=five, four_star=four, three_star=three, two_star=two, one_star=one)


def parse_review_date(text: str | None, *, now: datetime) -> datetime | None:
    """Daraz shows either an absolute date ("23 Feb 2026") or a relative one
    ("1 day ago", "3 weeks ago"). Returns a UTC-aware datetime.
    """
    if not text:
        return None
    text = text.strip()
    try:
        return datetime.strptime(text, "%d %b %Y").replace(tzinfo=UTC)
    except ValueError:
        pass
    lowered = text.lower()
    if lowered in ("just now", "today"):
        return now
    if lowered == "yesterday":
        return now - timedelta(days=1)
    match = _RELATIVE_DATE_RE.match(text)
    if match:
        return now - int(match.group(1)) * _RELATIVE_UNITS[match.group(2).lower()]
    return None


def is_blocked_page(html: str) -> bool:
    soup = BeautifulSoup(html, "lxml")
    if any(soup.select(s) for s in sel.CAPTCHA_SELECTORS):
        return True
    page_text = soup.get_text(" ", strip=True)
    return any(marker in page_text for marker in sel.BLOCKED_PAGE_TEXT_MARKERS)


# -- product ---------------------------------------------------------------


def parse_detail_info(
    payload: dict[str, Any], *, item_id: str, sku_id: str | None, url: str
) -> RawProduct | None:
    """Parse a captured getDetailInfo mtop response. Returns None when the
    response is not the full detail payload (the PDP also fires a short
    token-handshake call to the same API).
    """
    raw_module = (payload.get("data") or {}).get("module")
    if not isinstance(raw_module, str):
        return None
    try:
        module: dict[str, Any] = json.loads(raw_module)
    except json.JSONDecodeError:
        return None

    product = module.get("product") or {}
    title = product.get("title")
    if not title:
        return None

    crumbs = [c.get("title", "") for c in module.get("Breadcrumb") or []]
    if crumbs and crumbs[-1] == title:
        crumbs = crumbs[:-1]

    sku_infos = module.get("skuInfos") or {}
    sku_info = sku_infos.get(sku_id or "") or sku_infos.get("0") or {}
    sale_price = ((sku_info.get("price") or {}).get("salePrice") or {}).get("value")

    ratings = (module.get("review") or {}).get("ratings") or {}
    scores = ratings.get("scores") or []

    return RawProduct(
        item_id=item_id,
        sku_id=sku_id,
        url=url,
        product_name=title,
        category_breadcrumb=" > ".join(c for c in crumbs if c),
        price=float(sale_price) if sale_price is not None else None,
        overall_rating=ratings.get("average"),
        total_rating_count=int(ratings.get("rateCount") or 0),
        rating_distribution=_distribution([int(s) for s in scores]),
    )


def parse_product_page(html: str, *, item_id: str, sku_id: str | None, url: str) -> RawProduct:
    """DOM fallback for the product header."""
    soup = BeautifulSoup(html, "lxml")

    breadcrumb = " > ".join(text for a in soup.select(sel.BREADCRUMB) if (text := a.get_text(strip=True)))
    counts = [_parse_int(_text_or_none(el)) for el in soup.select(sel.RATING_DISTRIBUTION_COUNTS)]

    return RawProduct(
        item_id=item_id,
        sku_id=sku_id,
        url=url,
        product_name=_text_or_none(soup.select_one(sel.PRODUCT_NAME)) or "Unknown product",
        category_breadcrumb=breadcrumb,
        price=_parse_float(_text_or_none(soup.select_one(sel.PRICE))),
        overall_rating=_parse_float(_text_or_none(soup.select_one(sel.OVERALL_RATING))),
        total_rating_count=_parse_int(_text_or_none(soup.select_one(sel.TOTAL_RATING_COUNT))),
        rating_distribution=_distribution(counts),
    )


# -- reviews ---------------------------------------------------------------


def parse_review_list_page(payload: dict[str, Any], *, now: datetime) -> ReviewListPage:
    """Parse one getReviewList response. Reviewer names/ids are deliberately
    not read (§5.4: no personal data beyond what's needed).
    """
    model = payload.get("model") or {}
    paging = model.get("paging") or {}
    reviews: list[RawReview] = []

    items = model.get("items") or []
    for item in items:
        review_id = item.get("reviewRateId")
        rating = _clamp_rating(item.get("rating"))
        if review_id is None or rating is None:
            continue
        raw_text = item.get("reviewContent") or ""
        reviews.append(
            RawReview(
                review_id=str(review_id),
                rating=rating,
                text=raw_text.strip(),
                raw_text=raw_text,
                review_date=parse_review_date(item.get("reviewTime"), now=now),
                sku_info=item.get("skuInfo") or None,
                helpful_count=int(item.get("likeCount") or 0),
            )
        )

    return ReviewListPage(
        reviews=reviews,
        item_count=len(items),
        total_items=int(paging.get("totalItems") or 0),
        total_pages=int(paging.get("totalPages") or 0),
    )


def _star_rating_from_masks(item: Tag) -> int | None:
    fill = 0.0
    for path in item.select(sel.REVIEW_STAR_FILL_PATHS):
        mask = path.get("mask")
        match = _MASK_FILL_RE.search(mask if isinstance(mask, str) else "")
        if match:
            fill += float(match.group(1))
    return _clamp_rating(fill / 100) if fill else None


def parse_review_items_from_dom(html: str, *, now: datetime) -> list[RawReview]:
    """DOM fallback. The DOM carries no review id, so one is derived from the
    review's content plus its occurrence number among identical reviews on the
    page: two buyers' identical "good" 5-star reviews get distinct ids, while
    re-parsing an unchanged page yields the same ids (which is how the click
    loop detects that a page did not advance).
    """
    soup = BeautifulSoup(html, "lxml")
    reviews: list[RawReview] = []
    occurrences: Counter[str] = Counter()

    for item in soup.select(sel.REVIEW_ITEM):
        rating = _star_rating_from_masks(item)
        if rating is None:
            continue
        raw_text = _text_or_none(item.select_one(sel.REVIEW_TEXT)) or ""
        date_text = _text_or_none(item.select_one(sel.REVIEW_DATE))
        likes = _parse_int(_text_or_none(item.select_one(sel.REVIEW_LIKE_COUNT)))
        digest = hashlib.sha1(f"{raw_text}|{rating}|{date_text}|{likes}".encode()).hexdigest()[:16]
        occurrences[digest] += 1

        reviews.append(
            RawReview(
                review_id=f"dom-{digest}-{occurrences[digest]}",
                rating=rating,
                text=raw_text,
                raw_text=raw_text,
                review_date=parse_review_date(date_text, now=now),
                sku_info=None,
                helpful_count=likes,
            )
        )
    return reviews
