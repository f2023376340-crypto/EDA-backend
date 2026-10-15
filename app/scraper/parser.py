"""Pure HTML/JSON -> domain-model parsing. No I/O, no Playwright here — these
functions take already-fetched HTML/JSON and return RawProduct/RawReview
rows, so they can be unit-tested against saved fixtures per §6.5.

Selectors are unconfirmed (see selectors.py docstring); parsing here is
defensive (missing elements become None/0 rather than raising) so a single
layout mismatch degrades gracefully instead of failing the whole scrape.
"""

import re
from datetime import datetime
from typing import Any

from bs4 import BeautifulSoup
from bs4.element import Tag

from app.models.domain import RatingDistribution, RawProduct, RawReview
from app.scraper import selectors as sel

_STAR_WIDTH_RE = re.compile(r"width:\s*([\d.]+)%")


def is_blocked_page(html: str) -> bool:
    soup = BeautifulSoup(html, "lxml")
    if any(soup.select(s) for s in sel.CAPTCHA_SELECTORS):
        return True
    page_text = soup.get_text(" ", strip=True)
    return any(marker in page_text for marker in sel.BLOCKED_PAGE_TEXT_MARKERS)


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


def _star_rating_from_style(tag: Tag | None) -> int:
    """Daraz (as of last confirmation) renders a star rating as a filled-width
    overlay rather than a numeric attribute; convert the width% to 1-5.
    """
    if tag is None:
        return 1
    style = tag.get("style", "")
    match = _STAR_WIDTH_RE.search(style if isinstance(style, str) else "")
    if not match:
        return 1
    percent = float(match.group(1))
    return max(1, min(5, round(percent / 20)))


def parse_product_page(html: str, *, item_id: str, sku_id: str | None, url: str) -> RawProduct:
    soup = BeautifulSoup(html, "lxml")

    name = _text_or_none(soup.select_one(sel.PRODUCT_NAME)) or "Unknown product"
    breadcrumb_links = [a.get_text(strip=True) for a in soup.select(sel.BREADCRUMB)]
    breadcrumb = " > ".join(link for link in breadcrumb_links if link)

    price = _parse_float(_text_or_none(soup.select_one(sel.PRICE)))
    overall_rating = _parse_float(_text_or_none(soup.select_one(sel.OVERALL_RATING)))
    total_rating_count = _parse_int(_text_or_none(soup.select_one(sel.TOTAL_RATING_COUNT)))

    distribution_bars = soup.select(sel.RATING_DISTRIBUTION_BAR)
    rating_distribution = None
    if len(distribution_bars) == 5:
        counts = [_parse_int(_text_or_none(bar)) for bar in distribution_bars]
        rating_distribution = RatingDistribution(
            five_star=counts[0],
            four_star=counts[1],
            three_star=counts[2],
            two_star=counts[3],
            one_star=counts[4],
        )

    return RawProduct(
        item_id=item_id,
        sku_id=sku_id,
        url=url,
        product_name=name,
        category_breadcrumb=breadcrumb,
        price=price,
        overall_rating=overall_rating,
        total_rating_count=total_rating_count,
        rating_distribution=rating_distribution,
    )


def _parse_review_date(text: str | None) -> datetime | None:
    if not text:
        return None
    for fmt in ("%d %b %Y, %H:%M", "%Y-%m-%d", "%d-%m-%Y"):
        try:
            return datetime.strptime(text, fmt)
        except ValueError:
            continue
    return None


def parse_review_items_from_dom(html: str) -> list[RawReview]:
    """DOM fallback path: parse a rendered review-list page."""
    soup = BeautifulSoup(html, "lxml")
    reviews: list[RawReview] = []

    for idx, item in enumerate(soup.select(sel.REVIEW_ITEM)):
        raw_text = _text_or_none(item.select_one(sel.REVIEW_TEXT)) or ""
        if not raw_text:
            continue
        rating = _star_rating_from_style(item.select_one(sel.REVIEW_RATING_STARS))
        review_date = _parse_review_date(_text_or_none(item.select_one(sel.REVIEW_DATE)))
        sku_info = _text_or_none(item.select_one(sel.REVIEW_SKU_INFO))
        helpful_count = _parse_int(_text_or_none(item.select_one(sel.REVIEW_HELPFUL_COUNT)))

        review_id = item.get("data-review-id")
        review_id_str = str(review_id) if review_id else f"dom-{hash(raw_text)}-{idx}"

        reviews.append(
            RawReview(
                review_id=review_id_str,
                rating=rating,
                text=raw_text,
                raw_text=raw_text,
                review_date=review_date,
                sku_info=sku_info,
                helpful_count=helpful_count,
            )
        )
    return reviews


def parse_review_json_page(payload: dict[str, Any]) -> list[RawReview]:
    """Preferred path: parse one page of the intercepted review JSON endpoint.

    The exact response shape is UNCONFIRMED (see selectors.py). This assumes
    a top-level `items`/`reviews` list of objects with reasonably-named
    fields and is defensive about missing keys; it must be re-checked
    against a real captured payload.
    """
    items = payload.get("items") or payload.get("reviews") or payload.get("data") or []
    reviews: list[RawReview] = []

    for item in items:
        review_id = str(item.get("reviewId") or item.get("id") or "")
        if not review_id:
            continue
        raw_text = str(item.get("reviewContent") or item.get("content") or "")
        rating = item.get("rating") or item.get("ratingScore") or 1
        review_date = None
        if raw_ts := item.get("reviewTime") or item.get("createTime"):
            try:
                review_date = datetime.fromtimestamp(int(raw_ts) / 1000)
            except (ValueError, TypeError, OSError):
                review_date = None

        reviews.append(
            RawReview(
                review_id=review_id,
                rating=max(1, min(5, int(rating))),
                text=raw_text,
                raw_text=raw_text,
                review_date=review_date,
                sku_info=item.get("skuInfo"),
                helpful_count=int(item.get("usefulCount") or 0),
            )
        )
    return reviews
