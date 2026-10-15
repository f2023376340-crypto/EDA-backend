"""Parser tests against trimmed, PII-scrubbed captures of a real Daraz PDP
(Sensodyne Original Toothpaste 70gm, item 496954821) taken 2026-10-16.
"""

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from app.models.domain import RatingDistribution
from app.scraper import parser

FIXTURES = Path(__file__).parents[1] / "fixtures" / "daraz"
NOW = datetime(2026, 10, 16, 12, 0, tzinfo=UTC)
URL = "https://www.daraz.pk/products/sensodyne-original-toothpaste-70gm-i496954821-s2379054862.html"
CATEGORY = "Health & Beauty > Personal Care > Oral Care > Toothpaste"
DISTRIBUTION = RatingDistribution(five_star=156, four_star=6, three_star=1, two_star=2, one_star=2)


def _json(name: str) -> dict:  # type: ignore[type-arg]
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))  # type: ignore[no-any-return]


def _html() -> str:
    return (FIXTURES / "pdp.html").read_text(encoding="utf-8")


def test_parse_detail_info() -> None:
    product = parser.parse_detail_info(
        _json("detail_info.json"), item_id="496954821", sku_id="2379054862", url=URL
    )
    assert product is not None
    assert product.product_name == "Sensodyne Original Toothpaste 70gm"
    assert product.category_breadcrumb == CATEGORY
    assert product.price == 378.0
    assert product.overall_rating == 4.9
    assert product.total_rating_count == 169
    assert product.rating_distribution == DISTRIBUTION


def test_parse_detail_info_ignores_token_handshake_response() -> None:
    handshake = {"api": "mtop.global.detail.web.getdetailinfo", "data": {}, "ret": ["FAIL_SYS_TOKEN_EMPTY"]}
    assert parser.parse_detail_info(handshake, item_id="1", sku_id=None, url=URL) is None


def test_parse_product_page_dom_fallback() -> None:
    product = parser.parse_product_page(_html(), item_id="496954821", sku_id="2379054862", url=URL)
    assert product.product_name == "Sensodyne Original Toothpaste 70gm"
    assert product.category_breadcrumb == CATEGORY
    assert product.price == 284.0  # sale price on the day the HTML was saved
    assert product.overall_rating == 4.9
    assert product.total_rating_count == 169
    assert product.rating_distribution == DISTRIBUTION


def test_parse_review_list_page() -> None:
    page = parser.parse_review_list_page(_json("review_list_page.json"), now=NOW)

    assert page.total_items == 169
    assert page.total_pages == 34
    assert [r.review_id for r in page.reviews] == ["84193405954821", "84216280554821", "82041728554821"]
    first, _, third = page.reviews
    assert first.rating == 5
    assert first.review_date == NOW - timedelta(days=1)
    assert first.raw_text.endswith(" ")
    assert not first.text.endswith(" ")
    assert third.review_date == datetime(2026, 6, 2, tzinfo=UTC)


def test_parse_review_list_page_past_last_page() -> None:
    empty = {"success": True, "model": {"items": [], "paging": {"totalItems": 0, "totalPages": 0}}}
    page = parser.parse_review_list_page(empty, now=NOW)
    assert page.reviews == []
    assert page.total_pages == 0


def test_parse_review_items_from_dom() -> None:
    reviews = parser.parse_review_items_from_dom(_html(), now=NOW)

    assert len(reviews) == 3
    assert all(r.rating == 5 for r in reviews)
    assert reviews[0].text.startswith("Cleaning:90percents")
    assert len({r.review_id for r in reviews}) == 3
    # Ids are content-derived, so re-parsing the same page is idempotent.
    assert [r.review_id for r in parser.parse_review_items_from_dom(_html(), now=NOW)] == [
        r.review_id for r in reviews
    ]


def _dom_review(text: str, fills: tuple[int, ...] = (100, 100, 100, 100, 100)) -> str:
    stars = "".join(
        f'<div class="i-rate-star"><svg><path mask="url(#half_{p}%)"></path></svg></div>' for p in fills
    )
    return (
        f'<div class="item"><div class="top"><div class="starCtn">{stars}</div></div>'
        f'<div class="item-content"><div class="content">{text}</div></div></div>'
    )


def test_identical_dom_reviews_get_distinct_ids() -> None:
    html = f'<div class="mod-reviews">{_dom_review("good")}{_dom_review("good")}{_dom_review("nice")}</div>'
    ids = [r.review_id for r in parser.parse_review_items_from_dom(html, now=NOW)]
    assert len(set(ids)) == 3
    assert ids[0].endswith("-1") and ids[1].endswith("-2")
    assert ids == [r.review_id for r in parser.parse_review_items_from_dom(html, now=NOW)]


def test_dom_star_rating_sums_mask_fill() -> None:
    star = '<div class="i-rate-star"><svg><path mask="url(#half_{p}%)"></path></svg></div>'
    stars = "".join(star.format(p=p) for p in (100, 100, 100, 50, 0))
    html = (
        '<div class="mod-reviews"><div class="item"><div class="top">'
        f'<div class="starCtn">{stars}</div></div>'
        '<div class="item-content"><div class="content">ok</div></div></div></div>'
    )
    [review] = parser.parse_review_items_from_dom(html, now=NOW)
    assert review.rating == 4


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("23 Feb 2026", datetime(2026, 2, 23, tzinfo=UTC)),
        ("1 day ago", NOW - timedelta(days=1)),
        ("3 weeks ago", NOW - timedelta(weeks=3)),
        ("5 hours ago", NOW - timedelta(hours=5)),
        ("yesterday", NOW - timedelta(days=1)),
        ("", None),
        ("sometime", None),
    ],
)
def test_parse_review_date(text: str, expected: datetime | None) -> None:
    assert parser.parse_review_date(text, now=NOW) == expected


def test_is_blocked_page() -> None:
    assert parser.is_blocked_page("<iframe src='https://x/captcha'></iframe>")
    assert parser.is_blocked_page("<div id='nc_1_wrapper'></div>")
    assert not parser.is_blocked_page(_html())
