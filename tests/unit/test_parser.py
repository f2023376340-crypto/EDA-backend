"""Parser tests against small hand-written HTML matching the (unconfirmed)
selectors in app/scraper/selectors.py. Replace with saved real Daraz pages in
tests/fixtures/ once the selectors are confirmed live.
"""

from app.scraper import parser

PRODUCT_HTML = """
<html><body>
<ul class="breadcrumb"><li><a>Electronic Devices</a></li><li><a>Smartphones</a></li></ul>
<h1 class="pdp-mod-product-badge-title">Redmi Note 13</h1>
<div class="pdp-product-price"><span class="pdp-price">Rs. 52,999</span></div>
<div class="score-average">4.7</div>
<div class="pdp-review-summary"><span class="count">1,234 Ratings</span></div>
</body></html>
"""

REVIEWS_HTML = """
<div id="module_product_review">
  <div class="item review-item" data-review-id="900">
    <div class="star"><span class="rating" style="width: 80%"></span></div>
    <div class="content">Camera is great</div>
  </div>
  <div class="item review-item" data-review-id="901">
    <div class="star"><span class="rating" style="width:20%"></span></div>
    <div class="content"></div>
  </div>
</div>
"""


def test_parse_product_page() -> None:
    product = parser.parse_product_page(PRODUCT_HTML, item_id="1", sku_id=None, url="u")
    assert product.product_name == "Redmi Note 13"
    assert product.category_breadcrumb == "Electronic Devices > Smartphones"
    assert product.price == 52999.0
    assert product.overall_rating == 4.7
    assert product.total_rating_count == 1234


def test_parse_review_items_skips_empty_and_converts_star_width() -> None:
    reviews = parser.parse_review_items_from_dom(REVIEWS_HTML)
    assert [(r.review_id, r.rating, r.text) for r in reviews] == [("900", 4, "Camera is great")]


def test_parse_review_json_page_is_defensive() -> None:
    payload = {
        "items": [
            {"reviewId": 5, "reviewContent": "ok", "rating": 9, "reviewTime": 1_700_000_000_000},
            {"reviewContent": "no id, skipped"},
        ]
    }
    reviews = parser.parse_review_json_page(payload)
    assert len(reviews) == 1
    assert reviews[0].review_id == "5"
    assert reviews[0].rating == 5
    assert reviews[0].review_date is not None


def test_is_blocked_page() -> None:
    assert parser.is_blocked_page("<iframe src='https://x/captcha'></iframe>")
    assert not parser.is_blocked_page(PRODUCT_HTML)
