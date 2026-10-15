"""All DOM selectors and the review-endpoint URL pattern live here so a
Daraz markup change is a one-file fix (§5.4).

UNCONFIRMED: the Claude in Chrome browser extension was not connected this
session, so these selectors/patterns were not checked against a live Daraz
page. They are best-effort placeholders based on Daraz's known PDP
structure. Confirm against live traffic and update this file + README.md
together before relying on this scraper in production — see
app/scraper/README.md.
"""

# -- Product header -----------------------------------------------------

PRODUCT_NAME = "h1.pdp-mod-product-badge-title"
BREADCRUMB = "ul.breadcrumb a"
PRICE = ".pdp-product-price .pdp-price"
OVERALL_RATING = ".score-average"
TOTAL_RATING_COUNT = ".pdp-review-summary .count"
RATING_DISTRIBUTION_BAR = ".percent"

# -- Reviews (DOM fallback) ----------------------------------------------

REVIEW_SECTION = "#module_product_review"
REVIEW_ITEM = ".item.review-item"
REVIEW_RATING_STARS = ".star span.rating"
REVIEW_TEXT = ".content"
REVIEW_DATE = ".top .title:nth-of-type(2)"
REVIEW_SKU_INFO = ".skuInfo"
REVIEW_HELPFUL_COUNT = ".useful .count"
REVIEW_NEXT_PAGE_BUTTON = ".pdp-review-pagination .next"
REVIEW_NEXT_PAGE_BUTTON_DISABLED_CLASS = "disabled"

# -- Review JSON endpoint (preferred path) -------------------------------

# Daraz's PDP calls an internal review-listing endpoint on load; intercept it
# via a Playwright response listener matching this substring rather than
# hardcoding the full URL, since query params (itemId, pageSize, filter)
# vary and are not yet confirmed.
REVIEW_ENDPOINT_URL_HINT = "/pdp/review/list"

# -- Block / CAPTCHA detection --------------------------------------------

CAPTCHA_SELECTORS = [
    "iframe[src*='captcha']",
    "div[class*='captcha']",
    "#verify-slider",
]
BLOCKED_PAGE_TEXT_MARKERS = [
    "Access Denied",
    "Verify you are human",
]
