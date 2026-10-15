"""All DOM selectors and endpoint patterns for Daraz PDPs, so a markup
change is a one-file fix (§5.4). Confirmed against live daraz.pk on
2026-10-16; see README.md for the data sources and their shapes.
"""

# -- Structured sources (preferred) ---------------------------------------

# Signed mtop call the PDP makes on load. We cannot replay it (needs the page's
# signing token), but a Playwright response listener can read it. Its
# `data.module` is a JSON *string* holding product, Breadcrumb, skuInfos and
# review.ratings.
DETAIL_INFO_URL_HINT = "mtop.global.detail.web.getdetailinfo"

# Unsigned GET, works without cookies. Params: itemId, pageSize (50 accepted),
# pageNo (1-based), filter=0, sort=0. Past the last page it returns
# success=true with model.items == [].
REVIEW_LIST_URL = "https://my.daraz.pk/pdp/review/getReviewList"
REVIEW_LIST_URL_HINT = "/pdp/review/getReviewList"

# -- Product header (DOM fallback) ----------------------------------------

PRODUCT_NAME = "h1.pdp-mod-product-badge-title"
# The last crumb is the product itself, rendered as a <span>, so selecting only
# <a> anchors yields just the category path.
BREADCRUMB = "a.breadcrumb_item_anchor"
PRICE = ".pdp-product-price .pdp-price_type_normal"
OVERALL_RATING = ".mod-rating .score-average"
TOTAL_RATING_COUNT = ".mod-rating .summary .count"
# Five counts, ordered 5-star down to 1-star.
RATING_DISTRIBUTION_COUNTS = ".mod-rating .detail li .percent"

# -- Reviews (DOM fallback) -----------------------------------------------

REVIEW_SECTION = ".mod-reviews"
REVIEW_ITEM = ".mod-reviews .item"
REVIEW_TEXT = ".item-content .content"
# Often empty on the first server-rendered page.
REVIEW_DATE = ".top .title.right"
REVIEW_LIKE_COUNT = ".bottom .left-content"
# Each star SVG has a gold overlay path masked by url(#half_<fill>%); the fill
# percentages summed / 100 give the star rating.
REVIEW_STAR_FILL_PATHS = ".starCtn path[mask]"
REVIEW_NEXT_PAGE_BUTTON = ".review-pagination .next-pagination-item.next"

# -- Block / CAPTCHA detection --------------------------------------------

CAPTCHA_SELECTORS = [
    "iframe[src*='captcha']",
    "div[class*='captcha']",
    "#nc_1_wrapper",
    "#baxia-dialog-content",
]
BLOCKED_PAGE_TEXT_MARKERS = [
    "Access Denied",
    "Verify you are human",
    "unusual traffic",
]
