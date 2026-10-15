# Scraper notes

Confirmed against live daraz.pk on 2026-10-16 (Sensodyne Original Toothpaste 70gm, item `496954821`), using a Playwright network capture plus a manually saved PDP HTML. Test fixtures in `tests/fixtures/daraz/` are trimmed copies of those captures with reviewer names and IDs removed.

## Data sources

### 1. `getDetailInfo` (product header): preferred
- `POST https://acs-m.daraz.pk/h5/mtop.global.detail.web.getdetailinfo/1.0/?...&sign=...`
- Signed with a page-generated token, so **we can't replay it**. Read it with a Playwright response listener (`DETAIL_INFO_URL_HINT`).
- The PDP fires it twice. The first call is a short token handshake with no `data.module`, and the parser returns `None` for it.
- `data.module` is a **JSON string**. The fields we use:

| Field | Path in decoded module |
|---|---|
| Name | `product.title` |
| Category | `Breadcrumb[].title` (last entry is the product itself; dropped) |
| Price (sale) | `skuInfos[<skuId> or "0"].price.salePrice.value` |
| Overall rating | `review.ratings.average` |
| Rating count | `review.ratings.rateCount` |
| Distribution | `review.ratings.scores`: 5 counts, **5★ first** |

`tracking.pdt_price` is the struck-through list price, not the sale price. Don't use it.

### 2. `getReviewList` (reviews): preferred
- `GET https://my.daraz.pk/pdp/review/getReviewList?itemId=<id>&pageSize=<n>&pageNo=<n>&filter=0&sort=0`
- Unsigned, and **works without cookies** from plain `httpx`.
- `pageSize=50` is accepted (169 reviews took 4 pages). The site itself uses 5.
- Past the last page it returns `success: true`, `model.items: []`, and `paging.totalItems: 0`. That's the stop signal.
- `model.paging`: `totalItems`, `totalPages`, `currentPage`.
- **Hidden reviews:** `totalItems`/`totalPages` also count reviews Daraz doesn't serve (`model.ratings.hiddenCount`; probably rating-only or folded). For Sensodyne: 169 total, 91 hidden, so only **78** come back, and the last real page is short (50 + 28). The site's own pager still says 34 pages, but pages past 16 (at 5 per page) are empty. `review_api.fetch_reviews` treats a short page as the end, so stopping there isn't reported as partial.
- Per item: `reviewRateId` (review id, 14 digits), `rating` (int), `reviewContent` (text, may have trailing whitespace and full-width colons), `reviewTime`, `skuInfo` (often null), `likeCount`.
- `reviewTime` is either absolute (`"23 Feb 2026"`) or relative (`"1 day ago"`). `parse_review_date` handles both.
- Items also carry `buyerName`/`buyerId`. **The parser never reads them** (§5.4).

### 3. Rendered DOM: fallback only
Selectors are in `selectors.py`. Watch out for these:
- `.mod-rating` and `.mod-reviews` are **lazy-loaded on scroll**. They aren't in the DOM until the page is scrolled down, so a client must scroll before waiting for them.
- Review page 1 is server-rendered and fires no review XHR. Pages 2+ fire `getReviewList` with `pageSize=5`.
- Review star rating: each of the 5 star SVGs has a gold overlay `path[mask="url(#half_<fill>%)"]`. The rating is `sum(fill) / 100`.
- DOM reviews have no id, and their date span (`.title.right`) is often empty. The parser derives a content-hash id.
- The embedded `var __moduleData__ = {...}` script is a slimmed-down version (title, brand, ids, list price). It has **no** reviews, ratings or breadcrumb.

## Current flow (`client.scrape`)
1. Open the PDP in a fresh browser context, with a listener for `getDetailInfo`. Fail fast on 404, 403/429 or a CAPTCHA page.
2. Get the product from `getDetailInfo` (waits up to 10s). If it never arrives, scroll the rating module in and parse the DOM.
3. Get reviews with `review_api.fetch_reviews`: `httpx`, 50 per page, using the context's cookies and UA, a 1–2.5s random delay, and 3 retries (429 honours `Retry-After`, else waits ≥10s). Repeated 429/403 raises `SCRAPE_BLOCKED` if nothing has been collected yet; otherwise it returns partial results.
4. If the endpoint returns non-JSON or `success:false` on page 1, or no reviews at all, fall back to the DOM. Scroll the review module in, parse page 1, then click `next` until it's disabled or a click adds nothing. After each click, read the `getReviewList` response the page fires (real IDs and dates), or the re-rendered DOM if no response arrives.

Run it: `uv run python -m scripts.run_scraper "<url>" --max-pages 2 [--headed] [--reviews-from dom]`

Verified 2026-10-16. API path: 78/78 reviews, 0 missing dates. DOM path (`--max-pages 4`): 18 reviews (3 + 5 + 5 + 5); page-1 reviews have no date or ID in the DOM.

## Not yet confirmed
- CAPTCHA/bot-wall markers. None were triggered during testing. `#nc_1_wrapper` and `#baxia-dialog-content` are Alibaba's usual slider-captcha containers, but we haven't seen them on Daraz.
- How `getReviewList` responds under rate limiting (status code, `Retry-After`).
- Whether `skuInfo` is filled in for products with variants.

Keep this file in sync with `selectors.py` whenever Daraz's markup or endpoints change.
