# Scraper notes

## Status: selectors and endpoint UNCONFIRMED

`selectors.py` was written without access to live Daraz traffic. Everything in it is a best-effort placeholder until it is checked against a real PDP.

## How it works now

1. Navigate to the canonical URL rebuilt from `item_id` (never the raw user input).
2. Check for a CAPTCHA or block page, then wait for `PRODUCT_NAME`, then parse the product header.
3. A Playwright response listener captures any JSON response whose URL contains `REVIEW_ENDPOINT_URL_HINT`. Those payloads are parsed with `parser.parse_review_json_page`.
4. Pagination clicks `REVIEW_NEXT_PAGE_BUTTON`, waits a random 1–2.5s, and parses any newly captured JSON. If a click captures nothing new, it falls back to BeautifulSoup over the rendered DOM.
5. It stops on: no next button, a disabled next button, the reported total reached, or `MAX_REVIEW_PAGES`.

Direct `httpx` paging against the review endpoint (the preferred path in CLAUDE.md) is **not** implemented yet. The endpoint's query parameters are unknown, and guessing them would be worse than clicking.

## To confirm against live traffic

- [ ] The review endpoint URL, method, required headers/cookies, and pagination params (page/offset, page size, end-of-pages signal). Once known, replace click-pagination with `httpx` paging and record the details here.
- [ ] The real JSON response shape. Update `parse_review_json_page` field names (currently guesses such as `items`, `reviewId`, `reviewContent`, `rating`, `reviewTime`).
- [ ] Every CSS selector in `selectors.py`, including how star ratings are rendered (currently assumed to be a `width:%` overlay).
- [ ] The review date format. `_parse_review_date` tries a few formats.
- [ ] Anti-bot behaviour (CAPTCHA triggers, rate limits, required UA/locale).
- [ ] Save a real PDP HTML page and a review JSON page into `tests/fixtures/`, and point `tests/unit/test_parser.py` at them.

Quick check: `uv run python -m scripts.run_scraper "<daraz-url>" --headed` (script not written yet), or call `POST /api/v1/analyze` and inspect `backend/debug/` on failure.

Keep this file in sync with `selectors.py` whenever Daraz's markup or endpoints change.
