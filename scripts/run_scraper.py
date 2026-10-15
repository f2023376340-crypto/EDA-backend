"""Scrape one product into a new run folder: product.json + raw_reviews.csv.

uv run python -m scripts.run_scraper "<daraz-url>" --max-pages 2
uv run python -m scripts.run_scraper "<daraz-url>" --headed --reviews-from dom
"""

import argparse
import asyncio
import sys
from datetime import UTC, datetime

from app.core.config import settings
from app.core.errors import AppError
from app.core.url_validator import validate_daraz_url
from app.pipeline import io
from app.scraper import client


async def _run(args: argparse.Namespace) -> int:
    if args.headed:
        settings.scraper_headless = False
    try:
        ref = validate_daraz_url(args.url)
        result = await client.scrape(ref, max_pages=args.max_pages, review_source=args.reviews_from)
    except AppError as exc:
        print(f"{exc.code}: {exc.message} {exc.details}", file=sys.stderr)
        return 1
    finally:
        await client.close_browser()

    run_dir = io.new_run_dir(ref.item_id, now=datetime.now())
    io.write_product(run_dir, result.product)
    io.write_raw_reviews(run_dir, result.reviews)
    io.write_run_meta(
        run_dir,
        io.RunMeta(scraped_at=datetime.now(UTC), reviews_scraped=len(result.reviews), partial=result.partial),
    )

    product = result.product
    print(f"run:      {run_dir}")
    print(f"product:  {product.product_name}")
    print(f"category: {product.category_breadcrumb}")
    print(
        f"price:    {product.price}  rating: {product.overall_rating} ({product.total_rating_count} ratings)"
    )
    print(f"reviews:  {len(result.reviews)} scraped  partial={result.partial}")
    return 0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("url")
    ap.add_argument(
        "--max-pages", type=int, default=None, help="review pages (50/page via API, 5/page via DOM)"
    )
    ap.add_argument("--headed", action="store_true", help="show the browser window")
    ap.add_argument("--reviews-from", choices=["auto", "dom"], default="auto", help="force the DOM fallback")
    sys.exit(asyncio.run(_run(ap.parse_args())))


if __name__ == "__main__":
    main()
