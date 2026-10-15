"""The only module that reads/writes run files (CLAUDE.md §5.2). Column names
are the contract between stages; see the table in CLAUDE.md §2.
"""

from datetime import datetime
from pathlib import Path

import pandas as pd

from app.models.domain import RawProduct, RawReview

RUNS_DIR = Path(__file__).resolve().parents[2] / "data" / "runs"

REVIEW_COLUMNS = ["review_id", "rating", "text", "raw_text", "review_date", "sku_info", "helpful_count"]
_REVIEW_DTYPES = {
    "review_id": "string",
    "rating": "Int64",
    "text": "string",
    "raw_text": "string",
    "sku_info": "string",
    "helpful_count": "Int64",
}


def new_run_dir(item_id: str, *, now: datetime, base: Path = RUNS_DIR) -> Path:
    run_dir = base / f"{item_id}_{now:%Y%m%d-%H%M%S}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def write_product(run_dir: Path, product: RawProduct) -> Path:
    path = run_dir / "product.json"
    path.write_text(product.model_dump_json(indent=2), encoding="utf-8")
    return path


def read_product(run_dir: Path) -> RawProduct:
    return RawProduct.model_validate_json((run_dir / "product.json").read_text(encoding="utf-8"))


def write_raw_reviews(run_dir: Path, reviews: list[RawReview]) -> Path:
    path = run_dir / "raw_reviews.csv"
    df = pd.DataFrame([r.model_dump() for r in reviews], columns=REVIEW_COLUMNS)
    df.to_csv(path, index=False, encoding="utf-8")
    return path


def read_raw_reviews(run_dir: Path) -> pd.DataFrame:
    path = run_dir / "raw_reviews.csv"
    df = pd.read_csv(path, dtype=_REVIEW_DTYPES, keep_default_na=False, na_values=[""])
    missing = set(REVIEW_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    # Without format="ISO8601", pandas infers one format from the first row and
    # silently turns rows with fractional seconds (relative dates) into NaT.
    df["review_date"] = pd.to_datetime(df["review_date"], utc=True, errors="coerce", format="ISO8601")
    return df
