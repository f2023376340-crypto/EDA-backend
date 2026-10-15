"""The only module that reads/writes run files (CLAUDE.md §5.2). Column names
are the contract between stages; see the table in CLAUDE.md §2.
"""

from datetime import datetime
from pathlib import Path

import pandas as pd
from pydantic import BaseModel

from app.models.domain import RawProduct, RawReview
from app.models.jev import FeatureGenerationOutcome

RUNS_DIR = Path(__file__).resolve().parents[2] / "data" / "runs"

REVIEW_COLUMNS = ["review_id", "rating", "text", "raw_text", "review_date", "sku_info", "helpful_count"]
# One row per review sent to Jev; score/level/confidence are empty when the request failed.
SCORE_COLUMNS = ["review_id", "score", "level", "confidence", "model", "rubric_version"]

_REVIEW_DTYPES = {
    "review_id": "string",
    "rating": "Int64",
    "text": "string",
    "raw_text": "string",
    "sku_info": "string",
    "helpful_count": "Int64",
}
_SCORE_DTYPES = {
    "review_id": "string",
    "score": "float64",
    "level": "Int64",
    "confidence": "float64",
    "model": "string",
    "rubric_version": "string",
}


class RunMeta(BaseModel):
    """Scrape facts the later stages need but no CSV carries."""

    scraped_at: datetime
    reviews_scraped: int
    partial: bool


def new_run_dir(item_id: str, *, now: datetime, base: Path | None = None) -> Path:
    run_dir = (base or RUNS_DIR) / f"{item_id}_{now:%Y%m%d-%H%M%S}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def write_run_meta(run_dir: Path, meta: RunMeta) -> Path:
    return _write_json(run_dir / "run_meta.json", meta)


def read_run_meta(run_dir: Path) -> RunMeta:
    return RunMeta.model_validate_json((run_dir / "run_meta.json").read_text(encoding="utf-8"))


def _read_csv(path: Path, dtypes: dict[str, str], columns: list[str]) -> pd.DataFrame:
    df = pd.read_csv(path, dtype=dtypes, keep_default_na=False, na_values=[""])
    missing = set(columns) - set(df.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    return df[columns]


def _write_csv(path: Path, df: pd.DataFrame, columns: list[str]) -> Path:
    df.reindex(columns=columns).to_csv(path, index=False, encoding="utf-8")
    return path


def _write_json(path: Path, model: BaseModel) -> Path:
    path.write_text(model.model_dump_json(indent=2), encoding="utf-8")
    return path


# -- product / features / result ---------------------------------------------


def write_product(run_dir: Path, product: RawProduct) -> Path:
    return _write_json(run_dir / "product.json", product)


def read_product(run_dir: Path) -> RawProduct:
    return RawProduct.model_validate_json((run_dir / "product.json").read_text(encoding="utf-8"))


def write_features(run_dir: Path, outcome: FeatureGenerationOutcome) -> Path:
    return _write_json(run_dir / "features.json", outcome)


def read_features(run_dir: Path) -> FeatureGenerationOutcome:
    return FeatureGenerationOutcome.model_validate_json(
        (run_dir / "features.json").read_text(encoding="utf-8")
    )


def write_analysis(run_dir: Path, result: BaseModel) -> Path:
    """The final AnalysisResult, so a finished run can be served/inspected again."""
    return _write_json(run_dir / "analysis.json", result)


# -- reviews ---------------------------------------------------------------------


def _read_reviews(path: Path) -> pd.DataFrame:
    df = _read_csv(path, _REVIEW_DTYPES, REVIEW_COLUMNS)
    # Without format="ISO8601", pandas infers one format from the first row and
    # silently turns rows with fractional seconds (relative dates) into NaT.
    df["review_date"] = pd.to_datetime(df["review_date"], utc=True, errors="coerce", format="ISO8601")
    return df


def write_raw_reviews(run_dir: Path, reviews: list[RawReview]) -> Path:
    df = pd.DataFrame([r.model_dump() for r in reviews], columns=REVIEW_COLUMNS)
    return _write_csv(run_dir / "raw_reviews.csv", df, REVIEW_COLUMNS)


def read_raw_reviews(run_dir: Path) -> pd.DataFrame:
    return _read_reviews(run_dir / "raw_reviews.csv")


def write_clean_reviews(run_dir: Path, df: pd.DataFrame) -> Path:
    return _write_csv(run_dir / "clean_reviews.csv", df, REVIEW_COLUMNS)


def read_clean_reviews(run_dir: Path) -> pd.DataFrame:
    return _read_reviews(run_dir / "clean_reviews.csv")


# -- Jev scores --------------------------------------------------------------------


def write_scores(run_dir: Path, df: pd.DataFrame) -> Path:
    return _write_csv(run_dir / "jev_scores.csv", df, SCORE_COLUMNS)


def read_scores(run_dir: Path) -> pd.DataFrame:
    return _read_csv(run_dir / "jev_scores.csv", _SCORE_DTYPES, SCORE_COLUMNS)
