import shutil
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from app.pipeline import io
from app.pipeline.cleaning import clean_reviews

FIXTURE = Path(__file__).parents[1] / "fixtures" / "csv" / "raw_reviews_small.csv"


def _raw(tmp_path: Path) -> pd.DataFrame:
    shutil.copy(FIXTURE, tmp_path / "raw_reviews.csv")
    return io.read_raw_reviews(tmp_path)


def test_clean_reviews(tmp_path: Path) -> None:
    raw = _raw(tmp_path)
    before = raw.copy()

    clean = clean_reviews(raw)

    expected = pd.DataFrame(
        {
            # 002 duplicate dropped; 004 blank text and 006 missing rating dropped;
            # 007 shares 002's text but is a different review, so it stays.
            "review_id": ["001", "002", "003", "005", "007"],
            "rating": [5, 5, 4, 5, 1],
            "text": [
                "Good product",
                "good product",
                "Freshness:wow Cleaning:",
                "too high rating",
                "good product",
            ],
            "raw_text": [
                "<b>Good</b>   product",
                "good product",
                "Freshness：wow\nCleaning：",
                "too high rating",
                "good product",
            ],
            "review_date": [
                datetime(2026, 6, 2, tzinfo=UTC),
                datetime(2026, 6, 3, tzinfo=UTC),
                datetime(2026, 10, 14, 19, 32, 59, 656766, tzinfo=UTC),
                None,
                datetime(2026, 6, 6, tzinfo=UTC),
            ],
            "sku_info": [None, None, None, None, "Color:Red"],
            "helpful_count": [0, 1, 0, 0, 2],
        }
    )
    pd.testing.assert_frame_equal(clean, expected, check_dtype=False)
    assert clean["review_id"].dtype == "string"
    assert str(clean["rating"].dtype) == "Int64"
    assert str(clean["review_date"].dtype).startswith("datetime64")
    pd.testing.assert_frame_equal(raw, before)  # input untouched


def test_clean_reviews_round_trips_through_csv(tmp_path: Path) -> None:
    clean = clean_reviews(_raw(tmp_path))
    io.write_clean_reviews(tmp_path, clean)
    pd.testing.assert_frame_equal(io.read_clean_reviews(tmp_path), clean, check_dtype=False)
