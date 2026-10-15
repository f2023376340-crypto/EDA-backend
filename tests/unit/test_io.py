from datetime import UTC, datetime
from pathlib import Path

from app.models.domain import RawProduct, RawReview
from app.pipeline import io


def test_raw_reviews_round_trip_keeps_ids_as_strings(tmp_path: Path) -> None:
    run_dir = io.new_run_dir("496954821", now=datetime(2026, 10, 16, 0, 30, 50), base=tmp_path)
    reviews = [
        RawReview(
            review_id="00084193405954821",
            rating=5,
            text="Bohat acha hai",
            raw_text="Bohat acha hai ",
            review_date=datetime(2026, 6, 2, tzinfo=UTC),
            helpful_count=2,
        ),
        RawReview(review_id="2", rating=1, text="خراب", raw_text="خراب"),
        RawReview(
            review_id="3",
            rating=4,
            text="ok",
            raw_text="ok",
            review_date=datetime(2026, 10, 14, 19, 32, 59, 656766, tzinfo=UTC),
        ),
    ]

    io.write_raw_reviews(run_dir, reviews)
    df = io.read_raw_reviews(run_dir)

    assert run_dir.name == "496954821_20261016-003050"
    assert list(df.columns) == io.REVIEW_COLUMNS
    assert df["review_id"].tolist() == ["00084193405954821", "2", "3"]
    assert df["text"].tolist() == ["Bohat acha hai", "خراب", "ok"]
    assert df["rating"].tolist() == [5, 1, 4]
    assert str(df["review_date"].dtype).startswith("datetime64")
    assert df["review_date"].iloc[0] == datetime(2026, 6, 2, tzinfo=UTC)
    assert df["review_date"].isna().iloc[1]
    assert df["review_date"].iloc[2] == datetime(2026, 10, 14, 19, 32, 59, 656766, tzinfo=UTC)


def test_product_round_trip(tmp_path: Path) -> None:
    product = RawProduct(item_id="1", url="u", product_name="P", category_breadcrumb="A > B", price=378.0)
    io.write_product(tmp_path, product)
    assert io.read_product(tmp_path) == product
