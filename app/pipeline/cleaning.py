"""Stage 3 (CLAUDE.md §2): raw_reviews -> clean_reviews. Pure: DataFrame in,
new DataFrame out, input untouched, no file I/O.

Deduplication is by review_id only. Identical texts from different buyers are
separate reviews (Daraz ids are unique; DOM-fallback ids carry a per-page
occurrence number), so repeated short reviews like "good" all count.
"""

import pandas as pd

from app.pipeline.io import REVIEW_COLUMNS

_HTML_TAG = r"<[^>]+>"


def clean_reviews(raw: pd.DataFrame) -> pd.DataFrame:
    df = raw.copy()

    df["raw_text"] = df["raw_text"].fillna(df["text"]).fillna("")
    df["text"] = (
        df["text"]
        .fillna("")
        .astype("string")
        .str.replace(_HTML_TAG, " ", regex=True)
        .str.normalize("NFKC")
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )
    df["rating"] = pd.to_numeric(df["rating"], errors="coerce").clip(1, 5).astype("Int64")
    df["review_date"] = pd.to_datetime(df["review_date"], utc=True, errors="coerce", format="ISO8601")

    df = df[(df["text"] != "") & df["rating"].notna()]
    df = df.drop_duplicates(subset=["review_id"], keep="first")
    return df[REVIEW_COLUMNS].reset_index(drop=True)
