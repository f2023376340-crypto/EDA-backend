"""clean_reviews rows -> JEVReviewInput (CLAUDE.md §5.2)."""

import pandas as pd

from app.models.jev import JEVReviewInput

# Jev switches to a slower, less accurate long-context mode above ~32 KB of
# text; 8,000 chars stays under that even for 3-byte Urdu characters.
MAX_REVIEW_CHARS = 8000


def _truncate(text: str) -> str:
    return text if len(text) <= MAX_REVIEW_CHARS else text[: MAX_REVIEW_CHARS - 1].rstrip() + "…"


def reviews_to_jev_inputs(clean: pd.DataFrame) -> list[JEVReviewInput]:
    rows = clean.rename(columns={"rating": "star_rating"})[["review_id", "star_rating", "text"]]
    return [
        JEVReviewInput.model_validate(row | {"text": _truncate(str(row["text"]))})
        for row in rows.to_dict("records")
    ]
