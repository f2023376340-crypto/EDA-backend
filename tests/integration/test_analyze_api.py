"""POST /api/v1/analyze through the whole pipeline, with the scraper, Groq and
Jev replaced by fakes; run files go to a temp folder.
"""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient
from langchain_core.runnables import RunnableLambda

from app.llm import feature_generator
from app.llm.jev import runner
from app.llm.jev.client import JEVClient
from app.main import app
from app.models.domain import RawProduct, RawReview, ScrapeResult
from app.models.jev import Feature, FeatureSet
from app.pipeline import io, orchestrator
from tests.jev_fakes import fake_jev_client

URL = "https://www.daraz.pk/products/sensodyne-original-toothpaste-70gm-i496954821-s2379054862.html"
client = TestClient(app)


def test_invalid_url_returns_error_body() -> None:
    response = client.post("/api/v1/analyze", json={"product_url": "https://www.daraz.lk/products/x-i1.html"})
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "INVALID_URL"


class _FakeGroq:
    def with_structured_output(self, _schema: type) -> RunnableLambda[Any, Any]:
        features = [
            Feature(name=n, description=f"{n} desc") for n in ("Taste", "Freshness", "Packaging", "Price")
        ]
        return RunnableLambda(lambda _: FeatureSet(product_category="Toothpaste", features=features))


@pytest.fixture
def fakes(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    reviews = [
        RawReview(review_id="1", rating=5, text="good taste, fresh", raw_text="good taste, fresh"),
        RawReview(review_id="2", rating=2, text="packaging bad", raw_text="packaging bad"),
        RawReview(
            review_id="3",
            rating=4,
            text="ok",
            raw_text="ok",
            review_date=datetime(2026, 10, 1, tzinfo=UTC),
        ),
    ]
    product = RawProduct(
        item_id="496954821",
        url=URL,
        product_name="Sensodyne Original Toothpaste 70gm",
        category_breadcrumb="Health & Beauty > Oral Care > Toothpaste",
        price=378.0,
        overall_rating=4.9,
        total_rating_count=169,
    )

    async def fake_scrape(*_: Any, **__: Any) -> ScrapeResult:
        return ScrapeResult(product=product, reviews=reviews)

    @asynccontextmanager
    async def fake_jev() -> AsyncIterator[JEVClient]:
        yield fake_jev_client(lambda text: {"good taste, fresh": 3.8, "packaging bad": 1.0, "ok": 2.2}[text])

    monkeypatch.setattr(orchestrator.scraper, "scrape", fake_scrape)
    monkeypatch.setattr(feature_generator, "get_chat_model", lambda *_a, **_k: _FakeGroq())
    monkeypatch.setattr(runner, "jev_client", fake_jev)
    monkeypatch.setattr(io, "RUNS_DIR", tmp_path)
    feature_generator._cache.clear()
    return tmp_path


def test_analyze_runs_the_full_pipeline(fakes: Path) -> None:
    response = client.post("/api/v1/analyze", json={"product_url": URL})

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["product"]["name"] == "Sensodyne Original Toothpaste 70gm"
    assert [c["name"] for c in body["categories"]] == ["Taste", "Freshness", "Packaging", "Price", "Overall"]
    assert body["sentiment"] == {"positive": 1, "neutral": 1, "negative": 1}
    assert body["avg_score"] == pytest.approx((3.8 + 1.0 + 2.2) / 3, abs=0.01)
    assert [r["score"] for r in body["reviews"]] == [3.8, 1.0, 2.2]
    assert body["stats"]["reviews_scored"] == 3
    assert body["stats"]["jev_model"] == "jev-1.14.0"

    [run_dir] = list(fakes.iterdir())
    for name in ("product.json", "raw_reviews.csv", "clean_reviews.csv", "features.json", "jev_scores.csv"):
        assert (run_dir / name).exists(), name
    assert (run_dir / "analysis.json").exists()
