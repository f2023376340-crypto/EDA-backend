"""POST /api/v1/analyze through the whole pipeline, with the scraper, Groq and
Jev replaced by fakes; run files go to a temp folder.
"""

import json
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
from tests.jev_fakes import answers, fake_jev_client

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
        overall = {"good taste, fresh": 1.8, "packaging bad": 0.2, "ok": 1.1}
        # Category order follows the generated FeatureSet: 0 Taste, 1 Freshness, 2 Packaging, 3 Price.
        categories = {"good taste, fresh": {0: (0.95, 1.9), 1: (0.8, 1.6)}, "packaging bad": {2: (0.97, 0.3)}}
        yield fake_jev_client(answers(lambda text: overall[text], categories))

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
    features = {f["name"]: f for f in body["features"]}
    assert [f["name"] for f in body["features"]] == ["Taste", "Freshness", "Packaging", "Price", "Overall"]
    assert (features["Packaging"]["mention_count"], features["Packaging"]["avg_score"]) == (1, 0.3)
    assert features["Price"]["mention_count"] == 0
    assert features["Overall"]["mention_count"] == 1  # "ok" mentions nothing specific
    assert [f["name"] for f in body["reviews"][0]["features"]] == ["Taste", "Freshness"]
    assert body["sentiment"] == {"positive": 1, "neutral": 1, "negative": 1}
    assert body["avg_score"] == pytest.approx((1.8 + 0.2 + 1.1) / 3, abs=0.01)
    assert [r["score"] for r in body["reviews"]] == [1.8, 0.2, 1.1]
    assert body["stats"]["reviews_scored"] == 3
    assert body["stats"]["jev_model"] == "jev-1.14.0"

    [run_dir] = list(fakes.iterdir())
    for name in (
        "product.json",
        "run_meta.json",
        "raw_reviews.csv",
        "clean_reviews.csv",
        "features.json",
        "jev_scores.csv",
        "jev_category_scores.csv",
        "analysis.json",
    ):
        assert (run_dir / name).exists(), name
    assert len(io.read_category_scores(run_dir)) == 3 * 4  # 3 reviews x 4 categories (Overall not asked)


def _sse(text: str) -> list[tuple[str, dict[str, Any]]]:
    events = []
    for block in text.strip().split("\n\n"):
        lines = dict(line.split(": ", 1) for line in block.splitlines() if not line.startswith(":"))
        if "event" in lines:
            events.append((lines["event"], json.loads(lines["data"])))
    return events


def test_stream_reports_every_stage_then_the_result(fakes: Path) -> None:
    response = client.get("/api/v1/analyze/stream", params={"product_url": URL})

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    events = _sse(response.text)
    names = [name for name, _ in events]
    stages = [data["stage"] for name, data in events if name == "stage"]
    assert stages == ["validating", "scraping", "cleaning", "categorizing", "scoring", "compiling"]
    assert names.index("cleaned") < names.index("categories") < names.index("scored")
    assert names[-1] == "done"

    scored = [data for name, data in events if name == "scored"]
    assert [s["done"] for s in scored] == [1, 2, 3] and {s["total"] for s in scored} == {3}
    packaging = next(s for s in scored if s["text"] == "packaging bad")
    assert (packaging["score"], packaging["sentiment"]) == (0.2, "negative")
    assert packaging["features"] == [{"name": "Packaging", "score": 0.3}]
    scale = next(data for name, data in events if name == "scale")
    assert (scale["max_level"], scale["labels"]) == (2, ["Negative", "Neutral", "Positive"])
    assert names.index("scale") < names.index("scored")
    categories = next(data for name, data in events if name == "categories")
    assert [f["name"] for f in categories["features"]][-1] == "Overall"
    assert events[-1][1]["stats"]["reviews_scored"] == 3


def test_stream_reports_errors_as_events(fakes: Path) -> None:
    response = client.get(
        "/api/v1/analyze/stream", params={"product_url": "https://www.daraz.lk/products/x-i1.html"}
    )
    events = _sse(response.text)
    assert [name for name, _ in events] == ["stage", "failed"]
    assert events[-1][1]["code"] == "INVALID_URL"


def test_unexpected_crash_returns_json_with_cors(fakes: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    async def boom(*_: Any, **__: Any) -> ScrapeResult:
        raise NotImplementedError("subprocess unsupported")

    monkeypatch.setattr(orchestrator.scraper, "scrape", boom)
    with TestClient(app, raise_server_exceptions=False) as c:
        response = c.post(
            "/api/v1/analyze", json={"product_url": URL}, headers={"Origin": "http://localhost:3000"}
        )
    assert response.status_code == 500
    assert response.json()["error"]["code"] == "INTERNAL_ERROR"
    assert response.headers["access-control-allow-origin"] == "http://localhost:3000"


def test_saved_runs_are_listed_and_served(fakes: Path) -> None:
    run_id = client.post("/api/v1/analyze", json={"product_url": URL}).json()["run_id"]
    stale = fakes / "111_20250101-000000"
    stale.mkdir()
    (stale / "analysis.json").write_text('{"categories": []}', encoding="utf-8")

    listing = client.get("/api/v1/runs")
    assert listing.status_code == 200
    [summary] = listing.json()  # the old-format run is skipped
    assert summary["run_id"] == run_id
    assert summary["product_name"] == "Sensodyne Original Toothpaste 70gm"
    assert summary["reviews_scored"] == 3

    run = client.get(f"/api/v1/runs/{run_id}")
    assert run.status_code == 200
    assert run.json()["run_id"] == run_id

    assert client.get("/api/v1/runs/111_20250101-000000").json()["error"]["code"] == "RUN_NOT_FOUND"


@pytest.mark.parametrize(
    "run_id", ["nope", "..", "..%2F..%2Fetc", "496954821_20261016-999999", "1_2026101-000000"]
)
def test_unknown_or_malformed_run_ids_are_not_found(fakes: Path, run_id: str) -> None:
    response = client.get(f"/api/v1/runs/{run_id}")
    assert response.status_code == 404
