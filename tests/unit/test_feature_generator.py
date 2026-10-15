from typing import Any

import pytest
from langchain_core.runnables import RunnableLambda

from app.llm import feature_generator as fg
from app.models.domain import RawProduct
from app.models.jev import Feature, FeatureSet, JEVReviewInput


def _product(item_id: str = "111") -> RawProduct:
    return RawProduct(
        item_id=item_id,
        url=f"https://www.daraz.pk/products/x-i{item_id}.html",
        product_name="Redmi Note 13",
        category_breadcrumb="Electronic Devices > Mobiles > Smartphones",
    )


def _reviews() -> list[JEVReviewInput]:
    return [JEVReviewInput(review_id="r1", star_rating=5, text="great camera")]


def _feature_set(*names: str) -> FeatureSet:
    return FeatureSet(
        product_category="Smartphone",
        features=[Feature(name=n, description=f"{n} desc") for n in names],
    )


class _FakeLLM:
    """Stands in for ChatGroq: returns queued outputs from with_structured_output."""

    def __init__(self, outputs: list[Any]) -> None:
        self.outputs = outputs
        self.prompts: list[str] = []

    def with_structured_output(self, _schema: type) -> RunnableLambda[Any, Any]:
        def _call(prompt_value: Any) -> Any:
            self.prompts.append(prompt_value.to_string())
            out = self.outputs.pop(0)
            if isinstance(out, Exception):
                raise out
            return out

        return RunnableLambda(_call)


@pytest.fixture(autouse=True)
def _clear_cache() -> None:
    fg._cache.clear()


def _install(monkeypatch: pytest.MonkeyPatch, llm: _FakeLLM) -> None:
    monkeypatch.setattr(fg, "get_chat_model", lambda *_a, **_k: llm)


async def test_appends_overall_and_drops_model_overall_and_duplicates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    llm = _FakeLLM([_feature_set("Camera", "Battery", "camera", "Overall", "Display", "Performance")])
    _install(monkeypatch, llm)

    outcome = await fg.generate_features(_product(), _reviews())

    names = [f.name for f in outcome.feature_set.features]
    assert names == ["Camera", "Battery", "Display", "Performance", "Overall"]
    assert outcome.used_fallback is False


async def test_repairs_once_with_error_fed_back(monkeypatch: pytest.MonkeyPatch) -> None:
    llm = _FakeLLM(
        [
            _feature_set("Camera"),
            _feature_set("Camera", "Battery", "Display", "Performance", "Value for Money"),
        ]
    )
    _install(monkeypatch, llm)

    outcome = await fg.generate_features(_product(), _reviews())

    assert len(outcome.feature_set.features) == 6
    assert outcome.used_fallback is False
    assert "previous answer was invalid" in llm.prompts[1]


async def test_falls_back_after_failed_repair(monkeypatch: pytest.MonkeyPatch) -> None:
    llm = _FakeLLM([_feature_set("Camera"), _feature_set("Camera", "Battery")])
    _install(monkeypatch, llm)

    outcome = await fg.generate_features(_product(), _reviews())

    assert outcome.used_fallback is True
    assert outcome.feature_set.product_category == "Smartphones"
    assert [f.name for f in outcome.feature_set.features][-1] == "Overall"


async def test_caches_by_item_id(monkeypatch: pytest.MonkeyPatch) -> None:
    llm = _FakeLLM([_feature_set("Camera", "Battery", "Display", "Performance")])
    _install(monkeypatch, llm)

    first = await fg.generate_features(_product("222"), _reviews())
    second = await fg.generate_features(_product("222"), _reviews())

    assert first == second
    assert len(llm.prompts) == 1


def test_review_sample_tolerates_braces() -> None:
    sample = fg._build_review_sample([JEVReviewInput(review_id="r", star_rating=4, text="{weird} text")])
    assert "{weird}" in sample
