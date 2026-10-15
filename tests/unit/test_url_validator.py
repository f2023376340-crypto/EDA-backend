import pytest

from app.core.errors import InvalidUrlError
from app.core.url_validator import validate_daraz_url

CANONICAL = "https://www.daraz.pk/products/redmi-note-13-i123456789-s987654321.html"


@pytest.mark.parametrize(
    ("url", "item_id", "sku_id"),
    [
        (CANONICAL, "123456789", "987654321"),
        ("https://www.daraz.pk/products/redmi-note-13-i123456789.html", "123456789", None),
        (CANONICAL + "?spm=a2a0e.searchlist&search=1", "123456789", "987654321"),
        (CANONICAL + "#reviews", "123456789", "987654321"),
        ("https://m.daraz.pk/products/redmi-note-13-i123456789-s987654321.html", "123456789", "987654321"),
        ("https://daraz.pk/products/redmi-note-13-i123456789-s987654321.html", "123456789", "987654321"),
        ("http://www.daraz.pk/products/redmi-note-13-i123456789-s987654321.html", "123456789", "987654321"),
        ("  " + CANONICAL + "  ", "123456789", "987654321"),
        ("https://WWW.DARAZ.PK/products/redmi-note-13-i123456789-s987654321.html", "123456789", "987654321"),
    ],
)
def test_valid_urls(url: str, item_id: str, sku_id: str | None) -> None:
    ref = validate_daraz_url(url)
    assert ref.item_id == item_id
    assert ref.sku_id == sku_id


def test_empty_slug_link_from_daraz() -> None:
    ref = validate_daraz_url(
        "https://www.daraz.pk/products/-i574837330-s2655076903.html?pvid=446cb448&scm=1007.51705&spm=a2a0e.tm80335142"
    )
    assert (ref.item_id, ref.sku_id) == ("574837330", "2655076903")
    assert ref.canonical_url == "https://www.daraz.pk/products/-i574837330-s2655076903.html"


def test_canonical_url_drops_query_fragment_and_normalizes_host() -> None:
    ref = validate_daraz_url("http://m.daraz.pk/products/redmi-note-13-i123456789-s987654321.html?spm=x#frag")
    assert ref.canonical_url == CANONICAL


@pytest.mark.parametrize(
    "url",
    [
        "",
        "not a url",
        "javascript:alert(1)",
        "ftp://www.daraz.pk/products/x-i123.html",
        "https://www.daraz.com.bd/products/x-i123456-s1.html",
        "https://www.daraz.lk/products/x-i123456-s1.html",
        "https://daraz.pk.evil.com/products/x-i123456-s1.html",
        "https://evil.com/products/x-i123456-s1.html?u=https://www.daraz.pk",
        "https://127.0.0.1/products/x-i123456-s1.html",
        "https://www.daraz.pk/catalog/?q=phone",
        "https://www.daraz.pk/smartphones/",
        "https://www.daraz.pk/products/redmi-note-13.html",
        "https://www.daraz.pk/products/redmi-note-13-s987654321.html",
        "https://www.daraz.pk/products/../admin-i123.html",
        "https://s.daraz.pk/s.abc123",
    ],
)
def test_invalid_urls(url: str) -> None:
    with pytest.raises(InvalidUrlError):
        validate_daraz_url(url)
