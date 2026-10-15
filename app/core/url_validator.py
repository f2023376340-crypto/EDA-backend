"""Daraz.pk product URL validation (§6.2).

Authoritative backend check. Accepts only a Daraz.pk product-detail-page URL,
extracts item_id/sku_id, and rebuilds a canonical URL from those extracted
parts rather than trusting the raw input string — this is what lets callers
fetch the result without an SSRF risk from query params, fragments, or odd
host casing/encoding in the original URL.
"""

import re
from urllib.parse import urlsplit

from pydantic import BaseModel, ConfigDict

from app.core.errors import InvalidUrlError

_ALLOWED_HOSTS = {"daraz.pk", "www.daraz.pk", "m.daraz.pk"}

_PATH_RE = re.compile(r"^/products/(?P<slug>[\w\-]+)-i(?P<item_id>\d+)(?:-s(?P<sku_id>\d+))?\.html$")


class DarazProductRef(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    item_id: str
    sku_id: str | None = None
    slug: str
    canonical_url: str


def validate_daraz_url(raw_url: str) -> DarazProductRef:
    """Parse and validate a user-supplied URL, raising InvalidUrlError on anything
    that isn't a genuine Daraz.pk product page.
    """
    raw_url = raw_url.strip()
    try:
        parts = urlsplit(raw_url)
    except ValueError as exc:
        raise InvalidUrlError("Could not parse URL.", details={"url": raw_url}) from exc

    if parts.scheme not in ("http", "https"):
        raise InvalidUrlError("URL must use http or https.", details={"scheme": parts.scheme})

    host = parts.hostname or ""
    if host not in _ALLOWED_HOSTS:
        raise InvalidUrlError("URL must be a daraz.pk product page.", details={"host": host})

    match = _PATH_RE.match(parts.path)
    if not match:
        raise InvalidUrlError(
            "URL does not look like a Daraz product page.",
            details={"path": parts.path},
        )

    item_id = match.group("item_id")
    sku_id = match.group("sku_id")
    slug = match.group("slug")

    canonical_url = f"https://www.daraz.pk/products/{slug}-i{item_id}"
    if sku_id:
        canonical_url += f"-s{sku_id}"
    canonical_url += ".html"

    return DarazProductRef(item_id=item_id, sku_id=sku_id, slug=slug, canonical_url=canonical_url)
