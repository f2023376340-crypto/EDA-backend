from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_invalid_url_returns_error_body() -> None:
    response = client.post("/api/v1/analyze", json={"product_url": "https://www.daraz.lk/products/x-i1.html"})
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "INVALID_URL"
