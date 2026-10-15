"""FastAPI app factory.

Bootstrap stage: only a health check is wired up. Routers for
`/api/v1/analyze` and friends are added in a later session.
"""

from fastapi import FastAPI

app = FastAPI(title="Daraz AI Review Analyzer")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}
