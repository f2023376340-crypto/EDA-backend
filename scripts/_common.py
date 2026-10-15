"""Shared CLI plumbing for the stage scripts."""

import argparse
from pathlib import Path

from app.pipeline import io


def run_dir_parser(description: str | None) -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=description)
    ap.add_argument("--run", required=True, type=Path, help="run folder, e.g. data/runs/<item_id>_<ts>")
    return ap


def resolve_run_dir(path: Path) -> Path:
    for candidate in (path, io.RUNS_DIR / path.name):
        if (candidate / "product.json").exists():
            return candidate
    raise SystemExit(f"Not a run folder (no product.json): {path}")
