"""raw_reviews.csv -> clean_reviews.csv.

uv run python -m scripts.run_clean --run data/runs/<run>
"""

from app.pipeline import orchestrator
from scripts._common import resolve_run_dir, run_dir_parser


def main() -> None:
    run_dir = resolve_run_dir(run_dir_parser(__doc__).parse_args().run)
    clean = orchestrator.stage_clean(run_dir)
    print(f"{run_dir.name}: {len(clean)} clean reviews -> clean_reviews.csv")


if __name__ == "__main__":
    main()
