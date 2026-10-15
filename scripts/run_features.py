"""clean_reviews.csv + product.json -> features.json (one Groq call).

uv run python -m scripts.run_features --run data/runs/<run>
"""

import asyncio

from app.pipeline import orchestrator
from scripts._common import resolve_run_dir, run_dir_parser


def main() -> None:
    run_dir = resolve_run_dir(run_dir_parser(__doc__).parse_args().run)
    outcome = asyncio.run(orchestrator.stage_features(run_dir))
    if outcome is None:
        print("No clean reviews; nothing to generate.")
        return
    print(f"{outcome.feature_set.product_category} (fallback={outcome.used_fallback}):")
    for f in outcome.feature_set.features:
        print(f"  - {f.name}: {f.description}")


if __name__ == "__main__":
    main()
