"""clean_reviews.csv + features.json -> jev_scores.csv + jev_category_scores.csv
(one Jev request per review: overall Score + per-category Noul and Score).

uv run python -m scripts.run_jev --run data/runs/<run> --limit 30
"""

import asyncio

from app.llm.jev.client import load_rubric
from app.pipeline import io, orchestrator
from scripts._common import resolve_run_dir, run_dir_parser


def main() -> None:
    ap = run_dir_parser(__doc__)
    ap.add_argument(
        "--limit", type=int, default=None, help="only the first N clean reviews (cheap test runs)"
    )
    args = ap.parse_args()
    run_dir = resolve_run_dir(args.run)

    stats = asyncio.run(orchestrator.stage_jev(run_dir, limit=args.limit))
    scores = io.read_scores(run_dir)
    labels = load_rubric().labels
    print(f"{scores['score'].notna().sum()}/{len(scores)} reviews scored -> jev_scores.csv")
    print(f"mean score {scores['score'].mean():.2f} (0 = {labels[0]}, {len(labels) - 1} = {labels[-1]})")
    by_level = scores["level"].dropna().astype(int).value_counts().sort_index()
    print("  ".join(f"{labels[int(str(lv))]}: {n}" for lv, n in by_level.items()))

    categories = io.read_category_scores(run_dir)
    mentioned = categories[categories["mentioned"].fillna(False).astype(bool)]
    summary = mentioned.groupby("feature").agg(mentions=("review_id", "size"), avg_score=("score", "mean"))
    print("\nper category (mentions, avg score) -> jev_category_scores.csv")
    print(summary.sort_values("mentions", ascending=False).round(2).to_string())
    if stats:
        print(
            f"requests={stats.requests} failed={stats.failed} "
            f"input_tokens={stats.input_tokens} elapsed={stats.elapsed_s:.1f}s"
        )


if __name__ == "__main__":
    main()
