"""jev_scores.csv + clean_reviews.csv -> analysis.json.

uv run python -m scripts.run_aggregate --run data/runs/<run>
"""

from app.pipeline import orchestrator
from scripts._common import resolve_run_dir, run_dir_parser


def main() -> None:
    run_dir = resolve_run_dir(run_dir_parser(__doc__).parse_args().run)
    r = orchestrator.stage_aggregate(run_dir)
    s = r.sentiment
    print(f"{r.product.name}: {r.stats.reviews_scored}/{r.stats.reviews_analysed} reviews scored")
    print(f"avg score {r.avg_score} / {r.scale.max_level}   net sentiment {r.net_sentiment}")
    print(f"positive {s.positive}  neutral {s.neutral}  negative {s.negative}")
    print("levels: " + "  ".join(f"{lv.label} {lv.count}" for lv in r.levels))
    for w in r.recency:
        label = f"{w.days}d" if w.days else "all"
        print(f"  {label:>4}: {w.review_count:3} reviews  avg score {w.avg_score}  net {w.net_sentiment}")
    print(f"top terms: {', '.join(f'{t.term}({t.review_count})' for t in r.terms[:10])}")
    print(f"-> {run_dir / 'analysis.json'}")


if __name__ == "__main__":
    main()
