"""Run the pipeline across multiple OULAD presentations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from . import pipeline
from .config import DEMO_CACHE, TABLES, ensure_dirs, raw_data_dir


DEFAULT_PRESENTATIONS = ["AAA_2013J", "AAA_2014J", "BBB_2014J", "CCC_2014J"]


def available_presentations() -> list[str]:
    info = pd.read_csv(raw_data_dir() / "studentInfo.csv", usecols=["code_module", "code_presentation"])
    pairs = info.drop_duplicates().sort_values(["code_module", "code_presentation"])
    return [f"{row.code_module}_{row.code_presentation}" for row in pairs.itertuples(index=False)]


def _load_meta() -> dict:
    path = DEMO_CACHE / "meta.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def run_many(
    presentations: list[str],
    bootstrap_b: int,
    sample_n: int | None,
    n_jobs: int,
    random_baselines: int,
) -> pd.DataFrame:
    ensure_dirs()
    rows = []
    for presentation in presentations:
        print(f"\n=== {presentation} ===")
        result = pipeline.run(
            presentation=presentation,
            bootstrap_b=bootstrap_b,
            sample_n=sample_n,
            n_jobs=n_jobs,
            random_baselines=random_baselines,
        )
        winner = result["winner"]
        group_metrics = result["group_metrics"].set_index("strategy")
        meta = _load_meta()
        mode_b = group_metrics.loc["mode_b"].to_dict() if "mode_b" in group_metrics.index else {}
        rows.append(
            {
                "presentation": presentation,
                "n_learners": meta.get("n_learners"),
                "n_features": meta.get("n_features"),
                "winner_config": winner.get("config_id"),
                "winner_reducer": winner.get("reducer"),
                "winner_clusterer": winner.get("clusterer"),
                "winner_k": winner.get("k"),
                "winner_bootstrap_ari": winner.get("bootstrap_ari_mean"),
                "mode_b_complementarity": mode_b.get("complementarity"),
                "mode_b_cluster_coverage": mode_b.get("cluster_coverage"),
                "mode_b_outcome_diversity": mode_b.get("outcome_diversity"),
                "mode_b_high_risk_group_rate": mode_b.get("high_risk_group_rate"),
            }
        )
    summary = pd.DataFrame(rows)
    summary.to_csv(TABLES / "multi_presentation_summary.csv", index=False)
    summary.to_parquet(TABLES / "multi_presentation_summary.parquet", index=False)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run pipeline across multiple OULAD presentations")
    parser.add_argument(
        "--presentations",
        nargs="*",
        default=None,
        help="Presentations such as AAA_2014J BBB_2014J. Defaults to a small representative set.",
    )
    parser.add_argument("--list", action="store_true", help="List available presentations and exit")
    parser.add_argument("--bootstrap-b", type=int, default=5)
    parser.add_argument("--sample-n", type=int, default=300)
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument("--random-baselines", type=int, default=20)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    available = available_presentations()
    if args.list:
        print("\n".join(available))
        return
    presentations = args.presentations or [p for p in DEFAULT_PRESENTATIONS if p in available]
    if not presentations:
        presentations = available[:4]
    summary = run_many(
        presentations,
        bootstrap_b=args.bootstrap_b,
        sample_n=args.sample_n,
        n_jobs=args.n_jobs,
        random_baselines=args.random_baselines,
    )
    print("\nSummary")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
