"""Random-baseline significance checks for group-quality metrics."""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import group_eval, group_former
from .config import GROUP_SIZE, SEED


HIGHER_IS_BETTER = {"complementarity", "cluster_coverage", "outcome_diversity"}
LOWER_IS_BETTER = {
    "inter_group_variance",
    "engagement_balance",
    "demographic_fairness",
    "at_risk_concentration",
    "high_risk_group_rate",
    "outcome_balance",
}


def preferred_direction(strategy: str, metric: str) -> str:
    if metric == "intra_group_distance":
        return "lower" if strategy == "mode_a" else "higher"
    if metric in HIGHER_IS_BETTER:
        return "higher"
    if metric in LOWER_IS_BETTER:
        return "lower"
    return "two-sided"


def random_baseline_distribution(
    X_red: np.ndarray,
    labels: np.ndarray,
    feature_df: pd.DataFrame,
    G: int = GROUP_SIZE,
    n_runs: int = 100,
    seed: int = SEED,
) -> pd.DataFrame:
    rows = []
    for run_idx in range(n_runs):
        groups = group_former.form_random(len(feature_df), G, seed + run_idx)
        metrics = group_eval.evaluate_all(X_red, labels, groups, feature_df, G)
        metrics["baseline_run"] = run_idx
        rows.append(metrics)
    return pd.DataFrame(rows)


def compare_to_random(
    strategy_metrics: pd.DataFrame,
    random_distribution: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    strategies = strategy_metrics[strategy_metrics["strategy"].isin(["mode_a", "mode_b"])]
    metric_cols = [
        col
        for col in random_distribution.columns
        if col != "baseline_run" and pd.api.types.is_numeric_dtype(random_distribution[col])
    ]
    for _, strategy_row in strategies.iterrows():
        strategy = strategy_row["strategy"]
        for metric in metric_cols:
            value = float(strategy_row[metric])
            dist = random_distribution[metric].dropna().to_numpy(dtype=float)
            if len(dist) == 0:
                continue
            percentile = float((dist <= value).mean())
            p_greater = float((np.count_nonzero(dist >= value) + 1) / (len(dist) + 1))
            p_less = float((np.count_nonzero(dist <= value) + 1) / (len(dist) + 1))
            direction = preferred_direction(strategy, metric)
            if direction == "higher":
                preferred_p = p_greater
            elif direction == "lower":
                preferred_p = p_less
            else:
                preferred_p = min(p_greater, p_less)
            rows.append(
                {
                    "strategy": strategy,
                    "metric": metric,
                    "value": value,
                    "random_mean": float(dist.mean()),
                    "random_std": float(dist.std(ddof=0)),
                    "random_percentile": percentile,
                    "p_greater_or_equal": p_greater,
                    "p_less_or_equal": p_less,
                    "preferred_direction": direction,
                    "preferred_p_value": preferred_p,
                    "n_random_runs": int(len(dist)),
                }
            )
    return pd.DataFrame(rows)
