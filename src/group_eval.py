"""Group-level evaluation metrics."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist

from .adapters.base import DatasetSchema


OUTCOME_SCORE = {
    "Withdrawn": 0.0,
    "Fail": 1.0,
    "Pass": 2.0,
    "Distinction": 3.0,
}
AT_RISK_OUTCOMES = {"Withdrawn", "Fail"}


def intra_group_distance(X_red: np.ndarray, groups: list[list[int]]) -> float:
    distances = [pdist(X_red[group]).mean() for group in groups if len(group) >= 2]
    return float(np.mean(distances)) if distances else 0.0


def inter_group_variance(X_red: np.ndarray, groups: list[list[int]]) -> float:
    valid = [group for group in groups if group]
    if not valid:
        return 0.0
    centroids = np.array([X_red[group].mean(axis=0) for group in valid])
    return float(centroids.var(axis=0).sum())


def complementarity(labels: np.ndarray, groups: list[list[int]], G: int) -> float:
    labels = np.asarray(labels)
    values = []
    for group in groups:
        if not group:
            continue
        values.append(len(set(labels[group].tolist()) - {-1}) / max(1, min(G, len(group))))
    return float(np.mean(values)) if values else 0.0


def engagement_balance(engage_col: np.ndarray, groups: list[list[int]]) -> float:
    engage_col = np.asarray(engage_col, dtype=float)
    sd = float(np.nanstd(engage_col))
    if sd == 0 or np.isnan(sd):
        return 0.0
    mu = float(np.nanmean(engage_col))
    devs = [abs(np.nanmean(engage_col[group]) - mu) / sd for group in groups if group]
    return float(np.mean(devs)) if devs else 0.0


def demographic_fairness(attr_col: pd.Series | np.ndarray, groups: list[list[int]]) -> float:
    attr = pd.Series(attr_col).reset_index(drop=True)
    class_dist = attr.value_counts(normalize=True)
    tvs = []
    for group in groups:
        if not group:
            continue
        group_dist = attr.iloc[group].value_counts(normalize=True)
        keys = set(class_dist.index) | set(group_dist.index)
        tv = 0.5 * sum(abs(class_dist.get(key, 0.0) - group_dist.get(key, 0.0)) for key in keys)
        tvs.append(tv)
    return float(np.mean(tvs)) if tvs else 0.0


def cluster_coverage(labels: np.ndarray, groups: list[list[int]]) -> float:
    labels = np.asarray(labels)
    n_clusters = len(set(labels.tolist()) - {-1})
    if n_clusters == 0:
        return 0.0
    values = [len(set(labels[group].tolist()) - {-1}) / n_clusters for group in groups if group]
    return float(np.mean(values)) if values else 0.0


def outcome_diversity(outcomes: pd.Series, groups: list[list[int]], G: int) -> float:
    outcomes = outcomes.reset_index(drop=True).fillna("Unknown")
    n_outcomes = max(1, outcomes.nunique())
    denom = max(1, min(G, n_outcomes))
    values = [outcomes.iloc[group].nunique() / denom for group in groups if group]
    return float(np.mean(values)) if values else 0.0


def at_risk_concentration(outcomes: pd.Series, groups: list[list[int]]) -> float:
    outcomes = outcomes.reset_index(drop=True).fillna("Unknown")
    values = []
    for group in groups:
        if not group:
            continue
        group_outcomes = outcomes.iloc[group]
        values.append(float(group_outcomes.isin(AT_RISK_OUTCOMES).mean()))
    return float(np.mean(values)) if values else 0.0


def high_risk_group_rate(outcomes: pd.Series, groups: list[list[int]], threshold: float = 0.5) -> float:
    outcomes = outcomes.reset_index(drop=True).fillna("Unknown")
    rates = []
    for group in groups:
        if not group:
            continue
        rates.append(float(outcomes.iloc[group].isin(AT_RISK_OUTCOMES).mean() > threshold))
    return float(np.mean(rates)) if rates else 0.0


def outcome_balance(outcomes: pd.Series, groups: list[list[int]]) -> float:
    outcomes = outcomes.reset_index(drop=True)
    if pd.api.types.is_numeric_dtype(outcomes):
        scores = pd.to_numeric(outcomes, errors="coerce").astype(float)
    else:
        scores = outcomes.map(OUTCOME_SCORE).astype(float)
        if scores.isna().all():
            return 0.0
    scores = scores.fillna(scores.median())
    sd = float(scores.std(ddof=0))
    if sd == 0 or np.isnan(sd):
        return 0.0
    mu = float(scores.mean())
    devs = [abs(float(scores.iloc[group].mean()) - mu) / sd for group in groups if group]
    return float(np.mean(devs)) if devs else 0.0


def evaluate_all(
    X_red: np.ndarray,
    labels: np.ndarray,
    groups: list[list[int]],
    feature_df: pd.DataFrame,
    G: int,
    schema: DatasetSchema | None = None,
    attr: str | None = None,
) -> dict[str, float]:
    metrics = {
        "intra_group_distance": intra_group_distance(X_red, groups),
        "inter_group_variance": inter_group_variance(X_red, groups),
        "complementarity": complementarity(labels, groups, G),
        "cluster_coverage": cluster_coverage(labels, groups),
    }

    engagement_col = schema.engagement_col if schema else None
    if engagement_col and engagement_col in feature_df.columns:
        metrics["engagement_balance"] = engagement_balance(feature_df[engagement_col].to_numpy(), groups)

    fairness_cols = list(schema.fairness_cols) if schema else ([attr] if attr else [])
    fairness_values = []
    for fairness_col in fairness_cols:
        if fairness_col in feature_df.columns:
            value = demographic_fairness(feature_df[fairness_col], groups)
            metrics[f"demographic_fairness_{fairness_col}"] = value
            fairness_values.append(value)
    if fairness_values:
        metrics["demographic_fairness"] = float(np.mean(fairness_values))

    outcome_col = schema.outcome_col if schema else None
    if outcome_col and outcome_col in feature_df.columns:
        outcomes = feature_df[outcome_col]
        metrics.update(
            {
                "outcome_diversity": outcome_diversity(outcomes, groups, G),
                "at_risk_concentration": at_risk_concentration(outcomes, groups),
                "high_risk_group_rate": high_risk_group_rate(outcomes, groups),
                "outcome_balance": outcome_balance(outcomes, groups),
            }
        )
    return metrics
