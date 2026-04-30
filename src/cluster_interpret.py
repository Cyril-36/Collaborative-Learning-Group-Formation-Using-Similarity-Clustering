"""Cluster characterization helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd


ENGAGEMENT_FEATURES = {
    "total_clicks",
    "active_days",
    "mean_clicks_per_active_day",
    "engagement_span",
}
COLLAB_FEATURES = {
    "collaborative_clicks",
    "collaboration_click_ratio",
    "forum_clicks",
    "live_collab_clicks",
    "collaborative_active_days",
}
PERFORMANCE_FEATURES = {
    "weighted_score",
    "mean_tma_score",
    "n_assessments_submitted",
}
RISK_FEATURES = {
    "no_submissions",
    "mean_submission_lateness",
    "num_prev_attempts",
}


def canonical_cluster_order(
    X_scaled: np.ndarray,
    labels: np.ndarray,
    feature_columns: list[str],
    primary: str = "total_clicks",
    secondary: str = "active_days",
) -> dict[int, int]:
    """Return ``{raw_cluster_id: canonical_id}`` sorted by *primary* feature desc.

    Canonical ordering: cluster 0 = highest mean(primary), tie-broken by
    mean(secondary).  Stable across K-Means re-initialisations given the same
    scaled feature matrix — eliminates the "cluster 0 is sometimes at-risk,
    sometimes high-engagement" problem.

    Noise label ``-1`` (HDBSCAN) is preserved as-is.
    """
    primary_idx = feature_columns.index(primary)
    secondary_idx = feature_columns.index(secondary)

    valid = labels != -1
    df = pd.DataFrame({
        "cluster":   labels[valid],
        "primary":   X_scaled[valid, primary_idx],
        "secondary": X_scaled[valid, secondary_idx],
    })
    means = df.groupby("cluster").agg({"primary": "mean", "secondary": "mean"})
    ordered = means.sort_values(
        ["primary", "secondary"], ascending=[False, False], kind="stable",
    ).index
    return {int(raw): canon for canon, raw in enumerate(ordered)}


def apply_remap(labels: np.ndarray, remap: dict[int, int]) -> np.ndarray:
    """Apply a cluster-ID remap, preserving noise (``-1``) unchanged."""
    return np.array([remap.get(int(l), int(l)) for l in labels])


def _interpret_label(top_positive: list[str], top_negative: list[str]) -> str:
    pos = set(top_positive)
    neg = set(top_negative)
    if pos & COLLAB_FEATURES:
        return "High-collaboration learners"
    if pos & ENGAGEMENT_FEATURES and pos & PERFORMANCE_FEATURES:
        return "Highly engaged high performers"
    if pos & ENGAGEMENT_FEATURES:
        return "High-engagement learners"
    if pos & PERFORMANCE_FEATURES:
        return "High-performing learners"
    if neg & ENGAGEMENT_FEATURES or pos & RISK_FEATURES:
        return "At-risk low-engagement learners"
    return "Mixed-profile learners"


def characterize_clusters(
    X_scaled: np.ndarray,
    labels: np.ndarray,
    columns: list[str],
    top_n: int = 3,
) -> pd.DataFrame:
    """Summarize each non-noise cluster by strongest standardized feature deviations."""
    labels = np.asarray(labels)
    rows = []
    for cluster in sorted(set(labels.tolist()) - {-1}):
        mask = labels == cluster
        means = X_scaled[mask].mean(axis=0)
        pos_idx = np.argsort(means)[::-1][:top_n]
        neg_idx = np.argsort(means)[:top_n]
        top_positive = [columns[i] for i in pos_idx]
        top_negative = [columns[i] for i in neg_idx]
        rows.append(
            {
                "cluster": int(cluster),
                "size": int(mask.sum()),
                "top_positive_features": ", ".join(
                    f"{columns[i]} ({means[i]:+.2f}z)" for i in pos_idx
                ),
                "top_negative_features": ", ".join(
                    f"{columns[i]} ({means[i]:+.2f}z)" for i in neg_idx
                ),
                "interpretive_label": _interpret_label(top_positive, top_negative),
            }
        )
    return pd.DataFrame(rows)
