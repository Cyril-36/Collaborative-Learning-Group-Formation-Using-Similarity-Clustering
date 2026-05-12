"""Cluster characterization helpers (schema-driven)."""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

from .adapters.base import DatasetSchema


_DEVIATION_THRESHOLD = 0.25


def canonical_cluster_order(
    X_scaled: np.ndarray,
    labels: np.ndarray,
    feature_columns: list[str],
    primary: str | None = None,
    secondary: str | None = None,
) -> dict[int, int]:
    """Return ``{raw_cluster_id: canonical_id}`` sorted deterministically.

    When schema-selected ordering columns are available, cluster 0 has the
    highest mean(primary), tie-broken by mean(secondary). Otherwise, cluster 0
    is the largest cluster, tie-broken by centroid L2 norm. Noise label ``-1``
    is preserved as-is.
    """
    valid = labels != -1
    clusters = sorted(set(labels[valid].tolist()))
    if not clusters:
        return {}

    if primary and primary in feature_columns:
        primary_idx = feature_columns.index(primary)
        secondary_idx = (
            feature_columns.index(secondary)
            if secondary and secondary in feature_columns
            else primary_idx
        )
        df = pd.DataFrame({
            "cluster": labels[valid],
            "primary": X_scaled[valid, primary_idx],
            "secondary": X_scaled[valid, secondary_idx],
        })
        means = df.groupby("cluster").agg({"primary": "mean", "secondary": "mean"})
        ordered = means.sort_values(
            ["primary", "secondary"], ascending=[False, False], kind="stable",
        ).index
    else:
        rows = []
        for cluster in clusters:
            mask = labels == cluster
            centroid = X_scaled[mask].mean(axis=0)
            rows.append(
                {
                    "cluster": cluster,
                    "size": int(mask.sum()),
                    "centroid_norm": float(np.linalg.norm(centroid)),
                }
            )
        ordered = (
            pd.DataFrame(rows)
            .sort_values(["size", "centroid_norm", "cluster"], ascending=[False, False, True], kind="stable")
            ["cluster"]
        )
    return {int(raw): canon for canon, raw in enumerate(ordered)}


def apply_remap(labels: np.ndarray, remap: dict[int, int]) -> np.ndarray:
    """Apply a cluster-ID remap, preserving noise (``-1``) unchanged."""
    return np.array([remap.get(int(l), int(l)) for l in labels])


def _role_mean(
    means: np.ndarray, columns: list[str], role_cols: Iterable[str | None]
) -> float | None:
    """Mean standardized deviation across a role's columns; None if no columns present."""
    indices = [columns.index(c) for c in role_cols if c and c in columns]
    if not indices:
        return None
    return float(np.mean(means[indices]))


def _interpret_label(
    means: np.ndarray,
    columns: list[str],
    schema: DatasetSchema | None,
) -> str:
    """Build a label from schema role columns; fall back to z-score description."""
    if schema is None:
        return _generic_label(means)

    engagement_cols = [schema.engagement_col] if schema.engagement_col else []
    performance_cols = [schema.performance_col] if schema.performance_col else []

    engagement = _role_mean(means, columns, engagement_cols)
    performance = _role_mean(means, columns, performance_cols)

    parts: list[str] = []
    if engagement is not None:
        parts.append(_describe(engagement, "engagement"))
    if performance is not None:
        parts.append(_describe(performance, "performance"))

    if not parts:
        return _generic_label(means)

    parts = [p for p in parts if p]
    if not parts:
        return "Mixed-profile learners"
    return ", ".join(parts) + " learners"


def _describe(value: float, role: str) -> str:
    if value > _DEVIATION_THRESHOLD:
        return f"high-{role}"
    if value < -_DEVIATION_THRESHOLD:
        return f"low-{role}"
    return f"average-{role}"


def _generic_label(means: np.ndarray) -> str:
    spread = float(np.max(np.abs(means))) if means.size else 0.0
    if spread < _DEVIATION_THRESHOLD:
        return "Average-profile learners"
    return "Distinctive-profile learners"


def characterize_clusters(
    X_scaled: np.ndarray,
    labels: np.ndarray,
    columns: list[str],
    top_n: int = 3,
    schema: DatasetSchema | None = None,
) -> pd.DataFrame:
    """Summarize each non-noise cluster by strongest standardized feature deviations.

    When a ``schema`` is provided, the interpretive label is derived from the
    cluster's mean deviation along schema role columns (engagement, performance).
    Without a schema, a generic z-score-based label is used.
    """
    labels = np.asarray(labels)
    rows = []
    for cluster in sorted(set(labels.tolist()) - {-1}):
        mask = labels == cluster
        means = X_scaled[mask].mean(axis=0)
        pos_idx = np.argsort(means)[::-1][:top_n]
        neg_idx = np.argsort(means)[:top_n]
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
                "interpretive_label": _interpret_label(means, columns, schema),
            }
        )
    return pd.DataFrame(rows)
