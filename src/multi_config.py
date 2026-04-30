"""Run and score the 3 reducer x 4 clusterer configuration grid."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score

from .clusterers import CLUSTERERS
from .reducers import REDUCERS


CONFIGS = [
    (f"C{(i * 4) + j + 1:02d}", reducer, clusterer)
    for i, reducer in enumerate(["pca", "umap", "identity"])
    for j, clusterer in enumerate(["kmeans", "gmm", "agglo", "hdbscan"])
]


def _dbcv(X_red: np.ndarray, labels: np.ndarray) -> float:
    try:
        import hdbscan

        return float(hdbscan.validity.validity_index(X_red.astype("float64"), labels))
    except Exception:
        return float("nan")


def score_labels(X_red: np.ndarray, labels: np.ndarray, clusterer_name: str) -> dict[str, float]:
    valid_mask = labels != -1
    valid_labels = labels[valid_mask]
    row: dict[str, float] = {}
    if valid_mask.sum() >= 3 and 1 < len(set(valid_labels.tolist())) < valid_mask.sum():
        row["silhouette"] = float(silhouette_score(X_red[valid_mask], valid_labels))
        row["davies_bouldin"] = float(davies_bouldin_score(X_red[valid_mask], valid_labels))
        row["calinski_harabasz"] = float(calinski_harabasz_score(X_red[valid_mask], valid_labels))
    else:
        row["silhouette"] = float("nan")
        row["davies_bouldin"] = float("nan")
        row["calinski_harabasz"] = float("nan")

    if clusterer_name == "hdbscan" and valid_mask.sum() >= 3:
        row["dbcv"] = _dbcv(X_red, labels)
    else:
        row["dbcv"] = float("nan")
    return row


def run_all(X: np.ndarray):
    reductions: dict[str, np.ndarray] = {}
    rows = []
    labels_by_config: dict[str, np.ndarray] = {}

    for reducer_name in ["pca", "umap", "identity"]:
        X_red, _ = REDUCERS[reducer_name](X)
        reductions[reducer_name] = X_red

        for cid, _, clusterer_name in [cfg for cfg in CONFIGS if cfg[1] == reducer_name]:
            labels, info = CLUSTERERS[clusterer_name](X_red)
            labels = np.asarray(labels, dtype=int)
            labels_by_config[cid] = labels

            row = {
                "config_id": cid,
                "reducer": reducer_name,
                "clusterer": clusterer_name,
                "k": info.get("k"),
                "k_effective": info.get("k_effective", info.get("k")),
                "noise_ratio": info.get("noise_ratio", 0.0),
                "labels": labels,
            }
            if "bic" in info:
                row["bic"] = info["bic"]
            if "hdbscan_impl" in info:
                row["hdbscan_impl"] = info["hdbscan_impl"]
            row.update(score_labels(X_red, labels, clusterer_name))
            rows.append(row)

    return pd.DataFrame(rows), labels_by_config, reductions
