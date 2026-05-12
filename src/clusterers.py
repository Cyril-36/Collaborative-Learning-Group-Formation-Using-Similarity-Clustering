"""Clusterer implementations for the 12-configuration sweep."""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics import silhouette_score
from sklearn.mixture import GaussianMixture

from .config import HDBSCAN_MIN_CLUSTER_SIZE, HDBSCAN_MIN_SAMPLES, K_SWEEP, SEED


def effective_k(labels: np.ndarray) -> int:
    unique = set(np.asarray(labels).tolist())
    return len(unique - {-1})


def _can_score(labels: np.ndarray, n_samples: int) -> bool:
    k = len(set(np.asarray(labels).tolist()))
    return 1 < k < n_samples


def _pick_k_silhouette(X: np.ndarray, fit_fn, k_sweep: list[int] = K_SWEEP):
    best_k, best_score, best_labels = None, -np.inf, None
    for k in k_sweep:
        if k >= X.shape[0]:
            continue
        labels = fit_fn(X, k)
        if not _can_score(labels, X.shape[0]):
            continue
        score = silhouette_score(X, labels)
        if score > best_score or (np.isclose(score, best_score) and (best_k is None or k < best_k)):
            best_k, best_score, best_labels = k, score, labels
    if best_labels is None:
        raise ValueError("Could not find a valid k for silhouette selection")
    return best_k, best_labels


def cluster_kmeans(X: np.ndarray, k_sweep: list[int] | None = None):
    def fit(data, k):
        return KMeans(n_clusters=k, n_init=10, random_state=SEED).fit_predict(data)

    k, labels = _pick_k_silhouette(X, fit, k_sweep=k_sweep or K_SWEEP)
    return labels.astype(int), {"k": k, "k_effective": effective_k(labels), "noise_ratio": 0.0}


def cluster_gmm(X: np.ndarray, k_sweep: list[int] | None = None):
    best_k, best_bic, best_labels = None, np.inf, None
    for k in (k_sweep or K_SWEEP):
        if k >= X.shape[0]:
            continue
        model = GaussianMixture(n_components=k, covariance_type="full", random_state=SEED)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(X)
        bic = model.bic(X)
        if bic < best_bic or (np.isclose(bic, best_bic) and (best_k is None or k < best_k)):
            best_k, best_bic, best_labels = k, bic, model.predict(X)
    if best_labels is None:
        raise ValueError("Could not fit a valid GMM")
    return best_labels.astype(int), {
        "k": best_k,
        "k_effective": effective_k(best_labels),
        "bic": float(best_bic),
        "noise_ratio": 0.0,
    }


def cluster_agglo(X: np.ndarray, k_sweep: list[int] | None = None):
    def fit(data, k):
        return AgglomerativeClustering(n_clusters=k, linkage="ward").fit_predict(data)

    k, labels = _pick_k_silhouette(X, fit, k_sweep=k_sweep or K_SWEEP)
    return labels.astype(int), {"k": k, "k_effective": effective_k(labels), "noise_ratio": 0.0}


def _fit_external_hdbscan(X: np.ndarray):
    import hdbscan

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min(HDBSCAN_MIN_CLUSTER_SIZE, max(2, X.shape[0] // 5)),
        min_samples=min(HDBSCAN_MIN_SAMPLES, max(1, X.shape[0] // 10)),
        cluster_selection_method="eom",
    )
    return clusterer.fit_predict(X), clusterer, "hdbscan"


def _fit_sklearn_hdbscan(X: np.ndarray):
    from sklearn.cluster import HDBSCAN

    clusterer = HDBSCAN(
        min_cluster_size=min(HDBSCAN_MIN_CLUSTER_SIZE, max(2, X.shape[0] // 5)),
        min_samples=min(HDBSCAN_MIN_SAMPLES, max(1, X.shape[0] // 10)),
    )
    return clusterer.fit_predict(X), clusterer, "sklearn"


def cluster_hdbscan(X: np.ndarray):
    try:
        labels, clusterer, implementation = _fit_external_hdbscan(X)
    except ImportError:
        labels, clusterer, implementation = _fit_sklearn_hdbscan(X)

    labels = np.asarray(labels, dtype=int)
    noise_ratio = float(np.mean(labels == -1))
    return labels, {
        "k": effective_k(labels),
        "k_effective": effective_k(labels),
        "noise_ratio": noise_ratio,
        "clusterer": clusterer,
        "hdbscan_impl": implementation,
    }


CLUSTERERS: dict[str, Any] = {
    "kmeans": cluster_kmeans,
    "gmm": cluster_gmm,
    "agglo": cluster_agglo,
    "hdbscan": cluster_hdbscan,
}
