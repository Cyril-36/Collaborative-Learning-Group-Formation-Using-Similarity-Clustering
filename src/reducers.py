"""Dimensionality reducers used by the configuration matrix."""

from __future__ import annotations

import numpy as np
from sklearn.decomposition import PCA

from .config import (
    PCA_N_COMPONENTS,
    SEED,
    UMAP_2D_COMPONENTS,
    UMAP_MIN_DIST,
    UMAP_N_COMPONENTS,
    UMAP_N_NEIGHBORS,
)


def _safe_components(X: np.ndarray, requested: int) -> int:
    return max(1, min(requested, X.shape[1], X.shape[0] - 1))


def reduce_pca(X: np.ndarray, n_components: int = PCA_N_COMPONENTS):
    n = _safe_components(X, n_components)
    reducer = PCA(n_components=n, random_state=SEED)
    return reducer.fit_transform(X), reducer


def reduce_umap(
    X: np.ndarray,
    n_components: int = UMAP_N_COMPONENTS,
    n_neighbors: int = UMAP_N_NEIGHBORS,
    min_dist: float = UMAP_MIN_DIST,
):
    try:
        import umap
    except ImportError as exc:
        raise ImportError("Install umap-learn to use the UMAP reducer") from exc

    n = _safe_components(X, n_components)
    reducer = umap.UMAP(
        n_components=n,
        n_neighbors=min(n_neighbors, max(2, X.shape[0] - 1)),
        min_dist=min_dist,
        metric="euclidean",
        random_state=SEED,
        n_jobs=1,
    )
    return reducer.fit_transform(X), reducer


def reduce_umap_2d(X: np.ndarray):
    return reduce_umap(X, n_components=UMAP_2D_COMPONENTS)


def reduce_identity(X: np.ndarray):
    return X.copy(), None


REDUCERS = {
    "pca": reduce_pca,
    "umap": reduce_umap,
    "identity": reduce_identity,
}
