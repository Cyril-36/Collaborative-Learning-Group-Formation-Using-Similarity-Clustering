"""Live-prediction artifacts and predict-one helper for the Streamlit demo.

This module sits at the boundary between the offline pipeline (which fits all
estimators) and the online demo (which feeds a single hypothetical learner
through the trained system).

The trained "model" here is three artifacts plus metadata:

    1. fitted ``ColumnTransformer`` (from ``preprocess.preprocess``)
    2. fitted reducer for the *winning* configuration (UMAP / PCA / identity)
    3. fitted clusterer for the *winning* configuration (KMeans / GMM / Agglo)

Plus a ``PredictSchema`` describing how to build a one-row input DataFrame
from a raw ``{column: value}`` dict — and how to read the answer back.

Persisted layout (in the pipeline cache directory):

    cache_dir/
        predict_artifacts.joblib   — the three fitted estimators + remap + keep mask
        predict_schema.json        — human-readable input field metadata

The Streamlit demo loads both at startup, then per click:

    >>> from src.predict import load_artifacts, predict_one
    >>> art = load_artifacts(Path("demo/demo_cache"))
    >>> result = predict_one({"total_clicks": 1200, "active_days": 50, ...}, art)
    >>> result["cluster"], result["umap_2d"], result["confidence"]
"""

from __future__ import annotations

import json
import math
import os
import threading
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# UMAP's transform() runs through numba; the default 'workqueue' threading
# layer is NOT threadsafe and crashes when Streamlit triggers two reruns at
# once (fast slider drags, browser nav). Switch to 'omp' which IS threadsafe.
# Must set BEFORE numba/umap import — keep this above the joblib import.
os.environ.setdefault("NUMBA_THREADING_LAYER", "omp")

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer

from .adapters.base import DatasetSchema
from .config import SEED
from .reducers import reduce_pca, reduce_umap, reduce_umap_2d


# Defense-in-depth: serialize predict_one calls so two concurrent reruns
# can't race UMAP's internal numba state.
_PREDICT_LOCK = threading.Lock()


_ARTIFACTS_FILENAME = "predict_artifacts.joblib"
_SCHEMA_FILENAME = "predict_schema.json"


# === Data classes =====================================================


@dataclass
class FieldSpec:
    """Metadata for one input field exposed to the demo UI.

    For numeric fields, ``p05`` and ``p95`` clip slider ranges to the training
    data's 5-95 percentile band so users cannot drag into extrapolation
    regions where UMAP's transform-on-new-point becomes unreliable.

    For categorical fields, ``categories`` enumerates the valid values that
    the fitted ``OneHotEncoder`` knows about (others map to all-zeros).
    """

    name: str
    kind: str                              # "numeric" | "categorical"
    median: float | str | None = None      # default fill value
    p05: float | None = None               # numeric only
    p95: float | None = None               # numeric only
    categories: list[str] = field(default_factory=list)  # categorical only
    is_role_engagement: bool = False
    is_role_performance: bool = False
    is_role_fairness: bool = False


@dataclass
class PredictSchema:
    """Human-readable description of the predict surface.

    Serialized to ``predict_schema.json`` so the Streamlit demo can render
    sliders/dropdowns without reaching into the joblib artifacts.
    """

    dataset_name: str
    adapter_name: str
    id_col: str
    n_clusters: int
    cluster_label_map: dict[int, str]      # canonical cluster id -> human-friendly name
    fields: list[FieldSpec]
    winner_config_id: str
    winner_reducer: str
    winner_clusterer: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset_name": self.dataset_name,
            "adapter_name": self.adapter_name,
            "id_col": self.id_col,
            "n_clusters": self.n_clusters,
            "cluster_label_map": {str(k): v for k, v in self.cluster_label_map.items()},
            "winner_config_id": self.winner_config_id,
            "winner_reducer": self.winner_reducer,
            "winner_clusterer": self.winner_clusterer,
            "fields": [
                {
                    "name": f.name,
                    "kind": f.kind,
                    "median": f.median,
                    "p05": f.p05,
                    "p95": f.p95,
                    "categories": f.categories,
                    "is_role_engagement": f.is_role_engagement,
                    "is_role_performance": f.is_role_performance,
                    "is_role_fairness": f.is_role_fairness,
                }
                for f in self.fields
            ],
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "PredictSchema":
        return cls(
            dataset_name=raw["dataset_name"],
            adapter_name=raw["adapter_name"],
            id_col=raw["id_col"],
            n_clusters=int(raw["n_clusters"]),
            cluster_label_map={int(k): v for k, v in raw.get("cluster_label_map", {}).items()},
            winner_config_id=raw["winner_config_id"],
            winner_reducer=raw["winner_reducer"],
            winner_clusterer=raw["winner_clusterer"],
            fields=[FieldSpec(**f) for f in raw["fields"]],
        )


@dataclass
class PredictArtifacts:
    """Fitted estimators + metadata needed to score one new learner end-to-end.

    Pickled to ``predict_artifacts.joblib``. Loaded once at app startup.
    """

    transformer: ColumnTransformer        # fitted on training feature matrix
    keep_mask: np.ndarray                  # boolean, shape (n_encoded_features,)
    feature_names: list[str]               # final post-keep-mask names

    reducer_winner: Any                    # fitted UMAP / PCA / None (identity)
    reducer_2d: Any                        # fitted UMAP for 2D viz; None if pca_2d fallback
    pca_2d_fallback: np.ndarray | None     # if UMAP-2D failed during pipeline run

    clusterer_winner: Any                  # fitted KMeans / GMM / Agglo / HDBSCAN
    cluster_centers: np.ndarray            # in *reduced* space, post-canonical-remap
    raw_to_canonical: dict[int, int]       # raw cluster id from sweep -> canonical id

    schema: PredictSchema

    # Source-column lists so we can rebuild the input DataFrame in the right shape
    numeric_cols: list[str]
    categorical_cols: list[str]


# === Build (called from pipeline.run after winning config is chosen) ===


def _per_column_stats(
    feature_matrix: pd.DataFrame, schema: DatasetSchema
) -> tuple[dict[str, float], dict[str, str], dict[str, tuple[float, float]]]:
    """Return median fills for numerics, mode fills for categoricals, and 5-95
    percentile ranges for numerics (used to clip slider ranges in the demo).
    """
    numeric_medians: dict[str, float] = {}
    cat_modes: dict[str, str] = {}
    p05_p95: dict[str, tuple[float, float]] = {}

    for col in schema.numeric_feature_cols:
        if col not in feature_matrix.columns:
            continue
        s = pd.to_numeric(feature_matrix[col], errors="coerce")
        if s.dropna().empty:
            continue
        numeric_medians[col] = float(s.median())
        p05_p95[col] = (float(s.quantile(0.05)), float(s.quantile(0.95)))

    for col in schema.categorical_feature_cols:
        if col not in feature_matrix.columns:
            continue
        mode = feature_matrix[col].mode(dropna=True)
        if not mode.empty:
            cat_modes[col] = str(mode.iloc[0])

    return numeric_medians, cat_modes, p05_p95


def _categorical_categories(transformer: ColumnTransformer) -> dict[str, list[str]]:
    """Pull learned categories out of the OneHotEncoder inside the cat pipeline."""
    out: dict[str, list[str]] = {}
    for name, _trans, cols in transformer.transformers_:
        if name != "cat":
            continue
        try:
            from sklearn.preprocessing import OneHotEncoder
            ohe: OneHotEncoder = _trans.named_steps["onehot"]
            for col, cats in zip(cols, ohe.categories_):
                out[col] = [str(c) for c in cats]
        except (KeyError, AttributeError):
            continue
    return out


def _refit_winner(
    X_scaled: np.ndarray,
    reducer_name: str,
    clusterer_name: str,
    expected_labels: np.ndarray,
) -> tuple[Any, Any, np.ndarray, np.ndarray]:
    """Refit the winning reducer + clusterer deterministically.

    Returns ``(fitted_reducer, fitted_clusterer, X_red, labels)``. With ``SEED``
    fixed, the labels should match ``expected_labels`` exactly — we verify and
    warn if they don't (rare; happens if numpy/sklearn versions differ).

    For HDBSCAN, ``fitted_clusterer`` is the trained estimator; for Agglo it's
    not refittable to new data so we fall back to a KMeans surrogate fit on
    the same X_red+labels (good enough for the live-predict demo, which is the
    only consumer of this fitted object).
    """
    # Reducer
    if reducer_name == "pca":
        X_red, reducer = reduce_pca(X_scaled)
    elif reducer_name == "umap":
        X_red, reducer = reduce_umap(X_scaled)
    elif reducer_name == "identity":
        X_red, reducer = X_scaled.copy(), None
    else:
        raise ValueError(f"unknown reducer_name={reducer_name!r}")

    # Clusterer — for the live-predict surface we only need
    # cluster_centers + a .predict path. KMeans gives us both natively;
    # for non-KMeans winners we fit a surrogate KMeans on the labels.
    if clusterer_name == "kmeans":
        from .clusterers import _pick_k_silhouette

        def _fit(data, k):
            return KMeans(n_clusters=k, n_init=10, random_state=SEED).fit_predict(data)

        k, _ = _pick_k_silhouette(X_red, _fit)
        clusterer = KMeans(n_clusters=k, n_init=10, random_state=SEED).fit(X_red)
        labels = clusterer.predict(X_red).astype(int)
    else:
        # Use the labels that were already chosen by the sweep, then fit a
        # KMeans **surrogate** with k = effective_k(labels) so predict-on-new
        # works. The surrogate is initialised at the per-cluster centroids
        # of expected_labels so it converges in one step and preserves the
        # canonical cluster assignment.
        valid_mask = expected_labels != -1
        valid_labels = expected_labels[valid_mask]
        unique = np.array(sorted(set(valid_labels.tolist())), dtype=int)
        k = len(unique)
        if k < 2:
            raise ValueError(
                f"cannot build predict surrogate for {clusterer_name!r} winner "
                f"with k_effective={k}"
            )
        init_centers = np.vstack([
            X_red[valid_mask][valid_labels == c].mean(axis=0) for c in unique
        ])
        clusterer = KMeans(n_clusters=k, n_init=1, init=init_centers, random_state=SEED)
        clusterer.fit(X_red[valid_mask])
        labels = expected_labels.copy()

    # Determinism check (informational — sweep uses the same SEED so this
    # should always match for the kmeans path).
    if clusterer_name == "kmeans":
        n_match = int((labels == expected_labels).sum())
        n_total = len(expected_labels)
        if n_match < n_total:
            # Cluster ids may differ by permutation; check ARI as a proxy
            from sklearn.metrics import adjusted_rand_score
            ari = float(adjusted_rand_score(expected_labels, labels))
            if ari < 0.99:
                warnings.warn(
                    f"refit labels diverge from sweep (ARI={ari:.3f}); "
                    f"predict-one assignments may differ slightly from cluster_labels.parquet",
                    stacklevel=2,
                )

    return reducer, clusterer, X_red, labels


def _refit_2d(X_scaled: np.ndarray) -> tuple[Any, np.ndarray | None]:
    """Refit the 2D viz reducer deterministically. Falls back to PCA if UMAP fails."""
    try:
        coords, reducer = reduce_umap_2d(X_scaled)
        return reducer, None
    except Exception as exc:  # pragma: no cover — install-time dependency issue
        warnings.warn(f"UMAP-2D refit failed ({exc}); using PCA-2D fallback", stacklevel=2)
        from sklearn.decomposition import PCA
        pca2 = PCA(n_components=2, random_state=SEED).fit(X_scaled)
        return None, pca2.transform(X_scaled)


def _build_field_specs(
    feature_matrix: pd.DataFrame, schema: DatasetSchema, transformer: ColumnTransformer
) -> list[FieldSpec]:
    """Produce one FieldSpec per RAW column the demo will ask the user about."""
    medians, modes, ranges = _per_column_stats(feature_matrix, schema)
    cat_categories = _categorical_categories(transformer)

    specs: list[FieldSpec] = []
    for col in schema.numeric_feature_cols:
        if col not in feature_matrix.columns:
            continue
        median = medians.get(col)
        p05, p95 = ranges.get(col, (None, None))
        specs.append(
            FieldSpec(
                name=col,
                kind="numeric",
                median=median,
                p05=p05,
                p95=p95,
                is_role_engagement=(col == schema.engagement_col),
                is_role_performance=(col == schema.performance_col),
                is_role_fairness=(col in schema.fairness_cols),
            )
        )

    for col in schema.categorical_feature_cols:
        if col not in feature_matrix.columns:
            continue
        specs.append(
            FieldSpec(
                name=col,
                kind="categorical",
                median=modes.get(col),
                categories=cat_categories.get(col, []),
                is_role_fairness=(col in schema.fairness_cols),
            )
        )

    return specs


def build_predict_artifacts(
    *,
    feature_matrix: pd.DataFrame,
    schema: DatasetSchema,
    transformer: ColumnTransformer,
    keep_mask: np.ndarray,
    feature_names: list[str],
    X_scaled: np.ndarray,
    winner_config_id: str,
    winner_reducer: str,
    winner_clusterer: str,
    winner_labels: np.ndarray,
    raw_to_canonical: dict[int, int],
    cluster_label_map: dict[int, str] | None = None,
) -> PredictArtifacts:
    """Refit the winning configuration deterministically and assemble artifacts.

    Called from ``pipeline.run`` after the winner has been chosen and the
    canonical remap is known. ``feature_matrix`` is the post-sample,
    post-build_features DataFrame (one row per learner, raw columns).
    ``X_scaled`` is the post-preprocess matrix already in the keep-masked shape.
    """
    reducer_w, clusterer_w, X_red, refit_labels = _refit_winner(
        X_scaled, winner_reducer, winner_clusterer, winner_labels
    )
    reducer_2d, pca_2d_fallback = _refit_2d(X_scaled)

    # Compute centroids in REDUCED space, labelled by canonical id, so the
    # demo can hand-roll distance-to-centroid without rerunning kmeans.predict.
    canonical_labels = np.array([raw_to_canonical.get(int(l), int(l)) for l in refit_labels])
    canonical_unique = sorted(set(int(c) for c in canonical_labels) - {-1})
    centers = np.vstack([
        X_red[canonical_labels == c].mean(axis=0) for c in canonical_unique
    ])

    schema_obj = PredictSchema(
        dataset_name=schema.dataset_name,
        adapter_name=schema.adapter_name,
        id_col=schema.id_col,
        n_clusters=len(canonical_unique),
        cluster_label_map=(cluster_label_map or {c: f"Cluster {c}" for c in canonical_unique}),
        fields=_build_field_specs(feature_matrix, schema, transformer),
        winner_config_id=winner_config_id,
        winner_reducer=winner_reducer,
        winner_clusterer=winner_clusterer,
    )

    return PredictArtifacts(
        transformer=transformer,
        keep_mask=np.asarray(keep_mask, dtype=bool),
        feature_names=list(feature_names),
        reducer_winner=reducer_w,
        reducer_2d=reducer_2d,
        pca_2d_fallback=pca_2d_fallback,
        clusterer_winner=clusterer_w,
        cluster_centers=centers,
        raw_to_canonical={int(k): int(v) for k, v in raw_to_canonical.items()},
        schema=schema_obj,
        numeric_cols=[c for c in schema.numeric_feature_cols if c in feature_matrix.columns],
        categorical_cols=[c for c in schema.categorical_feature_cols if c in feature_matrix.columns],
    )


def save_artifacts(art: PredictArtifacts, cache_dir: Path) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(art, cache_dir / _ARTIFACTS_FILENAME, compress=3)
    (cache_dir / _SCHEMA_FILENAME).write_text(
        json.dumps(art.schema.to_dict(), indent=2),
        encoding="utf-8",
    )


def load_artifacts(cache_dir: Path) -> PredictArtifacts:
    path = cache_dir / _ARTIFACTS_FILENAME
    if not path.exists():
        raise FileNotFoundError(
            f"predict artifacts missing at {path}; rerun the pipeline so it "
            f"writes them via predict.save_artifacts()."
        )
    return joblib.load(path)


def load_schema(cache_dir: Path) -> PredictSchema:
    path = cache_dir / _SCHEMA_FILENAME
    if not path.exists():
        raise FileNotFoundError(
            f"predict schema missing at {path}; rerun the pipeline so it "
            f"writes both the .joblib and .json artifacts."
        )
    return PredictSchema.from_dict(json.loads(path.read_text(encoding="utf-8")))


# === Predict-one (called from the Streamlit demo per click) ===========


def _row_dataframe(
    raw: dict[str, Any], art: PredictArtifacts
) -> pd.DataFrame:
    """Assemble the single-row DataFrame the fitted ColumnTransformer expects.

    Missing fields fall back to the per-column median / mode that
    ``build_predict_artifacts`` computed at fit time. This means the demo can
    expose any subset of the schema's fields and still produce a valid input.
    """
    row: dict[str, Any] = {}
    for spec in art.schema.fields:
        if spec.name in raw and raw[spec.name] is not None and raw[spec.name] != "":
            row[spec.name] = raw[spec.name]
        else:
            row[spec.name] = spec.median

    # Make sure every column the transformer was trained on is present, even if
    # the demo doesn't expose it (we still need the column to exist).
    for col in art.numeric_cols + art.categorical_cols:
        if col not in row:
            row[col] = next(
                (s.median for s in art.schema.fields if s.name == col),
                np.nan if col in art.numeric_cols else "missing",
            )

    df = pd.DataFrame([row])
    # Coerce numeric columns properly so SimpleImputer doesn't trip
    for col in art.numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    for col in art.categorical_cols:
        if col in df.columns:
            df[col] = df[col].astype(str)
    return df


def predict_one(raw: dict[str, Any], art: PredictArtifacts) -> dict[str, Any]:
    """Run a single hypothetical learner through the trained pipeline.

    Wrapped in a process-wide lock so concurrent Streamlit reruns can't race
    UMAP's internal numba state (the default 'workqueue' threading layer is
    not threadsafe; we also set NUMBA_THREADING_LAYER=omp at import time).

    Parameters
    ----------
    raw : dict
        ``{column_name: value}``. Missing entries fall back to per-column
        median / mode learned at fit time.
    art : PredictArtifacts
        Loaded from ``load_artifacts(cache_dir)``.

    Returns
    -------
    dict with keys
        ``cluster``      : int — canonical cluster id assigned to this learner
        ``cluster_label``: str — human-friendly label from cluster_label_map
        ``distance``     : float — Euclidean distance from this learner's
                           reduced-space coords to the assigned centroid
        ``confidence``   : str — one of ``"high"``, ``"medium"``, ``"low"``
                           based on distance vs the sample's max-centroid radius
        ``umap_2d``      : tuple[float, float] — coords for the 2D viz dot
        ``reduced_vec``  : np.ndarray — coords in the winning reducer's space
        ``feature_vec``  : np.ndarray — post-preprocess scaled feature vector
        ``all_distances``: dict[int, float] — distance to every centroid
    """
    with _PREDICT_LOCK:
        return _predict_one_locked(raw, art)


def _predict_one_locked(raw: dict[str, Any], art: PredictArtifacts) -> dict[str, Any]:
    df = _row_dataframe(raw, art)

    # 1. Preprocess
    full = np.asarray(art.transformer.transform(df), dtype=float)
    feature_vec = full[:, art.keep_mask].reshape(-1)

    # 2. Reduce (winner reducer)
    if art.reducer_winner is None:                        # identity
        reduced = feature_vec.copy()
    else:
        reduced = np.asarray(art.reducer_winner.transform(feature_vec.reshape(1, -1))).reshape(-1)

    # 3. Distance to each canonical centroid
    diffs = art.cluster_centers - reduced[None, :]
    all_dist = np.linalg.norm(diffs, axis=1)
    canonical_unique = sorted(set(int(c) for c in art.raw_to_canonical.values()))
    nearest_idx = int(np.argmin(all_dist))
    cluster = canonical_unique[nearest_idx]
    distance = float(all_dist[nearest_idx])

    # 4. Confidence — relative to the largest within-sample centroid distance
    #    (set during build_predict_artifacts via inter-centroid spread)
    spread = float(np.linalg.norm(art.cluster_centers - art.cluster_centers.mean(axis=0), axis=1).max())
    if spread > 0:
        ratio = distance / spread
        if ratio < 0.7:
            confidence = "high"
        elif ratio < 1.2:
            confidence = "medium"
        else:
            confidence = "low"
    else:
        confidence = "low"

    # 5. 2D viz coord
    if art.reducer_2d is not None:
        coord_2d = np.asarray(art.reducer_2d.transform(feature_vec.reshape(1, -1))).reshape(-1)[:2]
    elif art.pca_2d_fallback is not None:
        from sklearn.decomposition import PCA
        # Refit a quick projection — should never be needed at predict time if
        # the pipeline persisted reducer_2d, but here for completeness.
        pca = PCA(n_components=2, random_state=SEED).fit(art.pca_2d_fallback)
        coord_2d = pca.transform(feature_vec.reshape(1, -1))[0]
    else:
        coord_2d = reduced[:2] if reduced.size >= 2 else np.array([reduced[0], 0.0])

    return {
        "cluster": int(cluster),
        "cluster_label": art.schema.cluster_label_map.get(int(cluster), f"Cluster {cluster}"),
        "distance": distance,
        "confidence": confidence,
        "umap_2d": (float(coord_2d[0]), float(coord_2d[1])),
        "reduced_vec": reduced,
        "feature_vec": feature_vec,
        "all_distances": {int(canonical_unique[i]): float(all_dist[i]) for i in range(len(canonical_unique))},
    }


__all__ = [
    "FieldSpec",
    "PredictSchema",
    "PredictArtifacts",
    "build_predict_artifacts",
    "save_artifacts",
    "load_artifacts",
    "load_schema",
    "predict_one",
]
