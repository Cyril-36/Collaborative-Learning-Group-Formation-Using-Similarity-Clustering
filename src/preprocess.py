"""Schema-aware imputation, encoding, and scaling."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .adapters.base import CANONICAL_ID_COL, DatasetSchema


@dataclass
class PreprocessResult:
    ids: np.ndarray
    X: np.ndarray
    transformer: ColumnTransformer
    scaler: ColumnTransformer
    feature_names: list[str]
    clean_features: pd.DataFrame
    keep_mask: np.ndarray = None  # type: ignore[assignment]
    """Boolean mask over the transformer's raw output, marking finite + non-constant
    columns. Live-predict path applies this to ``transformer.transform(new_row)``
    so the new feature vector lines up with the training X exactly."""

    def __iter__(self) -> Iterator[object]:
        """Preserve the old six-value unpacking API used by tests/scripts."""
        yield self.ids
        yield self.X
        yield self.transformer
        yield self.scaler
        yield self.feature_names
        yield self.clean_features


def _legacy_schema(feature_matrix: pd.DataFrame) -> DatasetSchema:
    if CANONICAL_ID_COL not in feature_matrix.columns:
        raise ValueError("feature_matrix must contain an id_student column")
    numeric_cols = [
        col for col in feature_matrix.columns
        if col != CANONICAL_ID_COL
    ]
    return DatasetSchema(
        dataset_name="legacy",
        adapter_name="legacy",
        source_id_col=CANONICAL_ID_COL,
        feature_cols=numeric_cols,
        numeric_feature_cols=numeric_cols,
    )


def _feature_name(name: str) -> str:
    for prefix in ["num__", "cat__"]:
        if name.startswith(prefix):
            return name[len(prefix):]
    return name


def preprocess(
    feature_matrix: pd.DataFrame,
    schema: DatasetSchema | None = None,
    family_weights: dict[str, float] | None = None,
) -> PreprocessResult:
    """Return IDs, scaled feature matrix, fitted transformer, and clean features."""
    schema = schema or _legacy_schema(feature_matrix)
    if schema.id_col not in feature_matrix.columns:
        raise ValueError(f"feature_matrix must contain a {schema.id_col!r} column")

    ids = feature_matrix[schema.id_col].to_numpy()
    numeric_cols = [col for col in schema.numeric_feature_cols if col in feature_matrix.columns]
    categorical_cols = [col for col in schema.categorical_feature_cols if col in feature_matrix.columns]
    selected_cols = list(dict.fromkeys(numeric_cols + categorical_cols))
    if not selected_cols:
        raise ValueError("No clustering features are available after applying dataset schema")

    work = feature_matrix[selected_cols].copy()
    for col in numeric_cols:
        work[col] = pd.to_numeric(work[col], errors="coerce")

    transformers: list[tuple[str, Pipeline, list[str]]] = []
    if numeric_cols:
        transformers.append(
            (
                "num",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric_cols,
            )
        )
    if categorical_cols:
        transformers.append(
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                        # Z-scoring one-hot columns is intentional: all encoded
                        # dimensions then contribute comparably to distance-based clustering.
                        ("scaler", StandardScaler()),
                    ]
                ),
                categorical_cols,
            )
        )

    transformer = ColumnTransformer(transformers=transformers, remainder="drop", verbose_feature_names_out=True)
    X = transformer.fit_transform(work)
    X = np.asarray(X, dtype=float)
    feature_names = [_feature_name(name) for name in transformer.get_feature_names_out()]

    finite_mask = np.isfinite(X).all(axis=0)
    nonzero_mask = np.nanvar(X, axis=0) > 0
    keep = finite_mask & nonzero_mask
    if not keep.any():
        raise ValueError("No non-constant features remain after preprocessing")
    X = X[:, keep]
    feature_names = [name for name, keep_col in zip(feature_names, keep) if keep_col]

    encoded = pd.DataFrame(X, columns=feature_names, index=feature_matrix.index)
    if family_weights:
        for prefix, weight in family_weights.items():
            cols = [col for col in encoded.columns if col.startswith(prefix)]
            if cols:
                encoded.loc[:, cols] *= float(weight)
        X = encoded.to_numpy()

    clean_features = work.copy()
    if numeric_cols:
        num_imputer = SimpleImputer(strategy="median")
        clean_features[numeric_cols] = num_imputer.fit_transform(work[numeric_cols])
    for col in categorical_cols:
        mode = work[col].mode(dropna=True)
        fill = mode.iloc[0] if not mode.empty else "missing"
        clean_features[col] = work[col].fillna(fill).astype(str)
    kept_source_cols = [
        col for col in selected_cols
        if col in feature_names or any(name.startswith(f"{col}_") for name in feature_names)
    ]
    clean_features = clean_features[kept_source_cols]

    return PreprocessResult(
        ids=ids,
        X=X,
        transformer=transformer,
        scaler=transformer,
        feature_names=feature_names,
        clean_features=clean_features.reset_index(drop=True),
        keep_mask=keep,
    )
