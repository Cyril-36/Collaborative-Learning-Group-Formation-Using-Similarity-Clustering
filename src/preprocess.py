"""Imputation and scaling for the learner feature matrix."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


def preprocess(
    feature_matrix: pd.DataFrame,
    family_weights: dict[str, float] | None = None,
) -> tuple[np.ndarray, np.ndarray, SimpleImputer, StandardScaler, list[str], pd.DataFrame]:
    """Return IDs, scaled feature array, fitted transformers, and cleaned features."""
    if "id_student" not in feature_matrix.columns:
        raise ValueError("feature_matrix must contain an id_student column")

    ids = feature_matrix["id_student"].to_numpy()
    feats = feature_matrix.drop(columns=["id_student"]).copy()
    feats = feats.apply(pd.to_numeric, errors="coerce")

    imputer = SimpleImputer(strategy="median")
    feats_imp = pd.DataFrame(imputer.fit_transform(feats), columns=feats.columns, index=feats.index)
    feats_imp = feats_imp.replace([np.inf, -np.inf], np.nan)
    if feats_imp.isna().any().any():
        feats_imp = feats_imp.fillna(feats_imp.median(numeric_only=True)).fillna(0)

    nonzero = feats_imp.var(axis=0) > 0
    feats_imp = feats_imp.loc[:, nonzero]
    if feats_imp.empty:
        raise ValueError("No non-constant features remain after preprocessing")

    scaler = StandardScaler()
    scaled = pd.DataFrame(
        scaler.fit_transform(feats_imp),
        columns=feats_imp.columns,
        index=feats_imp.index,
    )

    if family_weights:
        for prefix, weight in family_weights.items():
            cols = [col for col in scaled.columns if col.startswith(prefix)]
            if cols:
                scaled.loc[:, cols] *= float(weight)

    return ids, scaled.to_numpy(), imputer, scaler, scaled.columns.tolist(), feats_imp
