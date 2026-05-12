"""Generic one-row-per-learner CSV adapter."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .base import CANONICAL_ID_COL, DatasetSchema, normalize_id_column, path_dataset_name


class GenericCsvAdapter:
    name = "generic_csv"

    def __init__(
        self,
        path: str | Path,
        id_column: str,
        *,
        dataset_name: str | None = None,
        feature_cols: list[str] | None = None,
        fairness_cols: list[str] | None = None,
        engagement_col: str | None = None,
        performance_col: str | None = None,
        outcome_col: str | None = None,
        stratification_col: str | None = None,
        display_cols: list[str] | None = None,
    ) -> None:
        self.path = Path(path)
        self.id_column = id_column
        self.dataset_name = dataset_name or path_dataset_name(self.path)
        self.feature_cols = feature_cols
        self.fairness_cols = fairness_cols or []
        self.engagement_col = engagement_col
        self.performance_col = performance_col
        self.outcome_col = outcome_col
        self.stratification_col = stratification_col
        self.display_cols = display_cols

    def load(self) -> pd.DataFrame:
        if not self.path.exists():
            raise FileNotFoundError(f"CSV dataset not found: {self.path}")
        return pd.read_csv(self.path)

    def build_features(self, raw: object) -> tuple[pd.DataFrame, DatasetSchema]:
        if not isinstance(raw, pd.DataFrame):
            raise TypeError("GenericCsvAdapter.load() must return a pandas DataFrame")
        df = normalize_id_column(raw, self.id_column)

        metadata_cols = set(self.fairness_cols)
        metadata_cols.update(col for col in [self.outcome_col, self.stratification_col] if col)
        if self.feature_cols is None:
            candidates = [
                col for col in df.columns
                if col != CANONICAL_ID_COL and col not in metadata_cols
            ]
            numeric_feature_cols = [
                col for col in candidates
                if pd.api.types.is_numeric_dtype(df[col])
            ]
            categorical_feature_cols = [
                col for col in candidates
                if not pd.api.types.is_numeric_dtype(df[col]) and df[col].nunique(dropna=True) <= 32
            ]
        else:
            missing = [col for col in self.feature_cols if col not in df.columns]
            if missing:
                raise ValueError(f"Feature columns missing from CSV: {missing}")
            numeric_feature_cols = [
                col for col in self.feature_cols
                if pd.api.types.is_numeric_dtype(df[col])
            ]
            categorical_feature_cols = [
                col for col in self.feature_cols
                if not pd.api.types.is_numeric_dtype(df[col])
            ]

        for col in [
            *self.fairness_cols,
            self.engagement_col,
            self.performance_col,
            self.outcome_col,
            self.stratification_col,
        ]:
            if col and col not in df.columns:
                raise ValueError(f"Schema column {col!r} not found in CSV")

        feature_cols = list(dict.fromkeys(numeric_feature_cols + categorical_feature_cols))
        display_cols = self.display_cols
        if display_cols is None:
            display_cols = [
                col for col in [
                    self.engagement_col,
                    self.performance_col,
                    *self.fairness_cols,
                    self.outcome_col,
                ]
                if col and col in df.columns
            ][:6]

        schema = DatasetSchema(
            dataset_name=self.dataset_name,
            adapter_name=self.name,
            source_id_col=self.id_column,
            feature_cols=feature_cols,
            numeric_feature_cols=numeric_feature_cols,
            categorical_feature_cols=categorical_feature_cols,
            fairness_cols=[col for col in self.fairness_cols if col],
            engagement_col=self.engagement_col,
            performance_col=self.performance_col,
            outcome_col=self.outcome_col,
            stratification_col=self.stratification_col,
            display_cols=list(dict.fromkeys(display_cols)),
        )
        return df, schema
