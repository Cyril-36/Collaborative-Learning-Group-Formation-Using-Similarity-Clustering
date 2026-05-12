"""OULAD dataset adapter."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .. import features, ingest
from ..config import PRESENTATION, parse_presentation
from .base import DatasetSchema


class OuladAdapter:
    name = "oulad"

    def __init__(
        self,
        presentation: str | None = None,
        source_dir: Path | None = None,
        feature_cols: list[str] | None = None,
    ) -> None:
        self.code_module, self.code_presentation = parse_presentation(presentation)
        self.source_dir = source_dir
        self.feature_cols = feature_cols

    @property
    def presentation(self) -> str:
        return f"{self.code_module}_{self.code_presentation}"

    def load(self) -> dict[str, pd.DataFrame]:
        return ingest.run(self.code_module, self.code_presentation, self.source_dir)

    def build_features(self, raw: object) -> tuple[pd.DataFrame, DatasetSchema]:
        if not isinstance(raw, dict):
            raise TypeError("OuladAdapter.load() must return a table dictionary")
        matrix = features.run(raw)
        if "final_result" in raw.get("info", pd.DataFrame()).columns:
            outcomes = raw["info"][["id_student", "final_result"]].drop_duplicates("id_student")
            matrix = matrix.merge(outcomes, on="id_student", how="left")

        numeric_features = [
            col for col in matrix.columns
            if col not in {"id_student", "final_result"} and pd.api.types.is_numeric_dtype(matrix[col])
        ]
        if self.feature_cols is not None:
            missing = [col for col in self.feature_cols if col not in matrix.columns]
            if missing:
                raise ValueError(f"Feature columns missing from OULAD features: {missing}")
            non_numeric = [
                col for col in self.feature_cols
                if not pd.api.types.is_numeric_dtype(matrix[col])
            ]
            if non_numeric:
                raise ValueError(f"OULAD feature columns must be numeric: {non_numeric}")
            numeric_features = list(dict.fromkeys(self.feature_cols))
        display_cols = [
            col for col in [
                "total_clicks",
                "weighted_score",
                "imd_band_ord",
                "collaborative_clicks",
                "collaboration_click_ratio",
                "final_result",
            ]
            if col in matrix.columns
        ]
        schema = DatasetSchema(
            dataset_name=f"OULAD {self.presentation}",
            adapter_name=self.name,
            source_id_col="id_student",
            feature_cols=numeric_features,
            numeric_feature_cols=numeric_features,
            categorical_feature_cols=[],
            fairness_cols=["imd_band_ord"] if "imd_band_ord" in matrix.columns else [],
            engagement_col="total_clicks" if "total_clicks" in matrix.columns else None,
            performance_col="weighted_score" if "weighted_score" in matrix.columns else None,
            outcome_col="final_result" if "final_result" in matrix.columns else None,
            stratification_col="imd_band_ord" if "imd_band_ord" in matrix.columns else None,
            display_cols=display_cols,
        )
        return matrix, schema


def default_oulad_adapter() -> OuladAdapter:
    return OuladAdapter(f"{PRESENTATION[0]}_{PRESENTATION[1]}")
