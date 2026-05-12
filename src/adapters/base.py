"""Dataset adapter interfaces for learner grouping inputs."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Protocol

import pandas as pd


CANONICAL_ID_COL = "id_student"


@dataclass
class DatasetSchema:
    dataset_name: str
    adapter_name: str
    source_id_col: str
    id_col: str = CANONICAL_ID_COL
    feature_cols: list[str] = field(default_factory=list)
    numeric_feature_cols: list[str] = field(default_factory=list)
    categorical_feature_cols: list[str] = field(default_factory=list)
    fairness_cols: list[str] = field(default_factory=list)
    engagement_col: str | None = None
    performance_col: str | None = None
    outcome_col: str | None = None
    stratification_col: str | None = None
    display_cols: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    def clustering_feature_cols(self) -> list[str]:
        """Columns that should enter preprocessing/clustering."""
        return list(dict.fromkeys(self.numeric_feature_cols + self.categorical_feature_cols))

    def role_cols(self) -> list[str]:
        cols: list[str] = []
        cols.extend(self.fairness_cols)
        for col in [self.engagement_col, self.performance_col, self.outcome_col, self.stratification_col]:
            if col:
                cols.append(col)
        cols.extend(self.display_cols)
        return list(dict.fromkeys([col for col in cols if col and col != self.id_col]))


class DatasetAdapter(Protocol):
    name: str

    def load(self) -> object:
        ...

    def build_features(self, raw: object) -> tuple[pd.DataFrame, DatasetSchema]:
        ...


def normalize_id_column(df: pd.DataFrame, source_id_col: str) -> pd.DataFrame:
    if source_id_col not in df.columns:
        raise ValueError(f"ID column {source_id_col!r} not found in dataset")
    out = df.copy()
    if source_id_col != CANONICAL_ID_COL:
        out = out.rename(columns={source_id_col: CANONICAL_ID_COL})
    if out[CANONICAL_ID_COL].isna().any():
        raise ValueError(f"ID column {source_id_col!r} contains missing values")
    if out[CANONICAL_ID_COL].duplicated().any():
        raise ValueError(f"ID column {source_id_col!r} must be unique per learner")
    return out


def path_dataset_name(path: str | Path) -> str:
    return Path(path).stem.replace(" ", "_")
