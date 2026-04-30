"""OULAD ingestion and scope filtering."""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import pandas as pd

from .config import (
    MIN_ENGAGEMENT_DAYS,
    PRESENTATION,
    PRESENTATION_LENGTH,
    raw_data_dir,
)


Tables = Dict[str, pd.DataFrame]


def _read_csv(path: Path, **kwargs) -> pd.DataFrame:
    return pd.read_csv(path, **kwargs)


def _filter_scope(df: pd.DataFrame, code_module: str, code_presentation: str) -> pd.DataFrame:
    if "code_module" not in df.columns or "code_presentation" not in df.columns:
        return df.copy()
    return df[
        (df["code_module"].eq(code_module))
        & (df["code_presentation"].eq(code_presentation))
    ].copy()


def _read_student_vle_scoped(
    path: Path,
    code_module: str,
    code_presentation: str,
    chunksize: int = 1_000_000,
) -> pd.DataFrame:
    chunks: list[pd.DataFrame] = []
    for chunk in pd.read_csv(path, chunksize=chunksize):
        filt = chunk[
            (chunk["code_module"].eq(code_module))
            & (chunk["code_presentation"].eq(code_presentation))
        ].copy()
        if not filt.empty:
            chunks.append(filt)
    if not chunks:
        return pd.DataFrame(
            columns=["code_module", "code_presentation", "id_student", "id_site", "date", "sum_click"]
        )
    return pd.concat(chunks, ignore_index=True)


def load_oulad(
    code_module: str | None = None,
    code_presentation: str | None = None,
    source_dir: Path | None = None,
) -> Tables:
    """Load OULAD CSVs, filtering large tables to the requested presentation."""
    code_module = code_module or PRESENTATION[0]
    code_presentation = code_presentation or PRESENTATION[1]
    base = source_dir or raw_data_dir()

    info = _filter_scope(_read_csv(base / "studentInfo.csv"), code_module, code_presentation)
    registration = _filter_scope(
        _read_csv(base / "studentRegistration.csv"),
        code_module,
        code_presentation,
    )
    assessments = _filter_scope(
        _read_csv(base / "assessments.csv"),
        code_module,
        code_presentation,
    )
    assessment = _read_csv(base / "studentAssessment.csv")
    if not assessments.empty:
        assessment = assessment[assessment["id_assessment"].isin(assessments["id_assessment"])].copy()
    vle = _read_student_vle_scoped(base / "studentVle.csv", code_module, code_presentation)
    vle_meta = _filter_scope(_read_csv(base / "vle.csv"), code_module, code_presentation)

    tables: Tables = {
        "info": info,
        "registration": registration,
        "assessment": assessment,
        "assessments": assessments,
        "vle": vle,
        "vle_meta": vle_meta,
    }
    if (base / "courses.csv").exists():
        tables["courses"] = _filter_scope(_read_csv(base / "courses.csv"), code_module, code_presentation)
    return tables


def exclude_early_withdrawals(
    tables: Tables,
    min_days: int = MIN_ENGAGEMENT_DAYS,
) -> Tables:
    """Drop learners who withdrew before the minimum engagement window."""
    reg = tables["registration"].copy()
    reg["date_unregistration"] = pd.to_numeric(reg["date_unregistration"], errors="coerce")
    early = set(reg.loc[reg["date_unregistration"].fillna(999999) < min_days, "id_student"])
    if not early:
        return tables

    for key in ["info", "registration", "assessment", "vle"]:
        tables[key] = tables[key][~tables[key]["id_student"].isin(early)].copy()
    return tables


def apply_study_window(
    tables: Tables,
    presentation_length: int = PRESENTATION_LENGTH,
) -> Tables:
    """Keep registrations before course start and VLE clicks in the study window."""
    reg = tables["registration"].copy()
    reg["date_registration"] = pd.to_numeric(reg["date_registration"], errors="coerce")
    eligible = set(reg.loc[reg["date_registration"].fillna(0) <= 0, "id_student"])

    for key in ["info", "registration", "assessment", "vle"]:
        tables[key] = tables[key][tables[key]["id_student"].isin(eligible)].copy()

    vle = tables["vle"].copy()
    vle["date"] = pd.to_numeric(vle["date"], errors="coerce")
    tables["vle"] = vle[(vle["date"] >= 0) & (vle["date"] <= presentation_length)].copy()
    return tables


def run(
    code_module: str | None = None,
    code_presentation: str | None = None,
    source_dir: Path | None = None,
) -> Tables:
    tables = load_oulad(code_module, code_presentation, source_dir)
    tables = apply_study_window(tables)
    tables = exclude_early_withdrawals(tables)
    return tables
