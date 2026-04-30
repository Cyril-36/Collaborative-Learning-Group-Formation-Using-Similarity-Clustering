"""Learner-level feature engineering."""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

from .config import (
    ACTIVITY_TYPES_TOP_N,
    DEFAULT_ACTIVITY_TYPES,
    PRESENTATION_LENGTH,
    USE_REGION_ONEHOT,
)


AGE_ORD = {"0-35": 0, "35-55": 1, "55<=": 2}
EDU_ORD = {
    "No Formal quals": 0,
    "Lower Than A Level": 1,
    "A Level or Equivalent": 2,
    "HE Qualification": 3,
    "Post Graduate Qualification": 4,
}
IMD_ORD = {f"{i * 10}-{(i + 1) * 10}%": i for i in range(10)}
COLLAB_ACTIVITY_TYPES = {"forumng", "oucollaborate", "ouwiki"}


def _safe_slug(value: str) -> str:
    value = value.lower().strip()
    value = re.sub(r"[^a-z0-9]+", "_", value)
    return value.strip("_")


def demographic_features(info: pd.DataFrame, registration: pd.DataFrame | None = None) -> pd.DataFrame:
    df = info.copy()
    df["age_band_ord"] = df["age_band"].map(AGE_ORD)
    imd = df["imd_band"].map(IMD_ORD)
    df["imd_band_ord"] = imd.fillna(imd.median())
    df["highest_education_ord"] = df["highest_education"].map(EDU_ORD)
    df["disability_bin"] = df["disability"].eq("Y").astype(int)
    df["gender_M"] = df["gender"].eq("M").astype(int)
    df["num_prev_attempts"] = pd.to_numeric(df["num_of_prev_attempts"], errors="coerce")
    df["studied_credits"] = pd.to_numeric(df["studied_credits"], errors="coerce")

    cols = [
        "id_student",
        "age_band_ord",
        "imd_band_ord",
        "highest_education_ord",
        "disability_bin",
        "gender_M",
        "num_prev_attempts",
        "studied_credits",
    ]
    out = df[cols].copy()

    if registration is not None and not registration.empty:
        reg = registration[["id_student", "date_registration"]].copy()
        reg["registration_day"] = pd.to_numeric(reg["date_registration"], errors="coerce")
        out = out.merge(reg[["id_student", "registration_day"]], on="id_student", how="left")
    else:
        out["registration_day"] = np.nan

    if USE_REGION_ONEHOT:
        top_regions = df["region"].value_counts().head(5).index.tolist()
        for region in top_regions:
            out[f"region_{_safe_slug(region)}"] = df["region"].eq(region).astype(int).values
        out["region_other"] = (~df["region"].isin(top_regions)).astype(int).values
    return out


def engagement_features(
    vle: pd.DataFrame,
    vle_meta: pd.DataFrame,
    presentation_length: int = PRESENTATION_LENGTH,
) -> pd.DataFrame:
    if vle.empty:
        columns = [
            "id_student",
            "total_clicks",
            "active_days",
            "mean_clicks_per_active_day",
            "first_click_day",
            "last_click_day",
            "engagement_span",
            "click_std",
            "click_cv",
            "weekend_ratio",
        ] + [f"clicks_{name}" for name in DEFAULT_ACTIVITY_TYPES[:ACTIVITY_TYPES_TOP_N]]
        return pd.DataFrame(columns=columns)

    merged = vle.merge(
        vle_meta[["id_site", "activity_type"]].drop_duplicates("id_site"),
        on="id_site",
        how="left",
    )
    merged["sum_click"] = pd.to_numeric(merged["sum_click"], errors="coerce").fillna(0)
    merged["date"] = pd.to_numeric(merged["date"], errors="coerce")
    merged = merged[(merged["date"] >= 0) & (merged["date"] <= presentation_length)].copy()
    merged["activity_type"] = merged["activity_type"].fillna("unknown")

    agg = merged.groupby("id_student").agg(
        total_clicks=("sum_click", "sum"),
        active_days=("date", "nunique"),
        first_click_day=("date", "min"),
        last_click_day=("date", "max"),
    ).reset_index()
    agg["mean_clicks_per_active_day"] = agg["total_clicks"] / agg["active_days"].replace(0, np.nan)
    agg["engagement_span"] = agg["last_click_day"] - agg["first_click_day"]

    daily = merged.groupby(["id_student", "date"], as_index=False)["sum_click"].sum()
    dispersion = daily.groupby("id_student")["sum_click"].agg(["std", "mean"]).reset_index()
    dispersion.columns = ["id_student", "click_std", "click_mean_daily"]
    dispersion["click_cv"] = dispersion["click_std"] / dispersion["click_mean_daily"].replace(0, np.nan)
    agg = agg.merge(dispersion[["id_student", "click_std", "click_cv"]], on="id_student", how="left")

    merged["weekday"] = (merged["date"].astype(int) % 7).astype(int)
    merged["is_weekend"] = merged["weekday"].isin([5, 6]).astype(int)
    weekend = (
        merged.assign(weekend_clicks=merged["sum_click"] * merged["is_weekend"])
        .groupby("id_student")
        .agg(weekend_clicks=("weekend_clicks", "sum"), total=("sum_click", "sum"))
        .reset_index()
    )
    weekend["weekend_ratio"] = weekend["weekend_clicks"] / weekend["total"].replace(0, np.nan)
    agg = agg.merge(weekend[["id_student", "weekend_ratio"]], on="id_student", how="left")

    merged["is_collab"] = merged["activity_type"].isin(COLLAB_ACTIVITY_TYPES)
    merged["collaborative_click_component"] = np.where(merged["is_collab"], merged["sum_click"], 0)
    merged["forum_click_component"] = np.where(merged["activity_type"].eq("forumng"), merged["sum_click"], 0)
    merged["live_collab_click_component"] = np.where(
        merged["activity_type"].eq("oucollaborate"),
        merged["sum_click"],
        0,
    )
    collab = merged.groupby("id_student").agg(
        collaborative_clicks=("collaborative_click_component", "sum"),
        forum_clicks=("forum_click_component", "sum"),
        live_collab_clicks=("live_collab_click_component", "sum"),
    ).reset_index()
    collab_days = (
        merged[merged["is_collab"]]
        .groupby("id_student")["date"]
        .nunique()
        .reset_index(name="collaborative_active_days")
    )
    collab = collab.merge(collab_days, on="id_student", how="left")
    agg = agg.merge(collab, on="id_student", how="left")
    agg["collaboration_click_ratio"] = agg["collaborative_clicks"] / agg["total_clicks"].replace(0, np.nan)

    non_collab_activity = merged[~merged["activity_type"].isin(COLLAB_ACTIVITY_TYPES)]
    top_types = non_collab_activity["activity_type"].value_counts().head(ACTIVITY_TYPES_TOP_N).index.tolist()
    for fallback in DEFAULT_ACTIVITY_TYPES:
        if len(top_types) >= ACTIVITY_TYPES_TOP_N:
            break
        if fallback not in COLLAB_ACTIVITY_TYPES and fallback not in top_types:
            top_types.append(fallback)

    pivot = merged[merged["activity_type"].isin(top_types)].pivot_table(
        index="id_student",
        columns="activity_type",
        values="sum_click",
        aggfunc="sum",
        fill_value=0,
    )
    pivot = pivot.reindex(columns=top_types, fill_value=0)
    pivot.columns = [f"clicks_{_safe_slug(col)}" for col in pivot.columns]
    agg = agg.merge(pivot.reset_index(), on="id_student", how="left")
    return agg.fillna(0)


def performance_features(assessment: pd.DataFrame, assessments: pd.DataFrame) -> pd.DataFrame:
    if assessment.empty or assessments.empty:
        return pd.DataFrame(
            columns=[
                "id_student",
                "mean_tma_score",
                "weighted_score",
                "n_assessments_submitted",
                "mean_submission_lateness",
                "score_trajectory_slope",
                "no_submissions",
            ]
        )

    merged = assessment.merge(
        assessments[["id_assessment", "date", "weight", "assessment_type"]],
        on="id_assessment",
        how="left",
    )
    for col in ["score", "weight", "date", "date_submitted"]:
        merged[col] = pd.to_numeric(merged[col], errors="coerce")
    merged = merged.dropna(subset=["id_student"])
    merged["lateness"] = (merged["date_submitted"] - merged["date"]).clip(lower=0)

    def per_learner(group: pd.DataFrame) -> pd.Series:
        tma = group[group["assessment_type"].eq("TMA")]
        valid_weighted = group.dropna(subset=["score", "weight"])
        weight_sum = valid_weighted["weight"].sum()
        valid_slope = group.dropna(subset=["date_submitted", "score"])
        if len(valid_slope) >= 2 and valid_slope["date_submitted"].nunique() >= 2:
            slope = float(np.polyfit(valid_slope["date_submitted"], valid_slope["score"], 1)[0])
        else:
            slope = 0.0
        return pd.Series(
            {
                "mean_tma_score": tma["score"].mean() if len(tma) else np.nan,
                "weighted_score": (valid_weighted["score"] * valid_weighted["weight"]).sum() / weight_sum
                if weight_sum > 0
                else np.nan,
                "n_assessments_submitted": float(len(group)),
                "mean_submission_lateness": group["lateness"].mean(),
                "score_trajectory_slope": slope,
            }
        )

    out = merged.groupby("id_student").apply(per_learner, include_groups=False).reset_index()
    out["no_submissions"] = 0
    return out


def build_feature_matrix(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    demo = demographic_features(tables["info"], tables.get("registration"))
    engage = engagement_features(tables["vle"], tables["vle_meta"])
    perf = performance_features(tables["assessment"], tables["assessments"])

    matrix = demo.merge(engage, on="id_student", how="left").merge(perf, on="id_student", how="left")
    matrix["no_submissions"] = matrix["n_assessments_submitted"].isna().astype(int)
    matrix["n_assessments_submitted"] = matrix["n_assessments_submitted"].fillna(0)
    numeric = matrix.columns.drop("id_student")
    matrix[numeric] = matrix[numeric].apply(pd.to_numeric, errors="coerce")
    return matrix


def run(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    return build_feature_matrix(tables)
