"""Configuration selection rules."""

from __future__ import annotations

import pandas as pd

from .config import MIN_STABILITY


def select_winning(metrics_df: pd.DataFrame, stability_df: pd.DataFrame):
    df = metrics_df.drop(columns=["labels"], errors="ignore").merge(
        stability_df[["config_id", "bootstrap_ari_mean"]],
        on="config_id",
        how="left",
    )
    eligible = df[df["bootstrap_ari_mean"].fillna(-1) >= MIN_STABILITY].copy()
    if eligible.empty:
        eligible = df.copy()

    eligible["selection_track"] = "primary"
    eligible.loc[eligible["clusterer"].eq("hdbscan"), "selection_track"] = "hdbscan_comparison"
    eligible["composite"] = float("nan")
    eligible["hdbscan_score"] = float("nan")

    primary = eligible[eligible["selection_track"].eq("primary")].copy()
    primary = primary.dropna(subset=["silhouette", "davies_bouldin", "calinski_harabasz"])
    if primary.empty:
        raise ValueError("No eligible primary-track configs after stability/metric filtering")

    primary["rank_sil"] = primary["silhouette"].rank(ascending=False, method="min")
    primary["rank_dbi"] = primary["davies_bouldin"].rank(ascending=True, method="min")
    primary["rank_ch"] = primary["calinski_harabasz"].rank(ascending=False, method="min")
    primary["rank_stab"] = primary["bootstrap_ari_mean"].rank(ascending=False, method="min")
    primary["composite"] = primary[["rank_sil", "rank_dbi", "rank_ch", "rank_stab"]].sum(axis=1)

    hdbscan = eligible[eligible["selection_track"].eq("hdbscan_comparison")].copy()
    if not hdbscan.empty:
        hdbscan["rank_dbcv"] = hdbscan["dbcv"].fillna(-1).rank(ascending=False, method="min")
        hdbscan["rank_noise"] = hdbscan["noise_ratio"].fillna(1).rank(ascending=True, method="min")
        hdbscan["rank_stab_hdb"] = hdbscan["bootstrap_ari_mean"].rank(ascending=False, method="min")
        hdbscan["hdbscan_score"] = hdbscan[["rank_dbcv", "rank_noise", "rank_stab_hdb"]].sum(axis=1)

    frames = [frame for frame in [primary, hdbscan] if not frame.empty]
    ranked = pd.concat(frames, ignore_index=True)
    ranked["selection_order"] = ranked["selection_track"].map({"primary": 0, "hdbscan_comparison": 1})
    ranked = ranked.sort_values(
        [
            "selection_order",
            "composite",
            "hdbscan_score",
            "bootstrap_ari_mean",
            "silhouette",
            "config_id",
        ],
        ascending=[True, True, True, False, False, True],
        na_position="last",
    ).drop(columns=["selection_order"])

    winner = ranked[ranked["selection_track"].eq("primary")].iloc[0]
    return winner, ranked
