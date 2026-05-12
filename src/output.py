"""Persist pipeline outputs for reports and the Streamlit demo."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .adapters.base import DatasetSchema
from .config import DEMO_CACHE, FIGURES, GROUP_SIZE, RESULTS, TABLES, ensure_dirs
from .plot_style import (
    ACCENT,
    CENTROID_COLOR,
    CLUSTER_FILLS,
    CLUSTER_MARKERS,
    INK,
    INK_FAINT,
    INK_MUTED,
    RULE,
    SURFACE,
    apply_style,
    use_mono_ticks,
)

# Apply project rcParams once at module import. Idempotent.
apply_style()


def _json_safe(value: Any) -> Any:
    if value is None or value is pd.NA:
        return None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        value = float(value)
        if not math.isfinite(value):
            return None
        return value
    if isinstance(value, (np.bool_)):
        return bool(value)
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, pd.Series):
        return {key: _json_safe(val) for key, val in value.to_dict().items()}
    if isinstance(value, dict):
        return {str(key): _json_safe(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return value


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(_json_safe(payload), indent=2, allow_nan=False), encoding="utf-8")


def groups_to_frame(
    ids: np.ndarray,
    labels: np.ndarray,
    groups: list[list[int]],
    features: pd.DataFrame,
    schema: DatasetSchema | None = None,
) -> pd.DataFrame:
    rows = []
    feature_lookup = features.reset_index(drop=True)
    if schema is not None:
        display_cols = [col for col in schema.display_cols if col in feature_lookup.columns]
    else:
        display_cols = [
            col for col in [
                "total_clicks",
                "weighted_score",
                "imd_band_ord",
                "collaborative_clicks",
                "collaboration_click_ratio",
                "final_result",
            ]
            if col in feature_lookup.columns
        ]
    for group_id, members in enumerate(groups, start=1):
        for idx in members:
            row = {
                "id_student": ids[idx],
                "group_id": group_id,
                "cluster": int(labels[idx]),
            }
            for col in display_cols:
                row[col] = feature_lookup.loc[idx, col]
            rows.append(row)
    return pd.DataFrame(rows)


def write_pipeline_diagram(path: Path) -> None:
    labels = [
        "Learner dataset",
        "Adapter",
        "Features",
        "Preprocess",
        "Reducers",
        "Clusterers",
        "Stability",
        "Selector",
        "Groups",
        "Evaluation",
    ]
    fig, ax = plt.subplots(figsize=(12, 2.4))
    ax.axis("off")
    xs = np.linspace(0.05, 0.95, len(labels))
    for x, label in zip(xs, labels):
        ax.text(
            x,
            0.55,
            label,
            ha="center",
            va="center",
            bbox=dict(
                boxstyle="round,pad=0.35",
                facecolor=SURFACE,
                edgecolor=INK_MUTED,
                linewidth=0.8,
            ),
            fontsize=9,
            color=INK,
        )
    for x1, x2 in zip(xs[:-1], xs[1:]):
        ax.annotate(
            "",
            xy=(x2 - 0.035, 0.55),
            xytext=(x1 + 0.035, 0.55),
            arrowprops=dict(arrowstyle="->", color=INK_MUTED, linewidth=0.6),
        )
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _fig7_validity_vs_stability(
    metrics_df: pd.DataFrame,
    stability_df: pd.DataFrame,
    winner: pd.Series,
) -> None:
    """Validity (silhouette) vs Stability (bootstrap-ARI) scatter, primary configs."""
    joined = metrics_df.drop(columns=["labels"], errors="ignore").merge(
        stability_df[["config_id", "bootstrap_ari_mean"]],
        on="config_id",
        how="left",
    )
    primary = joined[joined["clusterer"].ne("hdbscan")]
    if primary.empty:
        return

    fig, ax = plt.subplots(figsize=(8, 5.5))

    # All points in ink; differentiate clusterers by marker shape only
    clusterer_markers = {"kmeans": "o", "gmm": "s", "agglo": "^"}
    for clusterer, sub in primary.groupby("clusterer"):
        ax.scatter(
            sub["silhouette"],
            sub["bootstrap_ari_mean"],
            s=64,
            marker=clusterer_markers.get(clusterer, "o"),
            facecolor=INK,
            edgecolor="white",
            linewidth=0.8,
            label=clusterer,
            alpha=0.85,
        )

    # Stability threshold reference line
    ax.axhline(0.40, linestyle="--", color=INK_MUTED, linewidth=0.8)
    ax.text(
        0.005, 0.40, "stability floor (ARI = 0.40)",
        transform=ax.get_yaxis_transform(),
        ha="left", va="bottom",
        fontsize=8, color=INK_FAINT, style="italic",
    )

    # Highlight the winner — terracotta diamond, leader line, label
    winner_id = winner.get("config_id")
    win_row = primary[primary["config_id"] == winner_id]
    if not win_row.empty:
        wx = float(win_row["silhouette"].iloc[0])
        wy = float(win_row["bootstrap_ari_mean"].iloc[0])
        ax.scatter(
            wx, wy, marker="D", s=120,
            facecolor=ACCENT, edgecolor="white", linewidth=1.2, zorder=10,
        )
        ax.annotate(
            f"{winner_id} (winner)",
            xy=(wx, wy),
            xytext=(wx + 0.04, wy - 0.06),
            fontsize=10, color=INK, weight=600,
            arrowprops=dict(arrowstyle="-", color=INK_MUTED, linewidth=0.6),
        )

    # Tiny config_id labels on non-winner points
    for _, row in primary.iterrows():
        if row["config_id"] == winner_id:
            continue
        ax.annotate(
            row["config_id"],
            xy=(row["silhouette"], row["bootstrap_ari_mean"]),
            xytext=(5, 4),
            textcoords="offset points",
            fontsize=8, color=INK_FAINT,
        )

    ax.set_xlabel("silhouette  (higher is better)")
    ax.set_ylabel("bootstrap-ARI mean  (higher is better)")
    ax.grid(axis="x", visible=False)
    ax.legend(loc="lower right", title=None)
    use_mono_ticks(ax)

    fig.tight_layout()
    fig.savefig(FIGURES / "fig7_scatter.png", dpi=200)
    plt.close(fig)


def _fig8_embedding(
    labels: np.ndarray,
    X_vis: np.ndarray,
    winner: pd.Series,
) -> None:
    """UMAP 2D embedding — the hero figure.

    Cluster differentiation by SHAPE (per design.md §5.3 amendment), not colour.
    All centroids in terracotta accent — signals "structural finding," not "this
    cluster is the bad one."
    """
    if X_vis.shape[1] < 2:
        return

    labels_arr = np.asarray(labels)
    fig, ax = plt.subplots(figsize=(8, 8))

    for cluster_id in sorted(set(int(l) for l in labels_arr)):
        mask = labels_arr == cluster_id
        if not mask.any():
            continue
        ax.scatter(
            X_vis[mask, 0], X_vis[mask, 1],
            s=18, alpha=0.7,
            marker=CLUSTER_MARKERS.get(cluster_id, "o"),
            facecolor=CLUSTER_FILLS.get(cluster_id, INK_MUTED),
            edgecolor="white", linewidth=0.5,
        )
        # Centroid (skip noise cluster -1)
        if cluster_id != -1:
            cx, cy = X_vis[mask, 0].mean(), X_vis[mask, 1].mean()
            ax.scatter(
                cx, cy, marker="D", s=90,
                facecolor=CENTROID_COLOR, edgecolor="white",
                linewidth=1.2, zorder=10,
            )
            # In-cluster size annotation near each centroid; whitespace, no box
            ax.annotate(
                f"cluster {cluster_id}  n={int(mask.sum())}",
                xy=(cx, cy),
                xytext=(10, 10),
                textcoords="offset points",
                fontsize=10, color=INK, weight=600,
            )

    ax.set_xlabel("umap-1")
    ax.set_ylabel("umap-2")
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(axis="x", visible=False)
    use_mono_ticks(ax)

    # Caption sourced from data — never hard-code n
    ax.text(
        0.99, 0.01,
        f"n = {len(labels_arr)}  ·  winner = {winner.get('config_id', '—')}",
        transform=ax.transAxes,
        ha="right", va="bottom",
        fontsize=9, color=INK_FAINT, family="monospace",
    )

    fig.tight_layout()
    fig.savefig(FIGURES / "fig8_embedding.png", dpi=200)
    plt.close(fig)


def _fig10_group_metrics(group_metrics: pd.DataFrame) -> None:
    """Group metrics by strategy — 2×3 small-multiples grid.

    One metric per panel. Mode B is the only terracotta bar; everything else
    rides the ink ramp. Reading order: top-row = compactness/separation,
    bottom-row = social/equity metrics.
    """
    if group_metrics.empty:
        return

    metric_order = [
        "intra_group_distance",
        "inter_group_variance",
        "complementarity",
        "engagement_balance",
        "demographic_fairness",
        "cluster_coverage",
    ]
    strategies = ["random", "stratified", "mode_a", "mode_b"]
    strategy_labels = ["random", "stratified", "mode A", "mode B"]
    bar_colors = [INK_FAINT, INK_MUTED, INK, ACCENT]

    fig, axes = plt.subplots(2, 3, figsize=(13, 7.5))
    for ax, metric in zip(axes.flat, metric_order):
        if metric not in group_metrics.columns:
            ax.set_visible(False)
            continue
        vals = []
        present_labels = []
        present_colors = []
        for s, lbl, c in zip(strategies, strategy_labels, bar_colors):
            row = group_metrics.loc[group_metrics["strategy"] == s, metric]
            if row.empty:
                continue
            vals.append(float(row.iloc[0]))
            present_labels.append(lbl)
            present_colors.append(c)
        ax.bar(
            range(len(vals)),
            vals,
            color=present_colors,
            edgecolor="white",
            linewidth=0.5,
        )
        ax.set_xticks(range(len(vals)))
        ax.set_xticklabels(present_labels, fontsize=10)
        ax.set_title(metric.replace("_", " "), fontsize=11, loc="left", color=INK)
        ax.tick_params(labelsize=9)
        use_mono_ticks(ax)
        # Drop x-grid; small-multiples already have visual rhythm from the grid
        ax.grid(axis="x", visible=False)

    fig.suptitle("")
    fig.tight_layout()
    fig.savefig(FIGURES / "fig10_group_metrics.png", dpi=200)
    plt.close(fig)


def _fig2_feature_families(columns: list[str]) -> None:
    """Horizontal bar chart showing feature count by family."""
    families = {
        "Demographic": [
            "age_band_ord", "imd_band_ord", "highest_education_ord",
            "disability_bin", "gender_M", "num_prev_attempts", "studied_credits",
            "registration_day",
        ],
        "Engagement": [
            "total_clicks", "active_days", "first_click_day", "last_click_day",
            "mean_clicks_per_active_day", "engagement_span", "click_std",
            "click_cv", "weekend_ratio",
        ],
        "Collaboration": [
            "collaborative_clicks", "forum_clicks", "live_collab_clicks",
            "collaborative_active_days", "collaboration_click_ratio",
        ],
        "VLE activity": [
            "clicks_oucontent", "clicks_homepage", "clicks_subpage",
            "clicks_url", "clicks_resource", "clicks_dataplus", "clicks_glossary",
        ],
        "Performance": [
            "mean_tma_score", "weighted_score", "n_assessments_submitted",
            "mean_submission_lateness", "score_trajectory_slope", "no_submissions",
        ],
    }
    col_set = set(columns)
    family_names = []
    family_counts = []
    for name, members in families.items():
        count = sum(1 for m in members if m in col_set)
        if count > 0:
            family_names.append(name)
            family_counts.append(count)

    # Any unclassified columns
    classified = set()
    for members in families.values():
        classified.update(members)
    unclassified = col_set - classified
    if unclassified:
        family_names.append("Other")
        family_counts.append(len(unclassified))

    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.barh(
        range(len(family_names)), family_counts,
        color=[ACCENT if n == "Collaboration" else INK_MUTED for n in family_names],
        edgecolor="white", linewidth=0.5,
    )
    ax.set_yticks(range(len(family_names)))
    ax.set_yticklabels(family_names)
    ax.set_xlabel("number of features")
    ax.invert_yaxis()
    ax.grid(axis="y", visible=False)

    # Count labels on bars
    for bar, count in zip(bars, family_counts):
        ax.text(
            bar.get_width() + 0.15, bar.get_y() + bar.get_height() / 2,
            str(count), va="center", fontsize=10, color=INK,
        )

    use_mono_ticks(ax)
    fig.tight_layout()
    fig.savefig(FIGURES / "fig2_feature_families.png", dpi=200)
    plt.close(fig)


def _fig3_config_heatmap(
    metrics_df: pd.DataFrame,
    stability_df: pd.DataFrame,
    winner: pd.Series,
) -> None:
    """12-config matrix heatmap — composite metrics, winner row ringed."""
    joined = metrics_df.drop(columns=["labels"], errors="ignore").merge(
        stability_df[["config_id", "bootstrap_ari_mean"]],
        on="config_id", how="left",
    )
    display_metrics = [
        "silhouette", "davies_bouldin", "calinski_harabasz", "bootstrap_ari_mean",
    ]
    available = [m for m in display_metrics if m in joined.columns]
    if not available:
        return

    matrix = joined.set_index("config_id")[available].copy()
    # Normalise each column to [0, 1] for visual comparability
    for col in available:
        mn, mx = matrix[col].min(), matrix[col].max()
        if mx > mn:
            # DBI is lower-is-better: invert
            if col == "davies_bouldin":
                matrix[col] = 1.0 - (matrix[col] - mn) / (mx - mn)
            else:
                matrix[col] = (matrix[col] - mn) / (mx - mn)
        else:
            matrix[col] = 0.5

    fig, ax = plt.subplots(figsize=(9, 6))
    im = ax.imshow(matrix.values, aspect="auto", cmap="Greys", vmin=0, vmax=1)

    ax.set_xticks(range(len(available)))
    ax.set_xticklabels([m.replace("_", "\n") for m in available], fontsize=9)
    ax.set_yticks(range(len(matrix)))
    ax.set_yticklabels(matrix.index, fontsize=10)

    # Annotate cells with raw values from joined
    raw = joined.set_index("config_id")[available]
    for i, config_id in enumerate(matrix.index):
        for j, col in enumerate(available):
            val = raw.loc[config_id, col]
            fmt = f"{val:.3f}" if abs(val) < 10 else f"{val:.1f}"
            text_color = "white" if matrix.values[i, j] > 0.6 else INK
            ax.text(j, i, fmt, ha="center", va="center", fontsize=8, color=text_color)

    # Ring the winner row
    winner_id = winner.get("config_id")
    if winner_id in matrix.index:
        row_idx = list(matrix.index).index(winner_id)
        ax.add_patch(plt.Rectangle(
            (-0.5, row_idx - 0.5), len(available), 1,
            fill=False, edgecolor=ACCENT, linewidth=2.0, zorder=10,
        ))

    ax.set_title("Normalised validity metrics (higher = better)", loc="left", fontsize=12)
    use_mono_ticks(ax)
    fig.colorbar(im, ax=ax, shrink=0.6, label="normalised score")
    fig.tight_layout()
    fig.savefig(FIGURES / "fig3_config_heatmap.png", dpi=200)
    plt.close(fig)


def _fig9_bootstrap_boxplot(
    stability_df: pd.DataFrame,
    winner: pd.Series | None = None,
) -> None:
    """Bootstrap-ARI distribution boxplot, sorted by mean ARI descending."""
    if stability_df.empty or "bootstrap_ari_dist" not in stability_df.columns:
        return

    # Sort by mean ARI descending
    ordered = stability_df.sort_values("bootstrap_ari_mean", ascending=False)
    config_ids = ordered["config_id"].tolist()
    distributions = []
    for _, row in ordered.iterrows():
        dist = row["bootstrap_ari_dist"]
        if isinstance(dist, (list, np.ndarray)):
            distributions.append(np.asarray(dist))
        else:
            distributions.append(np.array([row.get("bootstrap_ari_mean", 0)]))

    fig, ax = plt.subplots(figsize=(10, 5.5))
    bp = ax.boxplot(
        distributions,
        vert=True,
        patch_artist=True,
        labels=config_ids,
        widths=0.6,
    )

    # Style boxes — highlight the actual winner, not just highest ARI
    winner_id = winner.get("config_id") if winner is not None else None
    for i, (box, median) in enumerate(zip(bp["boxes"], bp["medians"])):
        if config_ids[i] == winner_id:
            box.set(facecolor=ACCENT, alpha=0.4)
            median.set(color=ACCENT, linewidth=1.5)
        else:
            box.set(facecolor=SURFACE, alpha=0.8)
            median.set(color=INK, linewidth=1.0)
        box.set(edgecolor=INK_MUTED, linewidth=0.8)

    for whisker in bp["whiskers"]:
        whisker.set(color=INK_MUTED, linewidth=0.6)
    for cap in bp["caps"]:
        cap.set(color=INK_MUTED, linewidth=0.6)
    for flier in bp["fliers"]:
        flier.set(marker="o", markersize=3, markerfacecolor=INK_FAINT,
                  markeredgecolor="none", alpha=0.6)

    # Stability floor
    ax.axhline(0.40, linestyle="--", color=INK_MUTED, linewidth=0.8)
    ax.text(
        len(distributions) + 0.3, 0.40, "stability floor",
        va="bottom", fontsize=8, color=INK_FAINT, style="italic",
    )

    ax.set_ylabel("pairwise Adjusted Rand Index")
    ax.set_xlabel("configuration (sorted by mean ARI)")
    ax.tick_params(axis="x", labelrotation=45)
    ax.grid(axis="x", visible=False)
    use_mono_ticks(ax)

    fig.tight_layout()
    fig.savefig(FIGURES / "fig9_bootstrap_boxplot.png", dpi=200)
    plt.close(fig)


def write_figures(
    metrics_df: pd.DataFrame,
    stability_df: pd.DataFrame,
    labels: np.ndarray,
    X_vis: np.ndarray,
    group_metrics: pd.DataFrame,
    winner: pd.Series,
    columns: list[str] | None = None,
) -> None:
    """Write all matplotlib figures. Pass 1 (fig7/8/10) + Pass 2 (fig2/3/9).

    All use Variant B tokens via plot_style.
    """
    FIGURES.mkdir(parents=True, exist_ok=True)
    _fig7_validity_vs_stability(metrics_df, stability_df, winner)
    _fig8_embedding(labels, X_vis, winner)
    _fig10_group_metrics(group_metrics)

    # Pass 2 figures
    if columns is not None:
        _fig2_feature_families(columns)
    _fig3_config_heatmap(metrics_df, stability_df, winner)
    _fig9_bootstrap_boxplot(stability_df, winner=winner)


def write_report(
    path: Path,
    winner: pd.Series,
    ranked: pd.DataFrame,
    group_metrics: pd.DataFrame,
    constraints: dict[str, Any],
) -> None:
    lines = [
        "# Pipeline Run Report",
        "",
        f"Selected config: {winner['config_id']} ({winner['reducer']} + {winner['clusterer']})",
        "",
        "## Top Ranked Configurations",
        "",
        "```text",
        ranked.head(12).to_string(index=False),
        "```",
        "",
        "## Group Metrics",
        "",
        "```text",
        group_metrics.to_string(index=False),
        "```",
        "",
        "## Constraint Summary",
        "",
        "```json",
        json.dumps(_json_safe(constraints), indent=2, allow_nan=False),
        "```",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def write(
    ids: np.ndarray,
    features: pd.DataFrame,
    X_scaled: np.ndarray,
    reductions: dict[str, np.ndarray],
    labels_by_config: dict[str, np.ndarray],
    metrics_df: pd.DataFrame,
    stability_df: pd.DataFrame,
    winner: pd.Series,
    ranked: pd.DataFrame,
    winner_labels: np.ndarray,
    groups_a: list[list[int]],
    groups_b: list[list[int]],
    group_metrics: pd.DataFrame,
    random_baseline_metrics: pd.DataFrame | None,
    group_significance: pd.DataFrame | None,
    cluster_summary: pd.DataFrame | None,
    constraints: dict[str, Any],
    run_metadata: dict[str, Any] | None = None,
    cache_dir: Path = DEMO_CACHE,
    columns: list[str] | None = None,
    schema: DatasetSchema | None = None,
) -> None:
    ensure_dirs()
    cache_dir.mkdir(parents=True, exist_ok=True)
    write_global_artifacts = cache_dir.resolve() == DEMO_CACHE.resolve()

    features.to_parquet(cache_dir / "features.parquet", index=False)
    np.save(cache_dir / "X_scaled.npy", X_scaled)
    for name, values in reductions.items():
        np.save(cache_dir / f"reduced_{name}.npy", values)
    if "umap_2d" in reductions:
        np.save(cache_dir / "reduced_umap_2d.npy", reductions["umap_2d"])
    elif "pca" in reductions:
        np.save(cache_dir / "reduced_umap_2d.npy", reductions["pca"][:, :2])

    cluster_labels = pd.DataFrame({"id_student": ids})
    for cid, labels in labels_by_config.items():
        cluster_labels[cid] = labels
    cluster_labels.to_parquet(cache_dir / "cluster_labels.parquet", index=False)

    metrics_clean = metrics_df.drop(columns=["labels"], errors="ignore")
    metrics_clean.to_parquet(cache_dir / "config_metrics.parquet", index=False)
    if write_global_artifacts:
        metrics_clean.to_csv(TABLES / "config_metrics.csv", index=False)
    stability_df.to_parquet(cache_dir / "stability.parquet", index=False)
    if write_global_artifacts:
        stability_df.drop(columns=["bootstrap_ari_dist"], errors="ignore").to_csv(
            TABLES / "stability.csv",
            index=False,
        )
    ranked.to_parquet(cache_dir / "ranked_configs.parquet", index=False)
    _write_json(cache_dir / "winner.json", winner.to_dict())

    groups_a_df = groups_to_frame(ids, winner_labels, groups_a, features, schema)
    groups_b_df = groups_to_frame(ids, winner_labels, groups_b, features, schema)
    groups_a_df.to_parquet(cache_dir / "groups_mode_a.parquet", index=False)
    groups_b_df.to_parquet(cache_dir / "groups_mode_b.parquet", index=False)
    if write_global_artifacts:
        groups_a_df.to_csv(TABLES / "groups_mode_a.csv", index=False)
        groups_b_df.to_csv(TABLES / "groups_mode_b.csv", index=False)

    group_metrics.to_parquet(cache_dir / "group_metrics.parquet", index=False)
    if write_global_artifacts:
        group_metrics.to_csv(TABLES / "group_metrics.csv", index=False)
    if random_baseline_metrics is not None:
        random_baseline_metrics.to_parquet(cache_dir / "random_baseline_metrics.parquet", index=False)
        if write_global_artifacts:
            random_baseline_metrics.to_csv(TABLES / "random_baseline_metrics.csv", index=False)
    if group_significance is not None:
        group_significance.to_parquet(cache_dir / "group_significance.parquet", index=False)
        if write_global_artifacts:
            group_significance.to_csv(TABLES / "group_significance.csv", index=False)
    if cluster_summary is not None:
        cluster_summary.to_parquet(cache_dir / "cluster_summary.parquet", index=False)
        if write_global_artifacts:
            cluster_summary.to_csv(TABLES / "cluster_summary.csv", index=False)

    if schema is not None:
        _write_json(cache_dir / "schema.json", schema.to_dict())

    dataset_name = schema.dataset_name if schema is not None else winner.get("dataset_name", "")
    adapter_name = schema.adapter_name if schema is not None else winner.get("adapter_name", "")
    meta = {
        "n_learners": int(len(ids)),
        "n_features": int(len(columns) if columns is not None else len([col for col in features.columns if col != "id_student"])),
        "n_groups": int(len(groups_b)),
        "group_size": GROUP_SIZE,
        "dataset_name": dataset_name,
        "adapter_name": adapter_name,
        "presentation": f"{winner.get('presentation_module', '')}_{winner.get('presentation_code', '')}".strip("_"),
        "winner_config": winner.get("config_id"),
        "winner_reducer": winner.get("reducer"),
        "winner_clusterer": winner.get("clusterer"),
    }
    if schema is not None:
        meta.update(
            {
                "source_id_col": schema.source_id_col,
                "id_col": schema.id_col,
                "fairness_cols": schema.fairness_cols,
                "engagement_col": schema.engagement_col,
                "performance_col": schema.performance_col,
                "outcome_col": schema.outcome_col,
                "stratification_col": schema.stratification_col,
                "display_cols": schema.display_cols,
            }
        )
    if run_metadata:
        meta.update(run_metadata)
    _write_json(cache_dir / "meta.json", meta)
    _write_json(cache_dir / "constraints.json", constraints)

    # Persist feature columns for regen_figures.py
    if columns is not None:
        _write_json(cache_dir / "columns.json", columns)

    write_pipeline_diagram(cache_dir / "pipeline_diagram.png")
    if write_global_artifacts:
        write_figures(
            metrics_clean,
            stability_df,
            winner_labels,
            reductions.get("umap_2d", reductions.get("pca", X_scaled[:, :2])),
            group_metrics,
            winner,
            columns=columns,
        )

        # Graphviz diagrams (fig1/4/5/6)
        try:
            from .diagrams import render_all
            render_all(FIGURES)
        except ImportError:
            pass  # graphviz not installed — skip flowcharts

        write_report(RESULTS / "pipeline_report.md", winner, ranked, group_metrics, constraints)
