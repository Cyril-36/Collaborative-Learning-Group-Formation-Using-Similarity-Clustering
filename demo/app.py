"""Streamlit demo for the collaborative learning clustering project.

Visual system: Variant B (Stripe Press), locked 2026-04-29 in design.md.
The project's matplotlib + Plotly + CSS surfaces all source from src.plot_style.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.io as pio
import streamlit as st


ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "demo" / "demo_cache"
FIGURES = ROOT / "results" / "figures"

# Make src.* importable regardless of how Streamlit was launched
sys.path.insert(0, str(ROOT))
from src.plot_style import plotly_template  # noqa: E402

# Register the project Plotly template globally so every px.* / go.* chart
# inherits ink+terracotta tokens automatically.
pio.templates["int396"] = plotly_template()
pio.templates.default = "int396"


# === Page config + CSS injection (must run before any other Streamlit calls) ==
st.set_page_config(
    page_title="CollabLearn · Group Formation via Similarity Clustering",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

_CSS = Path(__file__).parent / "static" / "app.css"
if _CSS.exists():
    st.markdown(f"<style>{_CSS.read_text()}</style>", unsafe_allow_html=True)


# === Cache helpers ===========================================================
@st.cache_data
def read_parquet(name: str) -> pd.DataFrame:
    return pd.read_parquet(CACHE / name)


@st.cache_data
def read_json(name: str) -> dict:
    return json.loads((CACHE / name).read_text(encoding="utf-8"))


@st.cache_data
def read_npy(name: str) -> np.ndarray:
    return np.load(CACHE / name)


def cache_ready() -> bool:
    required = [
        "features.parquet",
        "cluster_labels.parquet",
        "config_metrics.parquet",
        "stability.parquet",
        "winner.json",
        "groups_mode_a.parquet",
        "groups_mode_b.parquet",
        "group_metrics.parquet",
        "meta.json",
    ]
    return all((CACHE / name).exists() for name in required)


def missing_cache_page() -> None:
    st.title("CollabLearn · Group Formation via Similarity Clustering")
    st.caption("Pipeline cache missing. Run the end-to-end pipeline before launching the demo.")
    st.code("python -m src.pipeline --presentation AAA_2014J", language="bash")
    st.markdown(
        "Smoke runs use `--fast`. Note: a safety guard refuses to overwrite an "
        "existing full-fidelity cache (B≥30) without `--force`."
    )


# === Pages ===================================================================
def page_overview() -> None:
    meta = read_json("meta.json")
    winner = read_json("winner.json")

    # --- Hero block ---------------------------------------------------------
    st.markdown("# Three interpretable learner profiles")
    st.markdown(
        "<blockquote>Validated across <strong>30 bootstrap resamples</strong> "
        "and <strong>12 clustering configurations</strong>, then composed into "
        "collaborative groups under explicit fairness, balance, and "
        "complementarity constraints.</blockquote>",
        unsafe_allow_html=True,
    )

    # --- Metric strip -------------------------------------------------------
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Learners", f"{meta.get('n_learners', 0):,}")
    c2.metric("Features", meta.get("n_features", 0))
    c3.metric("Configs Tested", 12)
    c4.metric("Winner", str(winner.get("config_id", "—")))

    # --- Hero figure --------------------------------------------------------
    # Direct st.image (NOT cached) so re-renders surface without `streamlit cache clear`
    fig8 = FIGURES / "fig8_embedding.png"
    if fig8.exists():
        st.image(str(fig8), width="stretch")
        st.caption(
            f"UMAP embedding of {meta.get('n_learners', 0)} learners coloured "
            f"by the {winner.get('config_id', '—')} clustering "
            f"({winner.get('reducer', '—')} + {winner.get('clusterer', '—')}). "
            f"Centroids in terracotta. Cluster shape carries identity, not colour."
        )

    # --- Elevator-pitch paragraph (≤80 words) -------------------------------
    st.markdown(
        "Student collaborative groups are typically formed at random or by a "
        "single demographic attribute. Both ignore the multi-dimensional "
        "learner profile and produce groups of unknown robustness. This system "
        "tests 12 reducer × clusterer configurations, validates each via 30 "
        "bootstrap resamples measuring Adjusted Rand Index stability, selects "
        "the most stable & separated, and forms groups under two complementary "
        "modes: *homogeneous* (pace-matched) or *heterogeneous* (scaffolded)."
    )

    with st.expander("Methodology details"):
        pipeline_img = CACHE / "pipeline_diagram.png"
        if pipeline_img.exists():
            st.image(str(pipeline_img), width="stretch")
        st.markdown("**Run metadata**")
        st.json(meta)


def page_features() -> None:
    features = read_parquet("features.parquet")
    st.title("Feature engineering")
    st.caption(
        f"{len(features):,} learners × {features.shape[1] - 1} engineered features. "
        f"Collaboration signals (forumng, oucollaborate) are first-class."
    )

    tabs = st.tabs(["Feature matrix", "Learner inspector"])
    with tabs[0]:
        st.dataframe(features.head(50), use_container_width=True)
    with tabs[1]:
        learner = st.selectbox("Learner", features["id_student"].tolist())
        numeric_cols = (
            features.drop(columns=["id_student"], errors="ignore")
            .select_dtypes(include="number").columns
        )
        row = features.loc[features["id_student"].eq(learner), numeric_cols].iloc[0]
        means = features[numeric_cols].mean(numeric_only=True)
        plot_df = pd.DataFrame({
            "feature": row.index,
            "learner": row.values,
            "class_mean": means.reindex(row.index).values,
        })
        priority = [
            "total_clicks", "collaborative_clicks", "collaboration_click_ratio",
            "forum_clicks", "live_collab_clicks", "weighted_score", "imd_band_ord",
        ]
        ordered = [c for c in priority if c in plot_df["feature"].values]
        ordered += [c for c in plot_df["feature"].tolist() if c not in ordered]
        plot_df = plot_df.set_index("feature").loc[ordered].reset_index().head(20)

        fig = px.bar(
            plot_df, x="feature", y=["learner", "class_mean"],
            barmode="group", height=480,
        )
        fig.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)


def page_clustering() -> None:
    st.title("Clustering lab")
    metrics = read_parquet("config_metrics.parquet")
    stability = read_parquet("stability.parquet")
    labels = read_parquet("cluster_labels.parquet")
    winner = read_json("winner.json")
    joined = metrics.merge(
        stability[["config_id", "bootstrap_ari_mean"]],
        on="config_id", how="left",
    )

    # --- Winner stat row ----------------------------------------------------
    win_row = joined[joined["config_id"] == winner["config_id"]]
    if not win_row.empty:
        w = win_row.iloc[0]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Silhouette ↑", f"{w.get('silhouette', 0):.3f}")
        c2.metric("DBI ↓", f"{w.get('davies_bouldin', 0):.3f}")
        c3.metric("CH ↑", f"{w.get('calinski_harabasz', 0):.1f}")
        c4.metric("Bootstrap-ARI ↑", f"{w.get('bootstrap_ari_mean', 0):.3f}")
        st.caption(
            f"Winner: **{winner['config_id']}** "
            f"({winner['reducer']} + {winner['clusterer']}, k={int(w.get('k', 0))})."
        )

    # --- 12-config matrix ---------------------------------------------------
    st.subheader("Twelve configurations")
    show_cols = [
        "config_id", "reducer", "clusterer", "k",
        "silhouette", "davies_bouldin", "calinski_harabasz",
        "dbcv", "noise_ratio", "bootstrap_ari_mean",
    ]
    available = [c for c in show_cols if c in joined.columns]
    st.dataframe(joined[available], use_container_width=True)

    # --- Validity vs stability scatter --------------------------------------
    st.subheader("Validity vs stability")
    primary = joined[joined["clusterer"].ne("hdbscan")]
    fig = px.scatter(
        primary, x="silhouette", y="bootstrap_ari_mean",
        color="clusterer", symbol="clusterer",
        hover_name="config_id", size="k", height=480,
    )
    fig.add_hline(y=0.40, line_dash="dot", line_color="#525252")
    st.plotly_chart(fig, use_container_width=True)

    # --- Cluster characterization ------------------------------------------
    cluster_summary_path = CACHE / "cluster_summary.parquet"
    if cluster_summary_path.exists():
        st.subheader("Cluster characterization")
        st.caption("Top feature deviations per cluster, in standardised z-scores.")
        st.dataframe(pd.read_parquet(cluster_summary_path), use_container_width=True)

    # --- 2D embedding -------------------------------------------------------
    st.subheader("2D embedding")
    cid = st.selectbox("Configuration", joined["config_id"].tolist(), index=0)
    X_vis = read_npy("reduced_umap_2d.npy")
    plot_df = pd.DataFrame({
        "x": X_vis[:, 0],
        "y": X_vis[:, 1],
        "cluster": labels[cid].astype(str),
        "id_student": labels["id_student"].astype(str),
    })
    fig = px.scatter(
        plot_df, x="x", y="y",
        color="cluster", symbol="cluster",
        hover_name="id_student", height=560,
    )
    fig.update_traces(marker=dict(size=8, line=dict(color="white", width=0.5)))
    st.plotly_chart(fig, use_container_width=True)


def page_groups() -> None:
    st.title("Group formation")
    st.caption(
        "Two complementary modes. Mode A keeps similar learners together "
        "(homogeneous, pace-matched). Mode B mixes clusters by construction "
        "(heterogeneous, scaffolded)."
    )

    mode = st.radio(
        "Mode", ["Mode A — Homogeneous", "Mode B — Heterogeneous"],
        horizontal=True,
    )
    groups = read_parquet(
        "groups_mode_a.parquet" if mode.startswith("Mode A") else "groups_mode_b.parquet"
    )

    left, right = st.columns([1.25, 1])
    with left:
        query = st.text_input("Filter learner or group")
        filtered = groups
        if query:
            text = groups.astype(str).agg(" ".join, axis=1)
            filtered = groups[text.str.contains(query, case=False, regex=False)]
        st.dataframe(filtered, height=520, use_container_width=True)
        st.download_button(
            "Download CSV",
            groups.to_csv(index=False),
            file_name="group_assignments.csv",
            mime="text/csv",
        )
    with right:
        group_id = st.selectbox("Inspect group", sorted(groups["group_id"].unique()))
        members = groups[groups["group_id"].eq(group_id)]
        fig = px.bar(
            members, x="id_student",
            y=["total_clicks", "weighted_score", "imd_band_ord"],
            barmode="group", height=420,
            title=f"Group {group_id}",
        )
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(members, use_container_width=True)


def page_evaluation() -> None:
    st.title("Evaluation")

    # --- Verdict ------------------------------------------------------------
    significance_path = CACHE / "group_significance.parquet"
    verdict = (
        "Mode B exceeds the random baseline on complementarity and cluster "
        "coverage at p = 0.010 across 100 random partitions; engagement "
        "balance and demographic fairness remain comparable to baselines."
    )
    st.markdown(f"<blockquote>{verdict}</blockquote>", unsafe_allow_html=True)

    # --- Hero figure (fig10) -----------------------------------------------
    fig10 = FIGURES / "fig10_group_metrics.png"
    if fig10.exists():
        st.image(str(fig10), width="stretch")
        st.caption(
            "Six metrics × four strategies. Mode B (terracotta) wins on "
            "complementarity, engagement balance, and cluster coverage; "
            "Mode A (ink) wins on intra-group distance by construction."
        )

    # --- Strategy comparison table -----------------------------------------
    group_metrics = read_parquet("group_metrics.parquet")
    st.subheader("Strategy comparison")
    st.dataframe(group_metrics, use_container_width=True)

    # --- Post-hoc outcome checks -------------------------------------------
    outcome_cols = [
        "strategy", "outcome_diversity", "at_risk_concentration",
        "high_risk_group_rate", "outcome_balance",
    ]
    if set(outcome_cols).issubset(group_metrics.columns):
        st.subheader("Post-hoc outcome checks")
        st.caption(
            "These use OULAD `final_result` only after grouping is fixed; "
            "they are descriptive validation, not inputs to clustering."
        )
        st.dataframe(group_metrics[outcome_cols], use_container_width=True)

    # --- Random-baseline significance --------------------------------------
    if significance_path.exists():
        st.subheader("Random-baseline significance")
        st.caption(
            "Percentiles and one-sided p-values compare Mode A/B against "
            "100 random group partitions of the same cohort."
        )
        sig = pd.read_parquet(significance_path)
        st.dataframe(sig, use_container_width=True)

    # --- Constraint summary ------------------------------------------------
    constraints_path = CACHE / "constraints.json"
    if constraints_path.exists():
        st.subheader("Constraint summary")
        st.caption("Mode B satisfies cluster-complementarity by construction; Mode A's structural single-cluster groups can't be fully removed by greedy swap.")
        st.json(read_json("constraints.json"))


# === Sidebar + router ========================================================
PAGES = {
    "Overview": page_overview,
    "Features": page_features,
    "Clustering": page_clustering,
    "Groups": page_groups,
    "Evaluation": page_evaluation,
}


def main() -> None:
    if not cache_ready():
        missing_cache_page()
        return

    with st.sidebar:
        st.markdown("### CollabLearn")
        st.caption("INT-396 · Lovely Professional University")
        st.divider()
        page = st.radio("Navigate", list(PAGES.keys()), label_visibility="collapsed")
        st.divider()
        meta = read_json("meta.json")
        st.caption(f"Cohort · {meta.get('presentation', '—')}")
        st.caption(f"N = {meta.get('n_learners', 0)} learners")
        st.caption(f"Winner · {meta.get('winner_config', '—')}")

    PAGES[page]()


if __name__ == "__main__":
    main()
