"""Streamlit demo for the dataset-aware learner grouping pipeline.

Visual system: Linear / Vercel anchor (dark · clinical · projection-sized),
locked 2026-05-09 in design.md §1. Tokens live in src.plot_style; CSS lives
in demo/static/app.css; both are sole sources of truth — never inline a
colour or font here.

Page order (rehearsed for an 8-10 minute live demo):

    1. Overview         — what this is, hero metrics, route-to-predict
    2. Live predict     — *the* model moment; sliders + presets + UMAP dot
    3. Clustering       — 12-config sweep, winner row in cyan
    4. Groups           — Mode A/B with click-to-inspect group
    5. Evaluation       — significance table; cyan accent on p ≤ 0.05

The previous Stripe-Press version is preserved as demo/app.stripe.py.bak —
restore with: cp demo/app.stripe.py.bak demo/app.py.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st


ROOT = Path(__file__).resolve().parent.parent
CACHE = Path(os.getenv("INT396_DEMO_CACHE", str(ROOT / "demo" / "demo_cache")))
FIGURES = ROOT / "results" / "figures"

sys.path.insert(0, str(ROOT))
from src import group_former  # noqa: E402
from src import predict as predict_mod  # noqa: E402
from src.plot_style import ACCENT, BG, INK, INK_FAINT, INK_MUTED, RULE, SURFACE, plotly_template  # noqa: E402

# Register Plotly template globally so every px / go chart inherits it.
pio.templates["int396"] = plotly_template()
pio.templates.default = "int396"


# === Page config + CSS injection (must run before any other Streamlit calls) ==
st.set_page_config(
    page_title="CollabLearn — Learner grouping",
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


@st.cache_resource
def load_predict_artifacts():
    """Cache the joblib-deserialised artifacts across reruns. ~80 KB on disk."""
    try:
        return predict_mod.load_artifacts(CACHE)
    except FileNotFoundError:
        return None


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


def predict_ready() -> bool:
    return (CACHE / "predict_artifacts.joblib").exists()


# === Adapter copy ============================================================
ADAPTER_COPY = {
    "oulad": {
        "subtitle": "OULAD reference case study",
        "verdict": (
            "The system makes two useful kinds of study groups. One puts similar "
            "students together for pace-matched work. The other mixes learner "
            "types for peer support. Both are compared against random grouping."
        ),
        "outcome_caption": (
            "Final results are checked only after groups are formed. They are "
            "not used to create the clusters."
        ),
    },
    "generic_csv": {
        "subtitle": "Generic learner-level CSV",
        "verdict": (
            "The same workflow also runs on another learner dataset. Treat this "
            "as evidence that the system transfers, not as a claim that the "
            "second dataset is easier or better."
        ),
        "outcome_caption": (
            "Outcome columns are checked after grouping and are never used as "
            "inputs to make the clusters."
        ),
    },
}


def adapter_copy() -> dict[str, str]:
    meta = read_json("meta.json")
    return ADAPTER_COPY.get(meta.get("adapter_name", ""), ADAPTER_COPY["generic_csv"])


# === Visual helpers ==========================================================
SIG_THRESHOLD = 0.05


def hero_metric(label: str, value: str, footnote: str = "", accent: bool = False) -> str:
    """Custom hero-metric HTML — bigger than st.metric, used on Overview hero."""
    cls = "hero-metric" + (" is-accent" if accent else "")
    foot = f'<div class="footnote">{footnote}</div>' if footnote else ""
    return f"""
    <div class="{cls}">
      <div class="label">{label}</div>
      <div class="value">{value}</div>
      {foot}
    </div>
    """


def eyebrow(text: str) -> None:
    st.markdown(f'<div class="eyebrow">{text}</div>', unsafe_allow_html=True)


def badge(text: str, accent: bool = False) -> str:
    cls = "badge-accent" if accent else "badge-neutral"
    return f'<span class="{cls}">{text}</span>'


def confidence_label(confidence: str) -> str:
    return {
        "high": "clear match",
        "medium": "borderline match",
        "low": "weak match",
    }.get(confidence, confidence)


def friendly_feature_label(name: str) -> str:
    """Turn source column names into labels a non-technical audience can read."""
    labels = {
        "total_clicks": "Total activity",
        "active_days": "Active study days",
        "weighted_score": "Assessment score",
        "n_assessments_submitted": "Assessments submitted",
        "engagement_span": "Study activity span",
        "last_click_day": "Last active day",
        "first_click_day": "First active day",
        "score_trajectory_slope": "Score trend",
        "mean_submission_lateness": "Average lateness",
        "no_submissions": "Missing submissions",
        "clicks_oucontent": "Content views",
        "collaborative_active_days": "Collaborative days",
        "registration_day": "Registration timing",
        "studied_credits": "Study credits",
        "imd_band_ord": "Socioeconomic band",
        "highest_education_ord": "Education level",
        "age_band_ord": "Age band",
    }
    return labels.get(name, name.replace("_", " ").title())


def friendly_role_label(role: str) -> str:
    return {
        "engagement": "activity signal",
        "performance": "assessment signal",
        "fairness": "context signal",
        "feature": "learner signal",
    }.get(role, role)


def friendly_feature_list(text: Any) -> str:
    """Convert 'feature (+0.82z), other (-0.2z)' into plain feature names."""
    if not isinstance(text, str) or not text:
        return "—"
    names = []
    for part in text.split(","):
        raw = part.strip().split(" (", 1)[0]
        if raw:
            names.append(friendly_feature_label(raw))
    return ", ".join(names) if names else "—"


def friendly_strategy_label(name: str) -> str:
    return {
        "mode_a": "Similar groups",
        "mode_b": "Mixed groups",
        "random": "Random grouping",
        "stratified": "Stratified baseline",
    }.get(str(name), str(name).replace("_", " ").title())


def friendly_metric_label(name: str) -> str:
    text = str(name)
    if text.startswith("demographic_fairness_"):
        suffix = text.removeprefix("demographic_fairness_")
        return f"Fairness: {friendly_feature_label(suffix)}"
    return {
        "intra_group_distance": "Tightness",
        "inter_group_variance": "Between-group similarity",
        "complementarity": "Profile mix",
        "cluster_coverage": "Variety coverage",
        "engagement_balance": "Activity balance",
        "demographic_fairness": "Fairness balance",
        "outcome_balance": "Outcome balance",
        "outcome_diversity": "Outcome variety",
        "at_risk_concentration": "At-risk concentration",
        "size_balance": "Size balance",
    }.get(str(name), str(name).replace("_", " ").title())


def compact_cluster_label(cluster_id: int, summary: pd.DataFrame | None = None) -> str:
    """Short labels for the Live Predict assignment map."""
    fallback = f"Cluster #{cluster_id}"
    if summary is None or summary.empty or "interpretive_label" not in summary.columns:
        return fallback
    row = summary[summary["cluster"].astype(int).eq(int(cluster_id))]
    if row.empty:
        return fallback
    label = str(row["interpretive_label"].iloc[0]).replace(" learners", "")
    replacements = {
        "high-engagement, high-performance": "High engagement",
        "low-engagement, average-performance": "Steady submitters",
        "low-engagement, low-performance": "At-risk",
    }
    return replacements.get(label, label.title())


def style_winner_row(df: pd.DataFrame, winner_id: str) -> Any:
    """Pandas Styler that paints the winning config row in cyan."""
    def _highlight(row):
        if row.get("config_id") == winner_id or row.get("Model") == winner_id:
            return [
                f"background-color: rgba(34, 211, 238, 0.10); "
                f"border-left: 2px solid {ACCENT}; color: {INK};"
            ] * len(row)
        return [""] * len(row)
    return df.style.apply(_highlight, axis=1)


def style_significant_p(df: pd.DataFrame) -> Any:
    """Highlight p ≤ 0.05 cells with the cyan accent (tinted bg + bold cyan text).

    The dataframe font is small enough that color-only highlighting reads as
    a typo, not a signal. Background tint + bold weight + cyan text together
    give the row enough presence to scan-find at a glance.
    """
    def _color(val):
        try:
            v = float(val)
        except (TypeError, ValueError):
            return ""
        if 0 <= v <= SIG_THRESHOLD:
            return (
                f"color: {ACCENT}; "
                f"font-weight: 700; "
                f"background-color: rgba(34, 211, 238, 0.12);"
            )
        return ""
    p_cols = [c for c in df.columns if "p_value" in c.lower() or "p-value" in c.lower()]
    return df.style.map(_color, subset=p_cols)


# === Discriminative-feature picker (used by Live Predict) ====================
@st.cache_data
def pick_input_fields(_schema_dict: dict, _features_hash: str, max_fields: int = 7) -> list[str]:
    """Pick the most demo-friendly input fields.

    Strategy: include the engagement/performance role fields, then top up with
    high-separation numeric fields so sliders produce visible cluster shifts.
    Fairness fields stay out of the default live demo surface; they are still
    held in the model artifacts and filled from their learned medians.
    """
    art = load_predict_artifacts()
    if art is None:
        return []

    role_fields = [
        f.name for f in art.schema.fields
        if f.is_role_engagement or f.is_role_performance
    ]

    # Score remaining numeric fields by per-cluster mean separation if we can,
    # else by training-data variance (proxy: just keep alphabetical for stability).
    other_numeric = [f.name for f in art.schema.fields
                     if f.kind == "numeric" and f.name not in role_fields]

    # If we have features.parquet, rank by simple between-cluster F-stat proxy
    try:
        features = read_parquet("features.parquet")
        labels = read_parquet("cluster_labels.parquet").set_index("id_student")
        winner_cid = read_json("winner.json")["config_id"]
        merged = features.set_index("id_student").join(labels[winner_cid].rename("__cluster__"))
        scored: list[tuple[str, float]] = []
        for col in other_numeric:
            if col not in merged.columns:
                continue
            s = pd.to_numeric(merged[col], errors="coerce")
            if s.var() <= 0:
                continue
            grp_means = s.groupby(merged["__cluster__"]).mean()
            scored.append((col, float(grp_means.std() / s.std())))
        scored.sort(key=lambda kv: kv[1], reverse=True)
        other_numeric = [c for c, _ in scored]
    except Exception:
        pass

    picked = list(role_fields)
    for c in other_numeric:
        if len(picked) >= max_fields:
            break
        if c not in picked:
            picked.append(c)
    return picked


# === Presets =================================================================
def build_presets() -> dict[str, dict[str, Any]]:
    """Pre-staged learner profiles so the demo always has a fallback path.

    Strategy: anchor each preset on an ACTUAL cluster centroid in raw-feature
    space, so the preset is guaranteed to land in the intended cluster (not
    just hand-waved by setting two role columns to extremes while the other
    33 features sit on the cohort median, which makes both presets nearly
    identical in clustering distance).
    """
    art = load_predict_artifacts()
    if art is None:
        return {}
    try:
        features = read_parquet("features.parquet")
        labels_df = read_parquet("cluster_labels.parquet")
        winner_cid = read_json("winner.json")["config_id"]
        merged = features.set_index("id_student").join(
            labels_df.set_index("id_student")[winner_cid].rename("__cluster__"),
        )
    except Exception:
        return {}

    # Identify the "high-engagement" cluster vs the "at-risk" cluster from
    # the schema's role columns. If engagement/performance roles exist, pick
    # the cluster with HIGHEST mean engagement+performance and the cluster
    # with LOWEST. Otherwise fall back to extreme cluster ids.
    role_cols = []
    eng = next((f.name for f in art.schema.fields if f.is_role_engagement), None)
    perf = next((f.name for f in art.schema.fields if f.is_role_performance), None)
    if eng and eng in merged.columns: role_cols.append(eng)
    if perf and perf in merged.columns: role_cols.append(perf)

    cluster_ranking = None
    if role_cols:
        # Z-score then sum — gives a single "engagement+performance" score per cluster.
        z = (merged[role_cols] - merged[role_cols].mean()) / merged[role_cols].std(ddof=0)
        score = z.groupby(merged["__cluster__"]).mean().mean(axis=1)
        cluster_ranking = score.sort_values(ascending=False)

    if cluster_ranking is None or len(cluster_ranking) < 2:
        return {}

    ranked_clusters = [int(c) for c in cluster_ranking.index.tolist()]
    high_cluster = ranked_clusters[0]
    low_cluster = ranked_clusters[-1]
    middle_cluster = ranked_clusters[len(ranked_clusters) // 2] if len(ranked_clusters) >= 3 else None

    def _centroid_dict(cluster_id: int) -> dict[str, Any]:
        sub = merged[merged["__cluster__"] == cluster_id]
        out: dict[str, Any] = {}
        for spec in art.schema.fields:
            if spec.name not in sub.columns:
                out[spec.name] = spec.median
                continue
            if spec.kind == "numeric":
                val = float(pd.to_numeric(sub[spec.name], errors="coerce").median())
                if spec.p05 is not None and spec.p95 is not None:
                    val = max(float(spec.p05), min(float(spec.p95), val))
                out[spec.name] = val
            else:
                mode = sub[spec.name].mode(dropna=True)
                out[spec.name] = str(mode.iloc[0]) if not mode.empty else spec.median
        return out

    presets = {
        "High-engagement learner": _centroid_dict(high_cluster),
    }
    if middle_cluster is not None and middle_cluster not in {high_cluster, low_cluster}:
        presets["Middle-profile learner"] = _centroid_dict(middle_cluster)
    presets["At-risk learner"] = _centroid_dict(low_cluster)
    return presets


# === Pages ===================================================================
def missing_cache_page() -> None:
    eyebrow("Pipeline cache missing")
    st.markdown("# Run the pipeline first")
    st.markdown(
        "This dashboard needs the saved pipeline results before it can show the demo. Build them with:"
    )
    st.code("python -m src.pipeline --presentation AAA_2014J", language="bash")
    st.caption(
        "Smoke runs use --fast (samples 300 learners, B=2 bootstrap). "
        "A safety guard refuses to overwrite an existing full-fidelity "
        "cache (B≥30) without --force."
    )


# ---- Page 1 — Overview ------------------------------------------------------
def page_overview() -> None:
    meta = read_json("meta.json")
    winner = read_json("winner.json")
    copy = adapter_copy()
    gm = read_parquet("group_metrics.parquet").set_index("strategy")

    eyebrow(copy["subtitle"])
    st.markdown("# Build study groups from learner patterns")
    st.markdown(
        "<p style='font-size: 18px; color: var(--ink-muted); max-width: 780px;'>"
        "The system reads learner activity, finds common learner profiles, "
        "and then forms study groups. One mode puts similar students together; "
        "the other mixes different learner types for peer support. The technical "
        "validation is still here, but the demo starts with what a teacher would see."
        "</p>",
        unsafe_allow_html=True,
    )

    st.write("")
    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(hero_metric("Students", f"{meta.get('n_learners', 0):,}",
                            f"{meta.get('n_features', 0)} learner signals"), unsafe_allow_html=True)
    c2.markdown(hero_metric("Chosen model", str(winner.get("config_id", "—")),
                            f"{winner.get('reducer','—')} + {winner.get('clusterer','—')}",
                            accent=True), unsafe_allow_html=True)
    sil = winner.get("silhouette")
    c3.markdown(hero_metric("Profile separation", f"{sil:.3f}" if sil else "—",
                            "higher means cleaner profile groups"), unsafe_allow_html=True)
    ari = winner.get("bootstrap_ari_mean")
    c4.markdown(hero_metric("Repeatability", f"{ari:.3f}" if ari else "—",
                            "same profiles found again and again"), unsafe_allow_html=True)

    st.markdown("## Demo path")
    st.markdown(
        "<p><strong>Try learner</strong> lets you move a few sliders and place a "
        "new student into a learner profile. <strong>Form groups</strong> shows "
        "the actual study groups. <strong>Check results</strong> shows whether "
        "the system beats random grouping.</p>",
        unsafe_allow_html=True,
    )

    # Live results from this run, not from the talk-track script.
    st.markdown("## Main takeaways")
    if not gm.empty:
        col_a, col_b, col_c = st.columns(3)
        if "intra_group_distance" in gm.columns:
            ma = gm.loc["mode_a", "intra_group_distance"]
            rnd = gm.loc["random", "intra_group_distance"]
            col_a.markdown(hero_metric(
                "Similar groups are tighter",
                f"{ma:.3f}",
                f"random groups were {rnd:.3f}",
            ), unsafe_allow_html=True)
        if "complementarity" in gm.columns:
            mb = gm.loc["mode_b", "complementarity"]
            rnd = gm.loc["random", "complementarity"]
            col_b.markdown(hero_metric(
                "Mixed groups include more profiles",
                f"{mb:.3f}",
                f"random groups were {rnd:.3f}",
            ), unsafe_allow_html=True)
        if "cluster_coverage" in gm.columns:
            mb = gm.loc["mode_b", "cluster_coverage"]
            ma = gm.loc["mode_a", "cluster_coverage"]
            col_c.markdown(hero_metric(
                "Most mixed groups have variety",
                f"{mb:.3f}",
                f"similar-group mode was {ma:.3f}",
            ), unsafe_allow_html=True)


# ---- Page 2 — Live predict (the model moment) -------------------------------
def page_live_predict() -> None:
    if not predict_ready():
        eyebrow("Live predict unavailable")
        st.markdown("# Live predict requires re-running the pipeline")
        st.markdown(
            "The cache at `{}` was built before the live-predict artifacts were "
            "added. Regenerate to enable this page:".format(CACHE)
        )
        st.code("python -m src.pipeline --presentation AAA_2014J", language="bash")
        return

    art = load_predict_artifacts()
    eyebrow("Try a learner profile")
    st.markdown("# Place a new student")
    st.markdown(
        "<p style='font-size: 17px; color: var(--ink-muted); max-width: 780px;'>"
        "Choose a sample student type, then move the sliders. The app shows "
        "which learner profile is the closest match and where that student "
        "would land on the profile map."
        "</p>",
        unsafe_allow_html=True,
    )

    presets = build_presets()
    fields = pick_input_fields(art.schema.to_dict(), "v1", max_fields=7)
    field_specs = {f.name: f for f in art.schema.fields}

    # --- Input column + result column ---
    left, right = st.columns([1, 1.25])

    with left:
        st.markdown("### Student inputs")
        preset_options = [*presets.keys(), "Custom"] if presets else ["Custom"]
        preset_choice = st.radio(
            "Start with",
            preset_options,
            horizontal=True,
            label_visibility="collapsed",
        )
        st.caption(
            "Presets fill the full student profile. The sliders are the fields "
            "you can adjust live. Some small changes may keep the same profile; "
            "larger changes can move the student."
        )

        # Seed session state for each input the FIRST time it's seen, plus
        # whenever the user changes preset. Streamlit forbids passing both
        # value= and a session-state-set key= to the same widget, so we
        # write into session state, then let widgets read via key= only.
        def _slider_bounds(spec: Any) -> tuple[float, float, float]:
            lo, hi = float(spec.p05), float(spec.p95)
            step = max((hi - lo) / 100.0, 0.01)
            return lo, hi, step

        def _snap_to_slider(value: Any, spec: Any) -> float:
            lo, hi, step = _slider_bounds(spec)
            clipped = max(lo, min(hi, float(value)))
            snapped = lo + round((clipped - lo) / step) * step
            return max(lo, min(hi, snapped))

        last_preset_key = "_last_preset"
        preset_changed = st.session_state.get(last_preset_key) != preset_choice
        for fname in fields:
            spec = field_specs[fname]
            session_key = f"input_{fname}"
            if preset_changed or session_key not in st.session_state:
                if preset_choice in presets:
                    desired = presets[preset_choice].get(fname, spec.median)
                else:
                    desired = st.session_state.get(session_key, spec.median)
                # Clip numerics to the slider's allowed range so Streamlit
                # doesn't reject the value as out-of-bounds.
                if spec.kind == "numeric" and spec.p05 is not None and spec.p95 is not None:
                    desired = _snap_to_slider(desired, spec)
                st.session_state[session_key] = desired
        st.session_state[last_preset_key] = preset_choice

        raw_input: dict[str, Any] = dict(presets.get(preset_choice, {}))
        for fname in fields:
            spec = field_specs[fname]
            role = []
            if spec.is_role_engagement: role.append("engagement")
            if spec.is_role_performance: role.append("performance")
            if spec.is_role_fairness: role.append("fairness")
            role_str = " · ".join(role) if role else "feature"
            label = f"{friendly_feature_label(fname)}  ·  {friendly_role_label(role_str)}"

            session_key = f"input_{fname}"
            if spec.kind == "numeric" and spec.p05 is not None and spec.p95 is not None:
                lo, hi, step = _slider_bounds(spec)
                if lo == hi:
                    raw_input[fname] = lo
                    continue
                raw_input[fname] = st.slider(
                    label,
                    min_value=lo, max_value=hi, step=step,
                    key=session_key,
                    help=f"Source field: {fname}",
                )
            elif spec.kind == "categorical" and spec.categories:
                raw_input[fname] = st.selectbox(
                    label,
                    spec.categories,
                    key=session_key,
                    help=f"Source field: {fname}",
                )

    # --- Predict ---
    try:
        result = predict_mod.predict_one(raw_input, art)
    except Exception as exc:
        st.error(f"predict_one failed: {exc}")
        return

    cluster_summary = read_parquet("cluster_summary.parquet")

    with right:
        st.markdown("### Best match")

        conf = result["confidence"]
        conf_badge = badge(confidence_label(conf), accent=(conf == "high"))
        st.markdown(
            f"<div style='margin-bottom: 16px;'>{conf_badge}</div>",
            unsafe_allow_html=True,
        )

        profile_name = compact_cluster_label(int(result["cluster"]), cluster_summary)
        st.markdown(hero_metric(
            "Matched profile",
            profile_name,
            f"cluster #{result['cluster']} · distance {result['distance']:.3f}",
            accent=True,
        ), unsafe_allow_html=True)

        # Profile match meter. This keeps labels inside the plotting area so
        # they never get clipped in the narrow right column.
        dists = result["all_distances"]
        ranked = sorted(dists.items(), key=lambda item: item[1])
        y_positions = list(reversed(range(len(ranked))))
        labels = [compact_cluster_label(int(cluster), cluster_summary) for cluster, _ in ranked]
        values = [float(value) for _, value in ranked]
        min_dist = min(values)
        max_dist = max(values)

        def _match_strength(distance: float) -> float:
            if np.isclose(max_dist, min_dist):
                return 100.0
            # Invert distance into a visual "match strength". Keep a small
            # minimum bar so weaker profiles remain visible without implying
            # they are close matches.
            return 12.0 + 88.0 * ((max_dist - distance) / (max_dist - min_dist))

        fig = go.Figure()
        for y, (cluster, value), label in zip(y_positions, ranked, labels):
            is_winner = int(cluster) == int(result["cluster"])
            color = ACCENT if is_winner else INK_MUTED
            strength = _match_strength(float(value))

            # Row label stays inside the figure; no y-axis tick labels, no clipping.
            fig.add_annotation(
                x=2, y=y,
                text=label,
                showarrow=False,
                xanchor="left",
                yanchor="middle",
                font=dict(
                    family="Inter",
                    size=14,
                    color=INK if is_winner else INK_MUTED,
                ),
            )
            fig.add_annotation(
                x=94, y=y,
                text=("best match" if is_winner else f"dist {float(value):.2f}"),
                showarrow=False,
                xanchor="right",
                yanchor="middle",
                font=dict(
                    family="JetBrains Mono",
                    size=12,
                    color=ACCENT if is_winner else INK_MUTED,
                ),
            )

            # Background rail and foreground match strength.
            fig.add_shape(
                type="line",
                x0=36,
                x1=78,
                y0=y,
                y1=y,
                line=dict(color=RULE, width=8),
                layer="below",
            )
            fig.add_shape(
                type="line",
                x0=36,
                x1=36 + 0.42 * strength,
                y0=y,
                y1=y,
                line=dict(
                    color=color,
                    width=8 if is_winner else 6,
                ),
            )
            fig.add_trace(go.Scatter(
                x=[36 + 0.42 * strength],
                y=[y],
                mode="markers",
                marker=dict(
                    size=16 if is_winner else 10,
                    color=color,
                    line=dict(color=BG, width=2),
                ),
                hovertemplate=f"{label}<br>cluster #{cluster}<br>distance = {float(value):.3f}<extra></extra>",
                showlegend=False,
                cliponaxis=False,
            ))
        fig.update_layout(
            height=230,
            margin=dict(l=8, r=8, t=44, b=18),
            showlegend=False,
            title=dict(
                text="profile match strength",
                font=dict(size=12, color=INK_FAINT, family="Inter"),
                x=0, xanchor="left", y=0.97,
            ),
            xaxis=dict(
                visible=False,
                range=[0, 100],
                fixedrange=True,
            ),
            yaxis=dict(
                visible=False,
                range=[-0.65, len(ranked) - 0.35],
                fixedrange=True,
            ),
            plot_bgcolor="#0B0F14",
        )
        st.plotly_chart(fig, use_container_width=True, theme=None)

    # --- Simplified assignment map -----------------------------------------
    st.markdown("### Student profile map")
    st.caption(
        "Each large circle is one learner profile. The cyan dot is the new "
        "student; the dashed line shows the closest match."
    )

    X_vis = read_npy("reduced_umap_2d.npy")
    labels = read_parquet("cluster_labels.parquet")
    winner_cid = read_json("winner.json")["config_id"]
    cluster_col = labels[winner_cid].astype(int).to_numpy()
    pred_cluster = int(result["cluster"])

    unique_clusters = sorted(set(cluster_col.tolist()) - {-1})
    zone_palette = ["#A78BFA", "#60A5FA", "#86EFAC", "#FBBF24", "#F472B6"]
    cluster_color = {c: zone_palette[i % len(zone_palette)] for i, c in enumerate(unique_clusters)}
    centroids: dict[int, tuple[float, float]] = {}

    fig = go.Figure()
    # Faint point cloud first, so the map is still honest to the real embedding.
    for c in unique_clusters:
        mask = cluster_col == c
        cx = float(X_vis[mask, 0].mean())
        cy = float(X_vis[mask, 1].mean())
        centroids[c] = (cx, cy)
        fig.add_trace(go.Scatter(
            x=X_vis[mask, 0], y=X_vis[mask, 1],
            mode="markers",
            marker=dict(
                size=5,
                color=cluster_color[c],
                opacity=0.28 if c != pred_cluster else 0.46,
                line=dict(width=0),
            ),
            name=f"{compact_cluster_label(c, cluster_summary)}" + ("  ← predicted" if c == pred_cluster else ""),
            hovertemplate=f"cluster #{c}<extra></extra>",
        ))

    pred_x, pred_y = result["umap_2d"]
    assigned_cx, assigned_cy = centroids[pred_cluster]

    # Dashed path from the new learner to the assigned centroid.
    fig.add_trace(go.Scatter(
        x=[pred_x, assigned_cx],
        y=[pred_y, assigned_cy],
        mode="lines",
        line=dict(color=ACCENT, width=3, dash="dash"),
        name="nearest centroid path",
        hoverinfo="skip",
        showlegend=False,
    ))

    # Large centroid zones and readable labels.
    for c in unique_clusters:
        cx, cy = centroids[c]
        assigned = c == pred_cluster
        label = compact_cluster_label(c, cluster_summary)
        fig.add_trace(go.Scatter(
            x=[cx],
            y=[cy],
            mode="markers+text",
            marker=dict(
                size=54 if assigned else 42,
                color=cluster_color[c],
                opacity=0.95 if assigned else 0.82,
                line=dict(color=ACCENT if assigned else SURFACE, width=4 if assigned else 2),
            ),
            text=[f"{label}<br><span style='font-size:12px'>cluster #{c}</span>"],
            textposition="bottom center",
            textfont=dict(family="Inter", size=14, color=INK),
            name=f"cluster #{c} centroid" + ("  ← assigned" if assigned else ""),
            hovertemplate=f"{label}<br>cluster #{c}<extra></extra>",
            showlegend=False,
        ))

    # The new learner — cyan dot with label.
    fig.add_trace(go.Scatter(
        x=[pred_x], y=[pred_y], mode="markers+text",
        marker=dict(size=24, color=ACCENT,
                    line=dict(color=BG, width=3),
                    symbol="circle"),
        text=["new learner"], textposition="top center",
        textfont=dict(family="Inter", size=14, color=ACCENT),
        name="hypothetical learner",
        hovertemplate=f"<b>new learner</b><br>cluster #{pred_cluster}<br>"
                      f"distance: {result['distance']:.3f}<extra></extra>",
    ))

    x_min, x_max = float(np.nanmin(X_vis[:, 0])), float(np.nanmax(X_vis[:, 0]))
    y_min, y_max = float(np.nanmin(X_vis[:, 1])), float(np.nanmax(X_vis[:, 1]))
    x_pad = (x_max - x_min) * 0.10
    y_pad = (y_max - y_min) * 0.14
    fig.update_layout(
        height=560,
        xaxis=dict(
            title="Learner profile direction 1 →",
            showticklabels=False,
            showgrid=True,
            gridcolor=RULE,
            zeroline=False,
            range=[x_min - x_pad, x_max + x_pad],
            fixedrange=True,
        ),
        yaxis=dict(
            title="Learner profile direction 2 →",
            showticklabels=False,
            showgrid=True,
            gridcolor=RULE,
            zeroline=False,
            range=[y_min - y_pad, y_max + y_pad],
            fixedrange=True,
        ),
        legend=dict(orientation="h", yanchor="bottom", y=1.03, xanchor="left", x=0),
        margin=dict(l=48, r=24, t=24, b=56),
        plot_bgcolor="#0B0F14",
    )
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("What is happening behind the scenes"):
        st.markdown(
            "- The app fills any hidden fields from the chosen preset, then "
            "uses the same saved model artifacts from the pipeline run.\n"
            "- It scales the student profile, places it on the learned map, "
            "and picks the nearest profile center.\n"
            "- A clear match means the student is close to that profile. "
            "Borderline or weak matches mean the student sits between profiles."
        )


# ---- Page 3 — Clustering ----------------------------------------------------
def page_clustering() -> None:
    metrics = read_parquet("config_metrics.parquet")
    stability = read_parquet("stability.parquet")
    labels = read_parquet("cluster_labels.parquet")
    winner = read_json("winner.json")
    joined = metrics.merge(
        stability[["config_id", "bootstrap_ari_mean"]],
        on="config_id", how="left",
    )
    winner_id = winner["config_id"]

    eyebrow("How the system chose a model")
    st.markdown("# Choose the profile finder")
    st.markdown(
        "<p style='font-size: 17px; color: var(--ink-muted); max-width: 780px;'>"
        "The system tested twelve ways to find learner profiles. The chosen "
        "one had to make reasonably clear groups and also stay repeatable when "
        "the data was resampled. That keeps us from choosing a model just "
        "because one score looked good once."
        "</p>",
        unsafe_allow_html=True,
    )

    win_row = joined[joined["config_id"] == winner_id]
    if not win_row.empty:
        w = win_row.iloc[0]
        c1, c2, c3, c4 = st.columns(4)
        c1.markdown(hero_metric("Separation", f"{w.get('silhouette', 0):.3f}",
                                "cleaner profile groups", accent=True), unsafe_allow_html=True)
        c2.markdown(hero_metric("Overlap check", f"{w.get('davies_bouldin', 0):.3f}",
                                "lower means less overlap"), unsafe_allow_html=True)
        c3.markdown(hero_metric("Group clarity", f"{w.get('calinski_harabasz', 0):.0f}",
                                "higher means clearer groups"), unsafe_allow_html=True)
        c4.markdown(hero_metric("Repeatability", f"{w.get('bootstrap_ari_mean', 0):.3f}",
                                f"{int(w.get('k', 0))} profiles found", accent=True), unsafe_allow_html=True)

    st.markdown("## Models tested")
    st.caption(
        f"The cyan row ({winner_id}) is the model used by the demo. The dotted "
        f"line below is the minimum repeatability score we required."
    )
    models_plain = pd.DataFrame({
        "Model": joined["config_id"],
        "Method": joined["reducer"].str.upper() + " + " + joined["clusterer"].str.title(),
        "Profiles": joined["k"].astype(int),
        "Separation": joined["silhouette"],
        "Overlap": joined["davies_bouldin"],
        "Repeatability": joined["bootstrap_ari_mean"],
        "Unassigned learners": joined.get("noise_ratio", 0.0),
    })
    # height tall enough that all 12 configurations fit without virtual-scroll —
    # otherwise C10/C11/C12 only render after the user manually scrolls inside
    # the dataframe, and they're not searchable in screen-reader / Cmd-F either.
    st.dataframe(
        style_winner_row(models_plain, winner_id).format({
            "Separation": "{:.3f}",
            "Overlap": "{:.3f}",
            "Repeatability": "{:.3f}",
            "Unassigned learners": "{:.3f}",
        }),
        use_container_width=True, hide_index=True,
        height=36 * (len(joined) + 1) + 4,
    )

    st.markdown("## Clear vs repeatable")
    primary = joined[joined["clusterer"].ne("hdbscan")].copy()
    primary["is_winner"] = primary["config_id"].eq(winner_id)
    fig = go.Figure()
    # Non-winner configs — INK_MUTED markers with INK labels (readable).
    fig.add_trace(go.Scatter(
        x=primary.loc[~primary["is_winner"], "silhouette"],
        y=primary.loc[~primary["is_winner"], "bootstrap_ari_mean"],
        mode="markers+text",
        marker=dict(size=14, color=INK_MUTED, line=dict(color=BG, width=1)),
        text=primary.loc[~primary["is_winner"], "config_id"],
        textposition="top center",
        textfont=dict(family="Inter", size=12, color=INK),
        hovertemplate="%{text}: sil=%{x:.3f}, ARI=%{y:.3f}<extra></extra>",
        showlegend=False,
    ))
    # Winner — bigger cyan marker, cyan label, no overlap with hline label.
    fig.add_trace(go.Scatter(
        x=primary.loc[primary["is_winner"], "silhouette"],
        y=primary.loc[primary["is_winner"], "bootstrap_ari_mean"],
        mode="markers+text",
        marker=dict(size=24, color=ACCENT, line=dict(color=BG, width=2)),
        text=primary.loc[primary["is_winner"], "config_id"],
        textposition="top center",
        textfont=dict(family="Inter", size=14, color=ACCENT, weight=600),
        hovertemplate="<b>%{text}</b> (winner): sil=%{x:.3f}, ARI=%{y:.3f}<extra></extra>",
        showlegend=False,
    ))
    # Stability gate — readable annotation pinned to the bottom-left so it
    # doesn't collide with config dots crowding the right side.
    fig.add_hline(
        y=0.40,
        line_dash="dot", line_color=INK_MUTED, line_width=1.2,
        annotation_text="minimum repeatability",
        annotation_position="bottom left",
        annotation_font=dict(family="Inter", size=12, color=INK_MUTED),
        annotation_bgcolor=BG,
    )
    fig.update_layout(
        height=480,
        xaxis_title="Profile separation",
        yaxis_title="Repeatability",
        margin=dict(l=64, r=24, t=24, b=56),
    )
    st.plotly_chart(fig, use_container_width=True)

    cluster_summary_path = CACHE / "cluster_summary.parquet"
    if cluster_summary_path.exists():
        st.markdown("## What each profile means")
        st.caption("Traits that make each learner profile different from the class average.")
        raw_summary = pd.read_parquet(cluster_summary_path)
        plain_summary = pd.DataFrame({
            "Profile": [
                compact_cluster_label(int(row["cluster"]), raw_summary)
                for _, row in raw_summary.iterrows()
            ],
            "Students": raw_summary["size"].astype(int),
            "Higher than class average": raw_summary["top_positive_features"].map(friendly_feature_list),
            "Lower than class average": raw_summary["top_negative_features"].map(friendly_feature_list),
            "Technical cluster": raw_summary["cluster"].map(lambda c: f"#{int(c)}"),
        })
        st.dataframe(plain_summary, use_container_width=True, hide_index=True)


# ---- Page 4 — Groups --------------------------------------------------------
def page_groups() -> None:
    meta = read_json("meta.json")
    winner = read_json("winner.json")
    eyebrow("Turn profiles into study groups")
    st.markdown("# Form study groups")
    st.markdown(
        "<p style='font-size: 17px; color: var(--ink-muted); max-width: 780px;'>"
        "Choose how you want the class grouped. Similar groups put students "
        "with close profiles together. Mixed groups combine different profiles "
        "so stronger and weaker learners can support each other."
        "</p>",
        unsafe_allow_html=True,
    )

    mode = st.radio(
        "Grouping style", ["Similar students together", "Mixed profiles together"],
        horizontal=True,
    )
    mode_key = "mode_a" if mode.startswith("Similar") else "mode_b"
    default_group_size = int(meta.get("group_size", 4) or 4)

    groups = read_parquet(
        "groups_mode_a.parquet" if mode_key == "mode_a" else "groups_mode_b.parquet"
    )

    c1, c2, c3 = st.columns(3)
    c1.markdown(hero_metric("Groups", f"{groups['group_id'].nunique():,}"),
                unsafe_allow_html=True)
    c2.markdown(hero_metric("Students", f"{len(groups):,}"),
                unsafe_allow_html=True)
    c3.markdown(hero_metric("Target size", str(default_group_size)),
                unsafe_allow_html=True)

    st.markdown("## Group list")
    query = st.text_input("Search student or group", placeholder="Type a student id or group number")
    filtered = groups
    if query:
        text = groups.astype(str).agg(" ".join, axis=1)
        filtered = groups[text.str.contains(query, case=False, regex=False)]
    display_filtered = filtered.rename(columns={
        "id_student": "Student ID",
        "group_id": "Group",
        "cluster": "Profile",
    })
    display_filtered = display_filtered.rename(columns={
        col: friendly_feature_label(col) for col in display_filtered.columns
    })
    st.dataframe(display_filtered, height=420, use_container_width=True, hide_index=True)
    st.download_button(
        "Download groups",
        groups.to_csv(index=False),
        file_name=f"groups_{mode_key}.csv",
        mime="text/csv",
    )

    st.markdown("## Look inside one group")
    group_id = st.selectbox("Group", sorted(groups["group_id"].unique()))
    members = groups[groups["group_id"].eq(group_id)]
    schema = json.loads((CACHE / "schema.json").read_text(encoding="utf-8")) \
        if (CACHE / "schema.json").exists() else {}
    display_cols = [
        col for col in schema.get("display_cols", [])
        if col in members.columns and pd.api.types.is_numeric_dtype(members[col])
    ][:4]
    if not display_cols:
        display_cols = ["cluster"]

    left, right = st.columns([1, 1])
    with left:
        display_members = members.rename(columns={
            "id_student": "Student ID",
            "group_id": "Group",
            "cluster": "Profile",
        })
        display_members = display_members.rename(columns={
            col: friendly_feature_label(col) for col in display_members.columns
        })
        st.dataframe(display_members, use_container_width=True, hide_index=True)
    with right:
        # One bar per learner per feature. Force learner ids to string so Plotly
        # uses a discrete (NOT viridis-continuous) palette, then thread the
        # Linear-discipline ink ramp through the discrete sequence.
        plot_df = members[["id_student", *display_cols]].copy()
        plot_df["id_student"] = plot_df["id_student"].astype(str)
        long = plot_df.melt(id_vars=["id_student"], value_vars=display_cols,
                            var_name="feature", value_name="value")
        long["feature_label"] = long["feature"].map(friendly_feature_label)
        ink_ramp = [INK, INK_MUTED, INK_FAINT, "#52525B"]
        fig = px.bar(
            long, x="feature_label", y="value", color="id_student",
            barmode="group", height=380,
            color_discrete_sequence=ink_ramp,
            hover_data={"feature": True, "feature_label": False},
        )
        fig.update_layout(
            xaxis_title="", yaxis_title="",
            legend=dict(
                title_text="learner",
                orientation="v",
                y=1.0, x=1.02, xanchor="left", yanchor="top",
                font=dict(size=11, color=INK_MUTED),
                bgcolor="rgba(0,0,0,0)",
            ),
            margin=dict(l=24, r=140, t=24, b=24),
        )
        fig.update_xaxes(tickangle=-30)
        st.plotly_chart(fig, use_container_width=True)


# ---- Page 5 — Evaluation ----------------------------------------------------
def page_evaluation() -> None:
    eyebrow("Compared with random grouping")
    st.markdown("# Check the result")
    verdict = adapter_copy()["verdict"]
    st.markdown(f"<blockquote>{verdict}</blockquote>", unsafe_allow_html=True)

    group_metrics = read_parquet("group_metrics.parquet").set_index("strategy")
    sig_path = CACHE / "group_significance.parquet"
    sig = pd.read_parquet(sig_path) if sig_path.exists() else pd.DataFrame()

    st.markdown("## Main checks")
    st.caption(
        "We compare the system's groups with 100 random groupings. A small "
        "p-value means the result is unlikely to be random."
    )

    def _hl(strategy: str, metric: str, direction: str) -> tuple[float, float, float | None]:
        val = float(group_metrics.loc[strategy, metric]) if metric in group_metrics.columns else float("nan")
        rnd = float(group_metrics.loc["random", metric]) if metric in group_metrics.columns else float("nan")
        p = None
        if not sig.empty:
            row = sig[sig["strategy"].eq(strategy) & sig["metric"].eq(metric)]
            if not row.empty:
                p = float(row.iloc[0]["preferred_p_value"])
        return val, rnd, p

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("**Similar groups**")
        val, rnd, p = _hl("mode_a", "intra_group_distance", "lower")
        accent = (p is not None and p <= SIG_THRESHOLD)
        st.markdown(hero_metric(
            "Tightness ↓",
            f"{val:.3f}",
            f"random {rnd:.3f} · p = {('—' if p is None else f'{p:.3f}')}",
            accent=accent,
        ), unsafe_allow_html=True)

    with col_b:
        st.markdown("**Mixed groups**")
        val, rnd, p = _hl("mode_b", "complementarity", "higher")
        accent = (p is not None and p <= SIG_THRESHOLD)
        st.markdown(hero_metric(
            "Profile mix ↑",
            f"{val:.3f}",
            f"random {rnd:.3f} · p = {('—' if p is None else f'{p:.3f}')}",
            accent=accent,
        ), unsafe_allow_html=True)

    st.markdown("## Full metric table")
    metrics_plain = group_metrics.reset_index().rename(columns={"strategy": "Grouping style"})
    metrics_plain["Grouping style"] = metrics_plain["Grouping style"].map(friendly_strategy_label)
    metrics_plain = metrics_plain.rename(columns={
        col: friendly_metric_label(col) for col in metrics_plain.columns
    })
    st.dataframe(metrics_plain, use_container_width=True, hide_index=True)

    if not sig.empty:
        st.markdown("## Random comparison details")
        st.caption(
            f"One-sided p-values across {sig['n_random_runs'].iloc[0] if 'n_random_runs' in sig.columns else 100} "
            "random partitions. Cyan cells: p ≤ 0.05."
        )
        sig_show = sig[["strategy", "metric", "value", "random_mean", "preferred_direction", "preferred_p_value"]].copy()
        sig_show["strategy"] = sig_show["strategy"].map(friendly_strategy_label)
        sig_show["metric"] = sig_show["metric"].map(friendly_metric_label)
        sig_show["preferred_direction"] = sig_show["preferred_direction"].map({
            "lower": "lower is better",
            "higher": "higher is better",
        }).fillna(sig_show["preferred_direction"])
        sig_show = sig_show.rename(columns={
            "strategy": "Grouping style",
            "metric": "What we checked",
            "value": "System value",
            "random_mean": "Random average",
            "preferred_direction": "Goal",
            "preferred_p_value": "p-value",
        })
        st.dataframe(
            style_significant_p(sig_show),
            use_container_width=True, hide_index=True,
        )

    constraints_path = CACHE / "constraints.json"
    if constraints_path.exists():
        with st.expander("Constraint details"):
            st.json(read_json("constraints.json"))


# === Sidebar + router ========================================================
PAGES: dict[str, Any] = {
    "Start":        page_overview,
    "Try learner":  page_live_predict,
    "Choose model": page_clustering,
    "Form groups":  page_groups,
    "Check result": page_evaluation,
}


def main() -> None:
    if not cache_ready():
        missing_cache_page()
        return

    with st.sidebar:
        st.markdown("### CollabLearn")
        st.caption("INT-396  ·  Lovely Professional University")
        st.divider()
        page = st.radio("Navigate", list(PAGES.keys()), label_visibility="collapsed")
        st.divider()
        meta = read_json("meta.json")
        winner = read_json("winner.json")
        st.caption(f"Dataset · {meta.get('dataset_name') or meta.get('presentation', '—')}")
        st.caption(f"N = {meta.get('n_learners', 0):,} learners")
        st.caption(f"Winner · {winner.get('config_id', '—')}")
        if not predict_ready():
            st.caption("⚠ predict artifacts missing — re-run pipeline")

    PAGES[page]()


if __name__ == "__main__":
    main()
