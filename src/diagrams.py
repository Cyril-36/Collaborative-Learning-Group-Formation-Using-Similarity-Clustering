"""Graphviz flowchart figures for the patent specification and methodology docs.

All diagrams use GRAPHVIZ_DEFAULTS from plot_style so they match the matplotlib
system (Inter Tight nodes, JetBrains Mono edge labels, SURFACE fills, INK_MUTED
lines). Render at 200 dpi to match savefig.dpi.

Figures produced:
    fig1 — Pipeline block diagram (end-to-end data flow)
    fig4 — Bootstrap stability validation flow
    fig5 — Dual-mode group allocation flow (Mode A vs Mode B)
    fig6 — Overall method flowchart (high-level)
"""

from __future__ import annotations

from pathlib import Path

import graphviz

from .plot_style import ACCENT, GRAPHVIZ_DEFAULTS, INK, INK_MUTED, SURFACE


def _base_graph(name: str, **overrides) -> graphviz.Digraph:
    """Create a Digraph pre-loaded with project defaults."""
    defaults = GRAPHVIZ_DEFAULTS.copy()
    g = graphviz.Digraph(
        name,
        graph_attr={**defaults["graph_attr"], **overrides},
        node_attr=defaults["node_attr"],
        edge_attr=defaults["edge_attr"],
    )
    return g


def fig1_pipeline(output_path: Path) -> None:
    """End-to-end pipeline block diagram."""
    g = _base_graph("fig1_pipeline", rankdir="LR")

    # Data layer
    with g.subgraph(name="cluster_data") as s:
        s.attr(rank="same")
        s.node("oulad", "Learner dataset")
        s.node("ingest", "Dataset\nadapter")

    # Feature engineering
    g.node("features", "Feature\nengineering\n(35 features)")
    g.node("preprocess", "Preprocess\nscale + impute")

    # Reduction + clustering
    g.node("reducers", "Reducers\nPCA · UMAP · t-SNE")
    g.node("clusterers", "Clusterers\nK-Means · GMM\nAgglo · HDBSCAN")
    g.node("configs", "12-config\nmatrix", style="filled", fillcolor=SURFACE,
           color=ACCENT, penwidth="1.2")

    # Validation
    g.node("stability", "Bootstrap\nstability\n(B=30)")
    g.node("selector", "Selector\n(composite rank)", style="filled",
           fillcolor=SURFACE, color=ACCENT, penwidth="1.2")

    # Groups
    g.node("groups", "Group\nformation\n(Mode A + B)")
    g.node("eval", "Evaluation\n+ significance")

    # Edges
    g.edge("oulad", "ingest")
    g.edge("ingest", "features")
    g.edge("features", "preprocess")
    g.edge("preprocess", "reducers")
    g.edge("preprocess", "clusterers")
    g.edge("reducers", "configs")
    g.edge("clusterers", "configs")
    g.edge("configs", "stability")
    g.edge("stability", "selector")
    g.edge("selector", "groups")
    g.edge("groups", "eval")

    g.render(
        filename=str(output_path.with_suffix("")),
        format="png",
        cleanup=True,
    )


def fig4_bootstrap(output_path: Path) -> None:
    """Bootstrap stability validation flow."""
    g = _base_graph("fig4_bootstrap", rankdir="TB")

    g.node("data", "Scaled feature\nmatrix (N learners)")
    g.node("resample", "Stratified\nresample\n(frac=0.80)")
    g.node("fit", "Fit configuration\n(reducer + clusterer)")
    g.node("labels", "Cluster\nassignments")
    g.node("ari", "Pairwise\nAdjusted Rand\nIndex")
    g.node("dist", "ARI\ndistribution\n(B×(B-1)/2 pairs)",
           style="filled", fillcolor=SURFACE, color=ACCENT, penwidth="1.2")
    g.node("decision", "Mean ARI\n≥ 0.40?", shape="diamond",
           style="filled", fillcolor=SURFACE)
    g.node("stable", "Config is\nstable", color=ACCENT, fontcolor=ACCENT)
    g.node("unstable", "Config\nrejected", color=INK_MUTED, fontcolor=INK_MUTED)

    g.edge("data", "resample", label="B=30\nresamples")
    g.edge("resample", "fit")
    g.edge("fit", "labels")
    g.edge("labels", "ari", label="all pairs")
    g.edge("ari", "dist")
    g.edge("dist", "decision")
    g.edge("decision", "stable", label="yes")
    g.edge("decision", "unstable", label="no")

    # Loop-back arrow
    g.edge("labels", "resample", style="dashed", constraint="false",
           label="repeat B times", fontcolor=INK_MUTED)

    g.render(
        filename=str(output_path.with_suffix("")),
        format="png",
        cleanup=True,
    )


def fig5_dual_mode(output_path: Path) -> None:
    """Dual-mode group allocation flow — Mode A (homogeneous) vs Mode B (heterogeneous)."""
    g = _base_graph("fig5_dual_mode", rankdir="TB")

    g.node("labels", "Canonical\ncluster labels")
    g.node("reduced", "Reduced\nfeature space")

    # Mode A subgraph
    with g.subgraph(name="cluster_mode_a") as a:
        a.attr(label="Mode A — Homogeneous", labeljust="l",
               style="dashed", color=INK_MUTED)
        a.node("a_intra", "Intra-cluster\nNN grouping")
        a.node("a_leftovers", "Leftover\nbalancing")
        a.node("a_refine", "Constraint\nrefinement")
        a.node("a_out", "Pace-matched\ngroups", style="filled",
               fillcolor=SURFACE, color=INK, penwidth="1.2")
        a.edge("a_intra", "a_leftovers")
        a.edge("a_leftovers", "a_refine")
        a.edge("a_refine", "a_out")

    # Mode B subgraph
    with g.subgraph(name="cluster_mode_b") as b:
        b.attr(label="Mode B — Heterogeneous", labeljust="l",
               style="dashed", color=ACCENT)
        b.node("b_round", "Round-robin\nacross clusters")
        b.node("b_balance", "Size\nbalancing")
        b.node("b_refine", "Constraint\nrefinement")
        b.node("b_out", "Scaffolded\ngroups", style="filled",
               fillcolor=SURFACE, color=ACCENT, penwidth="1.2")
        b.edge("b_round", "b_balance")
        b.edge("b_balance", "b_refine")
        b.edge("b_refine", "b_out")

    g.edge("labels", "a_intra")
    g.edge("reduced", "a_intra")
    g.edge("labels", "b_round")

    g.render(
        filename=str(output_path.with_suffix("")),
        format="png",
        cleanup=True,
    )


def fig6_method_overview(output_path: Path) -> None:
    """Overall method flowchart — high-level view for patent spec."""
    g = _base_graph("fig6_method", rankdir="TB")

    g.node("input", "Learner\ndataset", shape="cylinder",
           style="filled", fillcolor=SURFACE)
    g.node("fe", "Feature engineering\n(demographic + engagement\n+ performance + collaboration)")
    g.node("scale", "Standardisation\n+ imputation")
    g.node("sweep", "12-configuration\nsweep\n(3 reducers × 4 clusterers)")
    g.node("validate", "Bootstrap-ARI\nstability validation\n(B=30, frac=0.80)")
    g.node("select", "Winner selection\n(composite: silhouette\n+ ARI + CH + DBI)",
           style="filled", fillcolor=SURFACE, color=ACCENT, penwidth="1.2")
    g.node("remap", "Canonical\ncluster-ID\nremap")
    g.node("form", "Dual-mode\ngroup formation\n(homo + hetero)")
    g.node("eval", "Evaluation\nvs random baseline\n(p-values)", shape="box",
           style="filled", fillcolor=SURFACE, color=ACCENT, penwidth="1.2")
    g.node("output", "Learner groups\n+ interpretive profiles", shape="box",
           style="filled,bold", fillcolor=SURFACE)

    for a, b in [
        ("input", "fe"), ("fe", "scale"), ("scale", "sweep"),
        ("sweep", "validate"), ("validate", "select"), ("select", "remap"),
        ("remap", "form"), ("form", "eval"), ("eval", "output"),
    ]:
        g.edge(a, b)

    g.render(
        filename=str(output_path.with_suffix("")),
        format="png",
        cleanup=True,
    )


def render_all(output_dir: Path) -> None:
    """Render all 4 graphviz figures to *output_dir*."""
    output_dir.mkdir(parents=True, exist_ok=True)
    fig1_pipeline(output_dir / "fig1_pipeline.png")
    fig4_bootstrap(output_dir / "fig4_bootstrap.png")
    fig5_dual_mode(output_dir / "fig5_dual_mode.png")
    fig6_method_overview(output_dir / "fig6_method.png")


__all__ = [
    "fig1_pipeline",
    "fig4_bootstrap",
    "fig5_dual_mode",
    "fig6_method_overview",
    "render_all",
]
