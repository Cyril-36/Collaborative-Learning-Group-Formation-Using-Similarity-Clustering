"""Project visual system. Single source of truth for matplotlib + Plotly + graphviz.

Variant B (Stripe Press) tokens, locked 2026-04-29 in design.md §1.

Import from any figure-producing script:

    from src.plot_style import apply_style, plotly_template, INK, ACCENT, CLUSTER_FILLS
    apply_style()

The module registers bundled fonts at import time so PNG output is deterministic
across machines. Streamlit gets fonts via a separate path (CSS @import); see
design.md §0.4.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import matplotlib as mpl
import matplotlib.font_manager as fm
import plotly.graph_objects as go


# === Variant B tokens (locked 2026-04-29) =============================
BG = "#FFFFFF"
SURFACE = "#FAFAF7"
RULE = "#E5E5E5"
INK = "#0A0A0A"
INK_MUTED = "#525252"
INK_FAINT = "#8A8A8A"
ACCENT = "#C8553D"  # terracotta — single accent, used once per surface


# === Cluster mapping ===================================================
# Per design.md §5.3 amendment: cluster fills are ink-only; centroids carry
# the accent. Differentiation is by SHAPE (CLUSTER_MARKERS), not colour —
# preserves single-accent discipline and matches academic-paper convention
# where shape carries semantic meaning.
CLUSTER_FILLS = {
    0: INK,         # high-engagement (post-canonical-remap; Pass 2)
    1: INK_MUTED,   # high-performing
    2: INK_MUTED,   # at-risk — same fill as cluster 1
    -1: "#C7C7C7",  # noise (HDBSCAN)
}
CLUSTER_MARKERS = {0: "o", 1: "s", 2: "^", -1: "x"}
CENTROID_COLOR = ACCENT
INK_RAMP = [INK, INK_MUTED, ACCENT, INK_FAINT]


# === Font stacks =======================================================
FONT_DISPLAY = ["Inter Tight", "Inter", "DejaVu Sans"]
FONT_BODY = ["Source Serif 4", "Georgia", "DejaVu Serif"]
FONT_MONO = ["JetBrains Mono", "DejaVu Sans Mono"]


# === Bundled-font registration =========================================
def _register_bundled_fonts() -> None:
    """Load bundled .ttf fonts so PNG output is deterministic across machines."""
    fonts_dir = Path(__file__).resolve().parent.parent / "assets" / "fonts"
    if not fonts_dir.exists():
        return
    for ttf in fonts_dir.glob("*.ttf"):
        try:
            fm.fontManager.addfont(str(ttf))
        except Exception:  # pragma: no cover — addfont occasionally raises on
            pass           # malformed TTFs; warning-only, never fatal


def _check_fonts() -> None:
    """Warn once if required fonts aren't available after bundle registration."""
    available = {f.name for f in fm.fontManager.ttflist}
    required = {"Source Serif 4", "Inter Tight", "JetBrains Mono"}
    missing = required - available
    if missing:
        warnings.warn(
            f"Required fonts missing — figures will fall back: {missing}. "
            f"Install via assets/fonts/*.ttf or system font manager.",
            stacklevel=2,
        )


_register_bundled_fonts()
_check_fonts()


# === matplotlib rcParams ===============================================
def apply_style() -> None:
    """Apply project rcParams. Idempotent — call once per script entrypoint.

    NOTE on tabular figures: rcParams cannot enforce monospace digit alignment
    in matplotlib because the text renderer doesn't activate OpenType `tnum`
    features. The discipline is `_use_mono_ticks(ax)` on any axes showing
    decimals — that's what actually delivers tabular alignment.
    """
    mpl.rcParams.update({
        "font.family":        "serif",
        "font.serif":         FONT_BODY,
        "font.size":          11,
        "axes.titlesize":     13,
        "axes.titleweight":   600,
        "axes.labelsize":     11,
        "xtick.labelsize":    10,
        "ytick.labelsize":    10,
        "legend.fontsize":    10,
        "figure.titlesize":   15,
        "figure.titleweight": 600,

        "figure.facecolor":   BG,
        "axes.facecolor":     BG,
        "savefig.facecolor":  BG,
        "savefig.edgecolor":  "none",
        "savefig.bbox":       "tight",
        "savefig.dpi":        200,

        # Spines — explicit (don't inherit from user matplotlibrc)
        "axes.spines.top":    False,
        "axes.spines.right":  False,
        "axes.spines.left":   True,
        "axes.spines.bottom": True,
        "axes.edgecolor":     INK_MUTED,
        "axes.linewidth":     0.8,
        "axes.labelcolor":    INK,
        "axes.titlecolor":    INK,

        "xtick.color":        INK_MUTED,
        "ytick.color":        INK_MUTED,
        "xtick.direction":    "out",
        "ytick.direction":    "out",

        "axes.grid":          True,
        "axes.grid.axis":     "y",
        "grid.color":         RULE,
        "grid.linewidth":     0.5,

        "axes.prop_cycle":    mpl.cycler(color=INK_RAMP),
        "legend.frameon":     False,
    })


def use_mono_ticks(ax) -> None:
    """Switch numeric tick labels to JetBrains Mono so decimals align.

    Call on any axes that displays decimal numbers (silhouette, ARI, p-values).
    THIS — not an rcParam — is what delivers tabular alignment in matplotlib.
    """
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_family(FONT_MONO)


# === Plotly template ===================================================
def plotly_template() -> go.layout.Template:
    """Return a Plotly template matching the matplotlib system. Apply via:

        import plotly.io as pio
        pio.templates["int396"] = plotly_template()
        pio.templates.default = "int396"
    """
    return go.layout.Template(
        layout=go.Layout(
            font=dict(
                family="Source Serif 4, Georgia, serif",
                size=13,
                color=INK,
            ),
            title=dict(
                font=dict(
                    family="Inter Tight, Inter, sans-serif",
                    size=16,
                    color=INK,
                ),
                x=0.0,
                xanchor="left",
            ),
            paper_bgcolor=BG,
            plot_bgcolor=BG,
            colorway=INK_RAMP,
            xaxis=dict(
                gridcolor=RULE,
                gridwidth=0.5,
                linecolor=INK_MUTED,
                tickcolor=INK_MUTED,
                tickfont=dict(family="JetBrains Mono, monospace", size=11),
                zeroline=False,
            ),
            yaxis=dict(
                gridcolor=RULE,
                gridwidth=0.5,
                linecolor=INK_MUTED,
                tickcolor=INK_MUTED,
                tickfont=dict(family="JetBrains Mono, monospace", size=11),
                zeroline=False,
            ),
            legend=dict(
                bgcolor="rgba(0,0,0,0)",
                bordercolor="rgba(0,0,0,0)",
                font=dict(size=11),
            ),
            margin=dict(l=48, r=24, t=48, b=48),
        )
    )


# === Graphviz defaults (Pass 2 only — kept here as the SSOT) =========
GRAPHVIZ_DEFAULTS = dict(
    graph_attr={
        "bgcolor":  BG,
        "fontname": "Inter Tight",
        "fontsize": "12",
        "rankdir":  "LR",
        "nodesep":  "0.4",
        "ranksep":  "0.6",
        "pad":      "0.4",
    },
    node_attr={
        "shape":     "box",
        "style":     "filled",
        "fillcolor": SURFACE,
        "color":     INK_MUTED,
        "fontname":  "Inter Tight",
        "fontsize":  "11",
        "fontcolor": INK,
        "penwidth":  "0.8",
        "margin":    "0.2,0.1",
    },
    edge_attr={
        "color":     INK_MUTED,
        "penwidth":  "0.6",
        "arrowsize": "0.6",
        "fontname":  "JetBrains Mono",
        "fontsize":  "9",
        "fontcolor": INK_MUTED,
    },
)


__all__ = [
    "BG", "SURFACE", "RULE",
    "INK", "INK_MUTED", "INK_FAINT", "ACCENT",
    "CLUSTER_FILLS", "CLUSTER_MARKERS", "CENTROID_COLOR", "INK_RAMP",
    "FONT_DISPLAY", "FONT_BODY", "FONT_MONO",
    "apply_style", "use_mono_ticks", "plotly_template",
    "GRAPHVIZ_DEFAULTS",
]
