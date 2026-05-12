"""Project visual system. Single source of truth for matplotlib + Plotly + graphviz.

**Linear / Vercel anchor (dark · clinical · projection-sized).**
Anchor decided 2026-05-09 in design.md §1. Previous Stripe Press tokens are
preserved as ``src/plot_style_stripe.py.bak`` and can be restored by
``cp src/plot_style_stripe.py.bak src/plot_style.py``.

Public surface unchanged — every name imported by the rest of the project is
still exported with the same identifier, only the *values* changed:

    BG, SURFACE, RULE, INK, INK_MUTED, INK_FAINT, ACCENT
    CLUSTER_FILLS, CLUSTER_MARKERS, CENTROID_COLOR, INK_RAMP
    FONT_DISPLAY, FONT_BODY, FONT_MONO
    apply_style, use_mono_ticks, plotly_template, GRAPHVIZ_DEFAULTS

The discipline that defines this anchor:
    1. Single accent (``ACCENT``) reserved for SIGNIFICANT outcomes only —
       p ≤ 0.05 cells, the winning configuration row, the live-predict
       assigned cluster, active selections. Everything else is grayscale.
    2. Type sized for ~3-5 m projection: tick fonts ≥ 14px, body ≥ 16px,
       headlines 28-40px. No 12px text anywhere.
    3. Backgrounds are warm-near-black (#08090A), not pure #000 — pure
       black on a projector reads as a void; a hint of warmth keeps the
       surface feeling like a real product.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import matplotlib as mpl
import matplotlib.font_manager as fm
import plotly.graph_objects as go


# === Linear / Vercel dark tokens (locked 2026-05-09) ===================
BG = "#08090A"             # canvas — near-black, slightly warm
SURFACE = "#101113"        # cards, sidebar
SURFACE_RAISED = "#16171A" # hover / active row
RULE = "#1F2023"           # dividers, axis lines, table borders
INK = "#F4F4F5"            # primary text
INK_MUTED = "#A1A1AA"      # secondary text, axis labels
INK_FAINT = "#71717A"      # tertiary text, captions
ACCENT = "#22D3EE"         # single accent — cyan, used only for SIGNIFICANCE
ACCENT_DIM = "#0E7490"     # accent backgrounds, hover states for accent elements
POSITIVE = "#34D399"       # tertiary signal — gain / improvement (sparingly)
NEGATIVE = "#F87171"       # tertiary signal — loss / regression  (sparingly)


# === Cluster mapping ===================================================
# On a dark canvas the contrast logic flips: the "primary" cluster gets the
# brightest fill (INK ≈ white), with monotone steps down. The accent (cyan)
# is reserved for centroids and active-selection halo, NOT for cluster
# membership — so a single cyan hit on a chart always means "this is the one
# you should look at" (winning config, predicted assignment, p ≤ 0.05).
CLUSTER_FILLS = {
    0: INK,           # primary cluster (post-canonical-remap)
    1: INK_MUTED,     # secondary
    2: INK_FAINT,     # tertiary
    -1: "#3F3F46",    # noise (HDBSCAN)
}
CLUSTER_MARKERS = {0: "o", 1: "s", 2: "^", -1: "x"}
CENTROID_COLOR = ACCENT
INK_RAMP = [INK, INK_MUTED, INK_FAINT, ACCENT_DIM]


# === Font stacks =======================================================
# Inter throughout — Linear-faithful, projection-legible. JetBrains Mono
# for tabular numerics so silhouette / ARI / p-values align column-wise.
FONT_DISPLAY = ["Inter", "system-ui", "-apple-system", "Segoe UI", "Helvetica Neue", "sans-serif"]
FONT_BODY = ["Inter", "system-ui", "-apple-system", "Segoe UI", "Helvetica Neue", "sans-serif"]
FONT_MONO = ["JetBrains Mono", "SF Mono", "Menlo", "Consolas", "monospace"]


# === Bundled-font registration =========================================
def _register_bundled_fonts() -> None:
    """Load bundled .ttf fonts so PNG output is deterministic across machines."""
    fonts_dir = Path(__file__).resolve().parent.parent / "assets" / "fonts"
    if not fonts_dir.exists():
        return
    for ttf in fonts_dir.glob("*.ttf"):
        try:
            fm.fontManager.addfont(str(ttf))
        except Exception:  # pragma: no cover
            pass


def _check_fonts() -> None:
    """Warn once if Inter / JetBrains Mono aren't available after registration."""
    available = {f.name for f in fm.fontManager.ttflist}
    required = {"Inter", "JetBrains Mono"}
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
    in matplotlib because the text renderer doesn't activate OpenType ``tnum``
    features. The discipline is :func:`use_mono_ticks` on any axes showing
    decimals — that's what actually delivers tabular alignment.
    """
    mpl.rcParams.update({
        # --- Type ramp (sized for 3-5 m projection) ---------------------
        "font.family":        "sans-serif",
        "font.sans-serif":    FONT_DISPLAY,
        "font.size":          14,
        "axes.titlesize":     18,
        "axes.titleweight":   600,
        "axes.labelsize":     14,
        "xtick.labelsize":    13,
        "ytick.labelsize":    13,
        "legend.fontsize":    13,
        "figure.titlesize":   22,
        "figure.titleweight": 600,

        # --- Backgrounds ------------------------------------------------
        "figure.facecolor":   BG,
        "axes.facecolor":     BG,
        "savefig.facecolor":  BG,
        "savefig.edgecolor":  "none",
        "savefig.bbox":       "tight",
        "savefig.dpi":        200,

        # --- Spines (explicit; do not inherit user matplotlibrc) --------
        "axes.spines.top":    False,
        "axes.spines.right":  False,
        "axes.spines.left":   True,
        "axes.spines.bottom": True,
        "axes.edgecolor":     RULE,
        "axes.linewidth":     0.8,
        "axes.labelcolor":    INK_MUTED,
        "axes.titlecolor":    INK,

        "xtick.color":        INK_MUTED,
        "ytick.color":        INK_MUTED,
        "xtick.direction":    "out",
        "ytick.direction":    "out",

        # --- Grid (subtle) ----------------------------------------------
        "axes.grid":          True,
        "axes.grid.axis":     "y",
        "grid.color":         RULE,
        "grid.linewidth":     0.5,

        # --- Color cycle + legend frame ---------------------------------
        "axes.prop_cycle":    mpl.cycler(color=INK_RAMP),
        "legend.frameon":     False,
        "text.color":         INK,
    })


def use_mono_ticks(ax) -> None:
    """Switch numeric tick labels to JetBrains Mono so decimals align."""
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_family(FONT_MONO)


# === Plotly template ===================================================
def plotly_template() -> go.layout.Template:
    """Return a Plotly template matching the matplotlib system. Apply via:

        import plotly.io as pio
        pio.templates["int396"] = plotly_template()
        pio.templates.default = "int396"
    """
    sans = "Inter, system-ui, sans-serif"
    mono = "JetBrains Mono, SF Mono, Menlo, monospace"
    return go.layout.Template(
        layout=go.Layout(
            font=dict(family=sans, size=14, color=INK),
            title=dict(
                font=dict(family=sans, size=20, color=INK, weight=600),
                x=0.0,
                xanchor="left",
            ),
            paper_bgcolor=BG,
            plot_bgcolor=BG,
            colorway=INK_RAMP,
            xaxis=dict(
                gridcolor=RULE,
                gridwidth=0.5,
                linecolor=RULE,
                tickcolor=RULE,
                tickfont=dict(family=mono, size=13, color=INK_MUTED),
                title=dict(font=dict(family=sans, size=14, color=INK_MUTED)),
                zeroline=False,
            ),
            yaxis=dict(
                gridcolor=RULE,
                gridwidth=0.5,
                linecolor=RULE,
                tickcolor=RULE,
                tickfont=dict(family=mono, size=13, color=INK_MUTED),
                title=dict(font=dict(family=sans, size=14, color=INK_MUTED)),
                zeroline=False,
            ),
            legend=dict(
                bgcolor="rgba(0,0,0,0)",
                bordercolor="rgba(0,0,0,0)",
                font=dict(size=13, color=INK_MUTED),
            ),
            margin=dict(l=56, r=24, t=56, b=56),
            hoverlabel=dict(
                bgcolor=SURFACE_RAISED,
                bordercolor=RULE,
                font=dict(family=sans, size=13, color=INK),
            ),
        )
    )


# === Graphviz defaults =================================================
GRAPHVIZ_DEFAULTS = dict(
    graph_attr={
        "bgcolor":  BG,
        "fontname": "Inter",
        "fontsize": "13",
        "rankdir":  "LR",
        "nodesep":  "0.4",
        "ranksep":  "0.6",
        "pad":      "0.4",
        "fontcolor": INK,
    },
    node_attr={
        "shape":     "box",
        "style":     "filled,rounded",
        "fillcolor": SURFACE,
        "color":     RULE,
        "fontname":  "Inter",
        "fontsize":  "12",
        "fontcolor": INK,
        "penwidth":  "0.8",
        "margin":    "0.2,0.12",
    },
    edge_attr={
        "color":     INK_FAINT,
        "penwidth":  "0.7",
        "arrowsize": "0.6",
        "fontname":  "JetBrains Mono",
        "fontsize":  "10",
        "fontcolor": INK_MUTED,
    },
)


__all__ = [
    "BG", "SURFACE", "SURFACE_RAISED", "RULE",
    "INK", "INK_MUTED", "INK_FAINT", "ACCENT", "ACCENT_DIM",
    "POSITIVE", "NEGATIVE",
    "CLUSTER_FILLS", "CLUSTER_MARKERS", "CENTROID_COLOR", "INK_RAMP",
    "FONT_DISPLAY", "FONT_BODY", "FONT_MONO",
    "apply_style", "use_mono_ticks", "plotly_template",
    "GRAPHVIZ_DEFAULTS",
]
