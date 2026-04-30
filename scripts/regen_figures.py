"""Re-render figures from cached artifacts. Style-only, no experiments.

Use this script for visual iteration — never run `python -m src.pipeline`
just to refresh figures, since that re-fits clusterers and overwrites the
empirical artifacts in demo/demo_cache/.

Usage:
    python scripts/regen_figures.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Add repo root so `from src.output import ...` resolves when run as a script
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.output import write_figures, write_pipeline_diagram  # noqa: E402

CACHE = ROOT / "demo" / "demo_cache"
FIGURES = ROOT / "results" / "figures"


def main() -> None:
    metrics_df = pd.read_parquet(CACHE / "config_metrics.parquet")
    stability_df = pd.read_parquet(CACHE / "stability.parquet")
    winner = pd.Series(json.loads((CACHE / "winner.json").read_text()))

    # cluster_labels.parquet is wide-format: columns are id_student, C01, ..., C12.
    # Pull the column matching the winning config_id.
    labels_df = pd.read_parquet(CACHE / "cluster_labels.parquet")
    config_id = winner["config_id"]
    if config_id not in labels_df.columns:
        raise SystemExit(
            f"Winner config '{config_id}' not present in cluster_labels.parquet "
            f"(columns: {list(labels_df.columns)})"
        )
    labels = labels_df[config_id].to_numpy()

    # 2D embedding for fig8
    umap_path = CACHE / "reduced_umap_2d.npy"
    if umap_path.exists():
        X_vis = np.load(umap_path)
    else:
        # Fallback to PCA coords if UMAP wasn't persisted
        X_vis = np.load(CACHE / "reduced_pca.npy")[:, :2]

    group_metrics = pd.read_parquet(CACHE / "group_metrics.parquet")

    # Feature columns for fig2 (feature family breakdown)
    columns_path = CACHE / "columns.json"
    columns = None
    if columns_path.exists():
        columns = json.loads(columns_path.read_text())
    else:
        # Derive from features.parquet if columns.json doesn't exist yet
        feats = pd.read_parquet(CACHE / "features.parquet")
        columns = [c for c in feats.columns if c not in ("id_student", "final_result")]

    write_figures(metrics_df, stability_df, labels, X_vis, group_metrics, winner,
                  columns=columns)
    write_pipeline_diagram(CACHE / "pipeline_diagram.png")

    # Graphviz diagrams (fig1/4/5/6)
    try:
        from src.diagrams import render_all
        render_all(FIGURES)
        print("Graphviz diagrams rendered (fig1/4/5/6)")
    except ImportError:
        print("graphviz not installed — skipping flowcharts")

    print(f"Figures re-rendered from cache · n={len(labels)} · winner={config_id}")


if __name__ == "__main__":
    main()
