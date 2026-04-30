"""Bootstrap-ARI cluster stability evaluation."""

from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.metrics import adjusted_rand_score

from .clusterers import CLUSTERERS
from .config import BOOTSTRAP_B, BOOTSTRAP_FRAC, N_JOBS, SEED
from .multi_config import CONFIGS
from .reducers import REDUCERS


def _bootstrap_one(
    X: np.ndarray,
    reducer_name: str,
    clusterer_name: str,
    b: int,
    frac: float,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(SEED + b)
    n = X.shape[0]
    sample_size = max(10, min(n, int(round(n * frac))))
    idx = rng.choice(n, size=sample_size, replace=False)
    X_sub = X[idx]
    X_red, _ = REDUCERS[reducer_name](X_sub)
    labels, _ = CLUSTERERS[clusterer_name](X_red)
    return idx, np.asarray(labels, dtype=int)


def stability_for_config(
    X: np.ndarray,
    reducer_name: str,
    clusterer_name: str,
    B: int = BOOTSTRAP_B,
    frac: float = BOOTSTRAP_FRAC,
    n_jobs: int = N_JOBS,
) -> tuple[float, float, list[float]]:
    if B < 2:
        return 1.0, 0.0, [1.0]

    runs = Parallel(n_jobs=n_jobs)(
        delayed(_bootstrap_one)(X, reducer_name, clusterer_name, b, frac) for b in range(B)
    )

    aris: list[float] = []
    for i, j in combinations(range(B), 2):
        idx_i, lab_i = runs[i]
        idx_j, lab_j = runs[j]
        common = np.intersect1d(idx_i, idx_j)
        if len(common) < 10:
            continue
        map_i = dict(zip(idx_i.tolist(), lab_i.tolist()))
        map_j = dict(zip(idx_j.tolist(), lab_j.tolist()))
        labels_i = np.array([map_i[idx] for idx in common])
        labels_j = np.array([map_j[idx] for idx in common])
        aris.append(float(adjusted_rand_score(labels_i, labels_j)))

    if not aris:
        return float("nan"), float("nan"), []
    return float(np.mean(aris)), float(np.std(aris)), aris


def run_all(
    X: np.ndarray,
    B: int = BOOTSTRAP_B,
    frac: float = BOOTSTRAP_FRAC,
    n_jobs: int = N_JOBS,
) -> pd.DataFrame:
    rows = []
    for cid, reducer_name, clusterer_name in CONFIGS:
        mean, sd, aris = stability_for_config(X, reducer_name, clusterer_name, B, frac, n_jobs)
        rows.append(
            {
                "config_id": cid,
                "reducer": reducer_name,
                "clusterer": clusterer_name,
                "bootstrap_ari_mean": mean,
                "bootstrap_ari_std": sd,
                "bootstrap_ari_min": float(np.min(aris)) if aris else float("nan"),
                "bootstrap_ari_max": float(np.max(aris)) if aris else float("nan"),
                "bootstrap_ari_dist": aris,
            }
        )
    return pd.DataFrame(rows)
