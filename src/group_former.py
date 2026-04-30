"""Collaborative group formation algorithms."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

from .config import FAIRNESS_ATTR, GROUP_SIZE, SEED, SIZE_TOLERANCE


def _balanced_chunks(order: list[int], G: int = GROUP_SIZE) -> list[list[int]]:
    if not order:
        return []
    n_groups = max(1, math.ceil(len(order) / G))
    groups = [[] for _ in range(n_groups)]
    for i, idx in enumerate(order):
        groups[i % n_groups].append(int(idx))
    return groups


def _add_leftovers(groups: list[list[int]], leftovers: list[int], G: int) -> list[list[int]]:
    if not leftovers:
        return groups
    if not groups:
        return _balanced_chunks(leftovers, G)
    hi = G + SIZE_TOLERANCE
    for idx in leftovers:
        target = min(range(len(groups)), key=lambda i: len(groups[i]))
        if len(groups[target]) >= hi:
            groups.append([])
            target = len(groups) - 1
        groups[target].append(int(idx))
    return groups


def form_homogeneous(
    X_red: np.ndarray,
    labels: np.ndarray,
    G: int = GROUP_SIZE,
    seed: int = SEED,
) -> list[list[int]]:
    rng = np.random.default_rng(seed)
    labels = np.asarray(labels)
    clusters = sorted(set(labels.tolist()) - {-1})
    if not clusters:
        clusters = sorted(set(labels.tolist()))

    groups: list[list[int]] = []
    leftovers: list[int] = []
    for cluster in clusters:
        members = np.where(labels == cluster)[0]
        remaining = list(map(int, rng.permutation(members)))
        while len(remaining) >= G:
            seed_idx = remaining[0]
            pool = np.array(remaining, dtype=int)
            nn = NearestNeighbors(n_neighbors=min(G, len(pool))).fit(X_red[pool])
            _, neighbors = nn.kneighbors(X_red[[seed_idx]])
            chosen = pool[neighbors[0]].astype(int).tolist()
            groups.append(chosen)
            chosen_set = set(chosen)
            remaining = [idx for idx in remaining if idx not in chosen_set]
        leftovers.extend(remaining)

    return [g for g in _add_leftovers(groups, leftovers, G) if g]


def form_heterogeneous(
    labels: np.ndarray,
    G: int = GROUP_SIZE,
    seed: int = SEED,
) -> list[list[int]]:
    rng = np.random.default_rng(seed)
    labels = np.asarray(labels)
    clusters = sorted(set(labels.tolist()) - {-1})
    if not clusters:
        clusters = sorted(set(labels.tolist()))

    pools = {cluster: list(map(int, rng.permutation(np.where(labels == cluster)[0]))) for cluster in clusters}
    n_total = sum(len(pool) for pool in pools.values())
    n_groups = max(1, math.ceil(n_total / G))
    groups = [[] for _ in range(n_groups)]

    group_idx = 0
    while any(pools.values()):
        used_in_group = set(labels[groups[group_idx]].tolist()) if groups[group_idx] else set()
        ordered_clusters = sorted(clusters, key=lambda c: (c in used_in_group, -len(pools[c]), c))
        chosen_cluster = next((c for c in ordered_clusters if pools[c]), None)
        if chosen_cluster is None:
            break
        groups[group_idx].append(int(pools[chosen_cluster].pop(0)))
        group_idx = (group_idx + 1) % n_groups

    return [group for group in groups if group]


def form_random(n_learners: int, G: int = GROUP_SIZE, seed: int = SEED) -> list[list[int]]:
    rng = np.random.default_rng(seed)
    return _balanced_chunks(list(map(int, rng.permutation(n_learners))), G)


def form_stratified(
    feature_df: pd.DataFrame,
    G: int = GROUP_SIZE,
    attr: str = FAIRNESS_ATTR,
    seed: int = SEED,
) -> list[list[int]]:
    rng = np.random.default_rng(seed)
    df = feature_df.reset_index(drop=True).copy()
    if attr in df.columns and df[attr].nunique(dropna=True) > 1:
        strata = df[attr].fillna("missing")
    elif "weighted_score" in df.columns and df["weighted_score"].nunique(dropna=True) > 1:
        strata = pd.qcut(df["weighted_score"], q=4, duplicates="drop")
    else:
        strata = pd.Series(["all"] * len(df), index=df.index)

    pools: list[list[int]] = []
    for _, idx in df.groupby(strata, sort=True, observed=False).groups.items():
        pools.append(list(map(int, rng.permutation(list(idx)))))

    order: list[int] = []
    while any(pools):
        for pool in pools:
            if pool:
                order.append(int(pool.pop(0)))
    return _balanced_chunks(order, G)
