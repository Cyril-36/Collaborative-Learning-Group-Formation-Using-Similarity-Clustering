"""Constraint summaries and greedy-swap refinement for formed groups."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import FAIRNESS_TVD_MAX, GROUP_SIZE, MAX_SWAP_ITERS, SIZE_TOLERANCE


def tv_distance(p: pd.Series, q: pd.Series) -> float:
    keys = set(p.index) | set(q.index)
    return float(0.5 * sum(abs(p.get(key, 0.0) - q.get(key, 0.0)) for key in keys))


def fairness_violation(group_df: pd.DataFrame, class_df: pd.DataFrame, attr: str) -> float:
    if attr not in group_df.columns or attr not in class_df.columns:
        return 0.0
    group_dist = group_df[attr].value_counts(normalize=True)
    class_dist = class_df[attr].value_counts(normalize=True)
    return max(0.0, tv_distance(group_dist, class_dist) - FAIRNESS_TVD_MAX)


def size_violations(
    groups: list[list[int]],
    G: int = GROUP_SIZE,
    tolerance: int = SIZE_TOLERANCE,
) -> int:
    lo, hi = G - tolerance, G + tolerance
    return int(sum(not (lo <= len(group) <= hi) for group in groups))


def cluster_complementarity_violations(
    groups: list[list[int]],
    labels: np.ndarray,
    min_unique_clusters: int = 2,
) -> int:
    labels = np.asarray(labels)
    violations = 0
    for group in groups:
        present = set(labels[group].tolist()) - {-1}
        if len(group) > 1 and len(present) < min_unique_clusters:
            violations += 1
    return int(violations)


def _group_unique_clusters(group: list[int], labels: np.ndarray) -> int:
    return len(set(labels[group].tolist()) - {-1})


def greedy_swap_complementarity(
    groups: list[list[int]],
    labels: np.ndarray,
    min_unique_clusters: int = 2,
    max_iters: int = MAX_SWAP_ITERS,
) -> list[list[int]]:
    """Reduce cluster-complementarity violations via greedy pairwise swaps.

    For each violating group, try swapping one of its members with a member
    from a different group such that both groups gain or maintain unique-cluster
    count. Iterates until no violation remains or the iteration budget is
    exhausted.
    """
    labels = np.asarray(labels)
    groups = [list(g) for g in groups]
    n_groups = len(groups)

    for _ in range(max_iters):
        # find the first violating group
        violating = [
            gi for gi in range(n_groups)
            if len(groups[gi]) > 1
            and _group_unique_clusters(groups[gi], labels) < min_unique_clusters
        ]
        if not violating:
            break

        improved = False
        for gi in violating:
            current_score = _group_unique_clusters(groups[gi], labels)
            best_gain = 0
            best_swap = None  # (gi, pos_i, gj, pos_j)

            for gj in range(n_groups):
                if gi == gj:
                    continue
                donor_score = _group_unique_clusters(groups[gj], labels)
                for pos_i, idx_i in enumerate(groups[gi]):
                    for pos_j, idx_j in enumerate(groups[gj]):
                        # simulate swap
                        trial_gi = [idx_j if k == pos_i else v for k, v in enumerate(groups[gi])]
                        trial_gj = [idx_i if k == pos_j else v for k, v in enumerate(groups[gj])]
                        new_gi = _group_unique_clusters(trial_gi, labels)
                        new_gj = _group_unique_clusters(trial_gj, labels)
                        gain = (new_gi - current_score) + (new_gj - donor_score)
                        if gain > best_gain:
                            best_gain = gain
                            best_swap = (gi, pos_i, gj, pos_j)

            if best_swap is not None:
                gi2, pos_i, gj2, pos_j = best_swap
                groups[gi2][pos_i], groups[gj2][pos_j] = groups[gj2][pos_j], groups[gi2][pos_i]
                improved = True
                break

        if not improved:
            break  # stuck — no beneficial swap exists

    return groups


def summarize_demo_constraints(
    groups: list[list[int]],
    labels: np.ndarray,
    G: int = GROUP_SIZE,
) -> dict[str, int | str]:
    return {
        "size_violations": size_violations(groups, G),
        "cluster_complementarity_violations": cluster_complementarity_violations(groups, labels),
        "fairness_refinement": "evaluated_only",
        "engagement_refinement": "evaluated_only",
    }


def refine_demo(
    groups: list[list[int]],
    labels: np.ndarray,
    G: int = GROUP_SIZE,
) -> tuple[list[list[int]], dict[str, int | str]]:
    cleaned = [list(map(int, group)) for group in groups if group]
    # Apply greedy swap to reduce complementarity violations
    refined = greedy_swap_complementarity(cleaned, labels)
    summary = summarize_demo_constraints(refined, labels, G)
    return refined, summary
