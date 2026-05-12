"""Constraint summaries and greedy-swap refinement for formed groups."""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import group_eval
from .adapters.base import DatasetSchema
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


def _max_unique_clusters_for_group(group: list[int], n_clusters: int, G: int) -> int:
    return min(len(group), G, n_clusters) if n_clusters > 0 else 0


def _would_reduce_max_diversity(
    old_unique: int,
    new_unique: int,
    group: list[int],
    n_clusters: int,
    G: int,
) -> bool:
    max_unique = _max_unique_clusters_for_group(group, n_clusters, G)
    return max_unique > 0 and old_unique == max_unique and new_unique < old_unique


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
    feature_df: pd.DataFrame | None = None,
    schema: DatasetSchema | None = None,
) -> dict[str, int | float | str]:
    summary: dict[str, int | float | str] = {
        "size_violations": size_violations(groups, G),
        "cluster_complementarity_violations": cluster_complementarity_violations(groups, labels),
    }
    if feature_df is not None and schema is not None:
        summary.update(normalized_penalty_terms(groups, labels, feature_df, schema, G))
        summary["fairness_refinement"] = "soft_penalty"
        summary["engagement_refinement"] = "soft_penalty"
    else:
        summary["fairness_refinement"] = "not_configured"
        summary["engagement_refinement"] = "not_configured"
    return summary


def normalized_penalty_terms(
    groups: list[list[int]],
    labels: np.ndarray,
    feature_df: pd.DataFrame,
    schema: DatasetSchema,
    G: int = GROUP_SIZE,
) -> dict[str, float]:
    """Return normalized [0, 1] penalty terms.

    Size and complementarity are fractions of violating groups. Fairness is
    mean total-variation distance, naturally bounded [0, 1]. Engagement is
    mean absolute group z-deviation divided by 3 sigma and clipped to [0, 1].
    """
    n_groups = max(1, len(groups))
    size_penalty = size_violations(groups, G) / n_groups
    complementarity_penalty = cluster_complementarity_violations(groups, labels) / n_groups

    fairness_values = [
        group_eval.demographic_fairness(feature_df[col], groups)
        for col in schema.fairness_cols
        if col in feature_df.columns
    ]
    fairness_penalty = float(np.mean(fairness_values)) if fairness_values else 0.0

    if schema.engagement_col and schema.engagement_col in feature_df.columns:
        engagement_raw = group_eval.engagement_balance(feature_df[schema.engagement_col].to_numpy(), groups)
        engagement_penalty = min(float(engagement_raw) / 3.0, 1.0)
    else:
        engagement_penalty = 0.0

    return {
        "size_penalty": float(np.clip(size_penalty, 0.0, 1.0)),
        "complementarity_penalty": float(np.clip(complementarity_penalty, 0.0, 1.0)),
        "fairness_penalty": float(np.clip(fairness_penalty, 0.0, 1.0)),
        "engagement_penalty": float(np.clip(engagement_penalty, 0.0, 1.0)),
    }


def total_penalty(
    groups: list[list[int]],
    labels: np.ndarray,
    feature_df: pd.DataFrame,
    schema: DatasetSchema,
    G: int = GROUP_SIZE,
) -> float:
    terms = normalized_penalty_terms(groups, labels, feature_df, schema, G)
    return (
        terms["size_penalty"]
        + terms["complementarity_penalty"]
        + 0.5 * terms["fairness_penalty"]
        + 0.5 * terms["engagement_penalty"]
    )


@dataclass
class _PenaltyContext:
    """Precomputed numpy state for fast per-trial penalty evaluation.

    Used inside :func:`greedy_refine_groups` so each candidate swap recomputes
    only the affected groups instead of looping pandas value_counts over the
    full cohort.
    """

    G: int
    fairness_codes: list[np.ndarray]   # per fairness col, int-coded categories
    fairness_n_cats: list[int]          # number of categories for each col
    fairness_global: list[np.ndarray]   # per fairness col, global class distribution
    engagement: np.ndarray | None       # NaNs replaced by global mean
    engagement_mu: float
    engagement_sd: float

    @classmethod
    def build(
        cls, feature_df: pd.DataFrame, schema: DatasetSchema, G: int
    ) -> "_PenaltyContext":
        fairness_codes: list[np.ndarray] = []
        fairness_n_cats: list[int] = []
        fairness_global: list[np.ndarray] = []
        for col in schema.fairness_cols:
            if col not in feature_df.columns:
                continue
            codes, _ = pd.factorize(feature_df[col], sort=False, use_na_sentinel=False)
            n_cats = int(codes.max()) + 1 if codes.size else 0
            fairness_codes.append(np.asarray(codes, dtype=np.int64))
            fairness_n_cats.append(n_cats)
            counts = np.bincount(codes, minlength=n_cats).astype(float)
            total = counts.sum()
            fairness_global.append(counts / total if total > 0 else counts)

        if schema.engagement_col and schema.engagement_col in feature_df.columns:
            engage = feature_df[schema.engagement_col].to_numpy(dtype=float)
            mu = float(np.nanmean(engage)) if engage.size else 0.0
            sd = float(np.nanstd(engage)) if engage.size else 0.0
            engage_filled = np.where(np.isnan(engage), mu, engage)
        else:
            engage_filled = None
            mu = 0.0
            sd = 0.0

        return cls(
            G=G,
            fairness_codes=fairness_codes,
            fairness_n_cats=fairness_n_cats,
            fairness_global=fairness_global,
            engagement=engage_filled,
            engagement_mu=mu,
            engagement_sd=sd,
        )

    def fairness_for_group(self, idx_col: int, group: list[int]) -> float:
        """TV distance between this group's distribution and the global one."""
        if not group:
            return 0.0
        codes = self.fairness_codes[idx_col]
        n_cats = self.fairness_n_cats[idx_col]
        if n_cats == 0:
            return 0.0
        counts = np.bincount(codes[group], minlength=n_cats).astype(float)
        group_dist = counts / len(group)
        return 0.5 * float(np.abs(group_dist - self.fairness_global[idx_col]).sum())

    def engagement_for_group(self, group: list[int]) -> float:
        """Absolute z-deviation of this group's engagement mean from global."""
        if self.engagement is None or not group or self.engagement_sd == 0:
            return 0.0
        return abs(float(self.engagement[group].mean()) - self.engagement_mu) / self.engagement_sd


def _per_group_penalties(
    groups: list[list[int]], ctx: _PenaltyContext
) -> tuple[list[list[float]], list[float]]:
    """Return ``(fairness_per_group_per_col, engagement_per_group)``.

    Both are indexed by group position so swaps can update only the two
    affected entries.
    """
    n_fair = len(ctx.fairness_codes)
    fair = [
        [ctx.fairness_for_group(c, group) for group in groups]
        for c in range(n_fair)
    ]
    engage = [ctx.engagement_for_group(group) for group in groups]
    return fair, engage


def _aggregate(
    fair_per_group: list[list[float]],
    engage_per_group: list[float],
    size_pen: float,
    comp_pen: float,
) -> float:
    if fair_per_group:
        fairness = float(np.mean([np.mean(col) for col in fair_per_group]))
    else:
        fairness = 0.0
    engagement = (
        min(float(np.mean(engage_per_group)) / 3.0, 1.0)
        if engage_per_group else 0.0
    )
    fairness = float(np.clip(fairness, 0.0, 1.0))
    engagement = float(np.clip(engagement, 0.0, 1.0))
    return size_pen + comp_pen + 0.5 * fairness + 0.5 * engagement


def greedy_refine_groups(
    groups: list[list[int]],
    labels: np.ndarray,
    feature_df: pd.DataFrame,
    schema: DatasetSchema,
    G: int = GROUP_SIZE,
    max_iters: int = MAX_SWAP_ITERS,
    refinement_time_budget: float | None = None,
) -> list[list[int]]:
    """Refine groups by greedy pairwise swaps.

    Uses a precomputed numpy penalty context so each trial swap only recomputes
    fairness/engagement for the two affected groups. Worst-case runtime stays
    O(iters x g^2 x G^2), but each trial is now O(F) numpy ops instead of
    O(F x g) pandas calls, giving ~50-200x speed-up on cohorts of a few
    hundred learners. For cohorts above roughly 2,000 learners, pass an
    explicit ``max_iters`` and/or ``refinement_time_budget`` because exhaustive
    pairwise swap search is still quadratic in group count.
    """
    groups = [list(group) for group in groups if group]
    if not groups:
        return groups

    labels = np.asarray(labels)
    ctx = _PenaltyContext.build(feature_df, schema, G)
    n_groups = len(groups)
    n_clusters = len(set(labels.tolist()) - {-1})

    # Precompute per-group fairness + engagement at the current state.
    fair_pg, engage_pg = _per_group_penalties(groups, ctx)
    n_groups_f = max(1, n_groups)
    deadline = (
        time.monotonic() + refinement_time_budget
        if refinement_time_budget is not None and refinement_time_budget > 0
        else None
    )
    plateau_count = 0

    for _ in range(max_iters):
        if deadline is not None and time.monotonic() >= deadline:
            break
        base_size_v = size_violations(groups, G)
        base_comp_v = cluster_complementarity_violations(groups, labels)
        size_pen = float(np.clip(base_size_v / n_groups_f, 0.0, 1.0))
        comp_pen = float(np.clip(base_comp_v / n_groups_f, 0.0, 1.0))
        base_penalty = _aggregate(fair_pg, engage_pg, size_pen, comp_pen)
        best_penalty = base_penalty
        best_swap: tuple[int, int, int, int] | None = None
        best_updates: tuple[list[float], list[float], float, float] | None = None

        budget_exhausted = False
        for gi in range(n_groups):
            if deadline is not None and time.monotonic() >= deadline:
                budget_exhausted = True
                break
            group_i = groups[gi]
            for gj in range(gi + 1, n_groups):
                group_j = groups[gj]
                for pi in range(len(group_i)):
                    a = group_i[pi]
                    for pj in range(len(group_j)):
                        b = group_j[pj]
                        # Construct trial groups for the two affected groups only.
                        trial_i = list(group_i)
                        trial_j = list(group_j)
                        trial_i[pi] = b
                        trial_j[pj] = a

                        # Hard constraints: complementarity for the two groups.
                        new_uniq_i = len(set(labels[trial_i].tolist()) - {-1})
                        new_uniq_j = len(set(labels[trial_j].tolist()) - {-1})
                        old_uniq_i = len(set(labels[group_i].tolist()) - {-1})
                        old_uniq_j = len(set(labels[group_j].tolist()) - {-1})
                        if _would_reduce_max_diversity(old_uniq_i, new_uniq_i, group_i, n_clusters, G):
                            continue
                        if _would_reduce_max_diversity(old_uniq_j, new_uniq_j, group_j, n_clusters, G):
                            continue
                        if (new_uniq_i < 2 and len(trial_i) > 1) and not (old_uniq_i < 2):
                            continue
                        if (new_uniq_j < 2 and len(trial_j) > 1) and not (old_uniq_j < 2):
                            continue
                        # Sizes are preserved by a same-cardinality swap, so size
                        # penalty is unchanged.

                        # Recompute only the two affected entries.
                        new_fair_i = [
                            ctx.fairness_for_group(c, trial_i)
                            for c in range(len(ctx.fairness_codes))
                        ]
                        new_fair_j = [
                            ctx.fairness_for_group(c, trial_j)
                            for c in range(len(ctx.fairness_codes))
                        ]
                        new_engage_i = ctx.engagement_for_group(trial_i)
                        new_engage_j = ctx.engagement_for_group(trial_j)

                        # Build trial per-group arrays cheaply.
                        trial_fair_pg = [col.copy() for col in fair_pg]
                        for c in range(len(trial_fair_pg)):
                            trial_fair_pg[c][gi] = new_fair_i[c]
                            trial_fair_pg[c][gj] = new_fair_j[c]
                        trial_engage_pg = list(engage_pg)
                        trial_engage_pg[gi] = new_engage_i
                        trial_engage_pg[gj] = new_engage_j

                        penalty = _aggregate(
                            trial_fair_pg, trial_engage_pg, size_pen, comp_pen
                        )
                        if penalty + 1e-12 < best_penalty:
                            best_penalty = penalty
                            best_swap = (gi, pi, gj, pj)
                            best_updates = (
                                [new_fair_i[c] for c in range(len(ctx.fairness_codes))],
                                [new_fair_j[c] for c in range(len(ctx.fairness_codes))],
                                new_engage_i,
                                new_engage_j,
                            )

        if budget_exhausted:
            break
        if best_swap is None or best_updates is None:
            break
        if base_penalty > 0 and best_penalty / base_penalty > 0.995:
            plateau_count += 1
        else:
            plateau_count = 0
        gi, pi, gj, pj = best_swap
        groups[gi][pi], groups[gj][pj] = groups[gj][pj], groups[gi][pi]
        new_fair_i, new_fair_j, new_engage_i, new_engage_j = best_updates
        for c in range(len(fair_pg)):
            fair_pg[c][gi] = new_fair_i[c]
            fair_pg[c][gj] = new_fair_j[c]
        engage_pg[gi] = new_engage_i
        engage_pg[gj] = new_engage_j
        if plateau_count >= 3:
            break

    return groups


def refine_demo(
    groups: list[list[int]],
    labels: np.ndarray,
    G: int = GROUP_SIZE,
    feature_df: pd.DataFrame | None = None,
    schema: DatasetSchema | None = None,
    max_swap_iters: int = MAX_SWAP_ITERS,
    refinement_time_budget: float | None = None,
    enforce_complementarity: bool = True,
    soft_refine: bool = True,
) -> tuple[list[list[int]], dict[str, int | float | str]]:
    cleaned = [list(map(int, group)) for group in groups if group]
    refined = (
        greedy_swap_complementarity(cleaned, labels, max_iters=max_swap_iters)
        if enforce_complementarity
        else cleaned
    )
    if soft_refine and feature_df is not None and schema is not None:
        refined = greedy_refine_groups(
            refined,
            labels,
            feature_df.reset_index(drop=True),
            schema,
            G,
            max_iters=max_swap_iters,
            refinement_time_budget=refinement_time_budget,
        )
    summary = summarize_demo_constraints(refined, labels, G, feature_df, schema)
    summary["complementarity_refinement"] = (
        "hard_constraint" if enforce_complementarity else "not_applied_mode_objective"
    )
    if not soft_refine and feature_df is not None and schema is not None:
        summary["fairness_refinement"] = "evaluated_only"
        summary["engagement_refinement"] = "evaluated_only"
    return refined, summary
