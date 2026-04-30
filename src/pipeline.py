"""End-to-end pipeline runner."""

from __future__ import annotations

import argparse
import json
import sys
import time

import numpy as np
import pandas as pd

from . import (
    cluster_interpret,
    constraints,
    features,
    group_eval,
    group_former,
    ingest,
    multi_config,
    output,
    preprocess,
    selector,
    significance,
    stability,
)
from .config import DEMO_CACHE, GROUP_SIZE, SEED, ensure_dirs, parse_presentation
from .reducers import reduce_umap_2d


def _sample_feature_matrix(feature_matrix: pd.DataFrame, sample_n: int | None) -> pd.DataFrame:
    if sample_n is None or sample_n <= 0 or len(feature_matrix) <= sample_n:
        return feature_matrix.reset_index(drop=True)
    return feature_matrix.sample(n=sample_n, random_state=SEED).sort_values("id_student").reset_index(drop=True)


def _with_presentation(winner: pd.Series, code_module: str, code_presentation: str) -> pd.Series:
    winner = winner.copy()
    winner["presentation_module"] = code_module
    winner["presentation_code"] = code_presentation
    return winner


def _outcome_metadata(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    info = tables["info"]
    if "final_result" not in info.columns:
        return pd.DataFrame(columns=["id_student", "final_result"])
    return info[["id_student", "final_result"]].drop_duplicates("id_student").copy()


def run(
    presentation: str | None = None,
    bootstrap_b: int | None = None,
    bootstrap_frac: float | None = None,
    sample_n: int | None = None,
    n_jobs: int | None = None,
    random_baselines: int | None = None,
    fast_run: bool = False,
) -> dict[str, object]:
    ensure_dirs()
    code_module, code_presentation = parse_presentation(presentation)
    bootstrap_b = 30 if bootstrap_b is None else bootstrap_b
    bootstrap_frac = 0.80 if bootstrap_frac is None else bootstrap_frac
    n_jobs = -1 if n_jobs is None else n_jobs
    random_baselines = 100 if random_baselines is None else random_baselines

    started = time.time()
    print(f"110 Ingest {code_module}_{code_presentation}...")
    tables = ingest.run(code_module, code_presentation)

    print("120 Features...")
    feature_matrix = features.run(tables)
    feature_matrix = _sample_feature_matrix(feature_matrix, sample_n)
    if feature_matrix.empty:
        raise ValueError("No learners remain after scope filtering")

    print(f"130 Preprocess ({len(feature_matrix)} learners)...")
    ids, X_scaled, _, _, columns, clean_features = preprocess.preprocess(feature_matrix)
    features_for_output = pd.concat(
        [pd.Series(ids, name="id_student"), clean_features.reset_index(drop=True)],
        axis=1,
    )
    features_for_output = features_for_output.merge(_outcome_metadata(tables), on="id_student", how="left")

    print("160 Multi-config sweep (12 configs)...")
    metrics_df, labels_by_config, reductions = multi_config.run_all(X_scaled)

    print(f"170 Bootstrap stability (B={bootstrap_b})...")
    stability_df = stability.run_all(X_scaled, B=bootstrap_b, frac=bootstrap_frac, n_jobs=n_jobs)

    print("180 Select winning config...")
    winner, ranked = selector.select_winning(metrics_df, stability_df)
    winner = _with_presentation(winner, code_module, code_presentation)
    print(f"  Winner: {winner['config_id']} ({winner['reducer']} + {winner['clusterer']})")

    print("Refit winning configuration...")
    X_red = reductions[winner["reducer"]]
    winner_labels = labels_by_config[winner["config_id"]]

    # === Canonical cluster-ID remap ===
    # K-Means initialisation order is non-deterministic; without remapping,
    # "cluster 0" might be the at-risk persona on one run and high-engagement
    # on the next, breaking captions and cross-references silently.
    remap = cluster_interpret.canonical_cluster_order(
        X_scaled, winner_labels, columns,
    )
    winner_labels = cluster_interpret.apply_remap(winner_labels, remap)
    labels_by_config[winner["config_id"]] = winner_labels
    print(f"  Canonical remap: {remap}")

    print("Compute 2D visualization embedding...")
    try:
        reductions["umap_2d"], _ = reduce_umap_2d(X_scaled)
    except Exception as exc:
        print(f"  UMAP 2D failed ({exc}); falling back to PCA coordinates.")
        reductions["umap_2d"] = reductions.get("pca", X_scaled[:, :2])[:, :2]

    print("190 Group formation (Mode A + Mode B)...")
    groups_a = group_former.form_homogeneous(X_red, winner_labels, GROUP_SIZE)
    groups_b = group_former.form_heterogeneous(winner_labels, GROUP_SIZE)
    groups_a, constraints_a = constraints.refine_demo(groups_a, winner_labels, GROUP_SIZE)
    groups_b, constraints_b = constraints.refine_demo(groups_b, winner_labels, GROUP_SIZE)

    print("194 Group evaluation...")
    groups_random = group_former.form_random(len(ids), GROUP_SIZE)
    groups_stratified = group_former.form_stratified(features_for_output, GROUP_SIZE)
    evals = {
        "random": group_eval.evaluate_all(X_red, winner_labels, groups_random, features_for_output, GROUP_SIZE),
        "stratified": group_eval.evaluate_all(
            X_red,
            winner_labels,
            groups_stratified,
            features_for_output,
            GROUP_SIZE,
        ),
        "mode_a": group_eval.evaluate_all(X_red, winner_labels, groups_a, features_for_output, GROUP_SIZE),
        "mode_b": group_eval.evaluate_all(X_red, winner_labels, groups_b, features_for_output, GROUP_SIZE),
    }
    group_metrics = pd.DataFrame.from_dict(evals, orient="index").rename_axis("strategy").reset_index()

    print(f"195 Random-baseline significance ({random_baselines} partitions)...")
    random_distribution = significance.random_baseline_distribution(
        X_red,
        winner_labels,
        features_for_output,
        GROUP_SIZE,
        n_runs=random_baselines,
    )
    group_significance = significance.compare_to_random(group_metrics, random_distribution)
    cluster_summary = cluster_interpret.characterize_clusters(X_scaled, winner_labels, columns)

    print("196 Write outputs...")
    constraint_summary = {"mode_a": constraints_a, "mode_b": constraints_b}
    run_metadata = {
        "bootstrap_b":      bootstrap_b,
        "bootstrap_frac":   bootstrap_frac,
        "random_baselines": random_baselines,
        "sample_n":         sample_n,
        "fast_run":         bool(fast_run),
    }
    output.write(
        ids=ids,
        features=features_for_output,
        X_scaled=X_scaled,
        reductions=reductions,
        labels_by_config=labels_by_config,
        metrics_df=metrics_df,
        stability_df=stability_df,
        winner=winner,
        ranked=ranked,
        winner_labels=winner_labels,
        groups_a=groups_a,
        groups_b=groups_b,
        group_metrics=group_metrics,
        random_baseline_metrics=random_distribution,
        group_significance=group_significance,
        cluster_summary=cluster_summary,
        constraints=constraint_summary,
        cache_dir=DEMO_CACHE,
        run_metadata=run_metadata,
        columns=columns,
    )

    elapsed = time.time() - started
    print(f"Done in {elapsed:.1f}s.")
    return {
        "winner": winner,
        "ranked": ranked,
        "group_metrics": group_metrics,
        "group_significance": group_significance,
        "cluster_summary": cluster_summary,
        "cache_dir": DEMO_CACHE,
        "columns": columns,
        "elapsed_seconds": elapsed,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the collaborative learning clustering pipeline")
    parser.add_argument("--presentation", default="AAA_2014J", help="OULAD presentation, e.g. AAA_2014J")
    parser.add_argument("--bootstrap-b", type=int, default=30, help="Bootstrap resamples per configuration")
    parser.add_argument("--bootstrap-frac", type=float, default=0.80, help="Bootstrap sample fraction")
    parser.add_argument("--sample-n", type=int, default=None, help="Optional learner subsample for smoke runs")
    parser.add_argument("--n-jobs", type=int, default=-1, help="Parallel jobs for bootstrap stability")
    parser.add_argument(
        "--random-baselines",
        type=int,
        default=100,
        help="Random partitions for metric significance checks",
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Smoke mode: sample 300 learners and use B=2 bootstrap resamples",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Override the --fast safety guard that protects full-fidelity caches",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    # === --fast guard: refuse to overwrite full-fidelity cache ===
    # Must run BEFORE any args mutation (don't conflate "what user asked for"
    # with "what we'd actually do"). Reads bootstrap_b from existing meta.json;
    # if the field is missing the existing cache is unprotected — hand-patch
    # meta.json once with bootstrap_b/random_baselines/fast_run before relying
    # on this guard.
    if args.fast and not args.force:
        meta_path = DEMO_CACHE / "meta.json"
        if meta_path.exists():
            try:
                existing = json.loads(meta_path.read_text())
            except json.JSONDecodeError:
                existing = {}
            if existing.get("bootstrap_b", 0) >= 30:
                print(
                    f"ERROR: --fast would overwrite full-fidelity cache "
                    f"(bootstrap_b={existing['bootstrap_b']}). "
                    f"Pass --force to override, or run without --fast.",
                    file=sys.stderr,
                )
                raise SystemExit(2)

    if args.fast:
        args.sample_n = args.sample_n or 300
        args.bootstrap_b = min(args.bootstrap_b, 2)
        args.n_jobs = 1
        args.random_baselines = min(args.random_baselines, 20)
    run(
        presentation=args.presentation,
        bootstrap_b=args.bootstrap_b,
        bootstrap_frac=args.bootstrap_frac,
        sample_n=args.sample_n,
        n_jobs=args.n_jobs,
        random_baselines=args.random_baselines,
        fast_run=args.fast,
    )


if __name__ == "__main__":
    main()
