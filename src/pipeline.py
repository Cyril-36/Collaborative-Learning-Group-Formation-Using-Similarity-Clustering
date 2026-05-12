"""End-to-end pipeline runner."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd

from . import (
    cluster_interpret,
    constraints,
    group_eval,
    group_former,
    multi_config,
    output,
    predict as predict_module,
    preprocess,
    selector,
    significance,
    stability,
)
from .adapters import DatasetAdapter, DatasetSchema, GenericCsvAdapter, OuladAdapter
from .config import DEMO_CACHE, GROUP_SIZE, SEED, ensure_dirs
from .reducers import reduce_umap_2d


def _sample_feature_matrix(
    feature_matrix: pd.DataFrame,
    sample_n: int | None,
    id_col: str = "id_student",
) -> pd.DataFrame:
    if sample_n is None or sample_n <= 0 or len(feature_matrix) <= sample_n:
        return feature_matrix.reset_index(drop=True)
    return (
        feature_matrix
        .sample(n=sample_n, random_state=SEED)
        .sort_values(id_col)
        .reset_index(drop=True)
    )


def _with_dataset_metadata(winner: pd.Series, schema: DatasetSchema) -> pd.Series:
    winner = winner.copy()
    winner["dataset_name"] = schema.dataset_name
    winner["adapter_name"] = schema.adapter_name
    if schema.adapter_name == "oulad" and schema.dataset_name.startswith("OULAD "):
        presentation = schema.dataset_name.replace("OULAD ", "", 1)
        parts = presentation.split("_", 1)
        if len(parts) == 2:
            winner["presentation_module"] = parts[0]
            winner["presentation_code"] = parts[1]
    return winner


def _build_default_adapter(
    presentation: str | None,
    csv_path: str | Path | None,
    id_column: str | None,
    fairness_columns: list[str] | None,
    feature_columns: list[str] | None,
    engagement_column: str | None,
    performance_column: str | None,
    outcome_column: str | None,
    stratification_column: str | None,
    display_columns: list[str] | None,
) -> DatasetAdapter:
    if csv_path is not None:
        if not id_column:
            raise ValueError("--id-column is required when --csv is used")
        return GenericCsvAdapter(
            csv_path,
            id_column,
            feature_cols=feature_columns,
            fairness_cols=fairness_columns or [],
            engagement_col=engagement_column,
            performance_col=performance_column,
            outcome_col=outcome_column,
            stratification_col=stratification_column,
            display_cols=display_columns,
        )
    return OuladAdapter(presentation, feature_cols=feature_columns)


def _stratification_col(schema: DatasetSchema) -> str | None:
    if schema.stratification_col:
        return schema.stratification_col
    if schema.fairness_cols:
        return schema.fairness_cols[0]
    return schema.performance_col


def _canonical_order_cols(schema: DatasetSchema, columns: list[str]) -> tuple[str | None, str | None]:
    primary = schema.engagement_col if schema.engagement_col in columns else None
    if primary is None and schema.performance_col in columns:
        primary = schema.performance_col
    secondary = schema.performance_col if schema.performance_col in columns else None
    return primary, secondary


def run(
    presentation: str | None = None,
    bootstrap_b: int | None = None,
    bootstrap_frac: float | None = None,
    sample_n: int | None = None,
    n_jobs: int | None = None,
    random_baselines: int | None = None,
    fast_run: bool = False,
    adapter: DatasetAdapter | None = None,
    csv_path: str | Path | None = None,
    id_column: str | None = None,
    fairness_columns: list[str] | None = None,
    feature_columns: list[str] | None = None,
    engagement_column: str | None = None,
    performance_column: str | None = None,
    outcome_column: str | None = None,
    stratification_column: str | None = None,
    display_columns: list[str] | None = None,
    cache_dir: str | Path | None = None,
    max_swap_iters: int | None = None,
    refinement_time_budget: float | None = None,
    min_k: int | None = None,
    max_k: int | None = None,
) -> dict[str, object]:
    ensure_dirs()
    bootstrap_b = 30 if bootstrap_b is None else bootstrap_b
    bootstrap_frac = 0.80 if bootstrap_frac is None else bootstrap_frac
    n_jobs = -1 if n_jobs is None else n_jobs
    random_baselines = 100 if random_baselines is None else random_baselines
    max_swap_iters = constraints.MAX_SWAP_ITERS if max_swap_iters is None else max_swap_iters
    cache_path = Path(cache_dir) if cache_dir is not None else DEMO_CACHE

    adapter = adapter or _build_default_adapter(
        presentation,
        csv_path,
        id_column,
        fairness_columns,
        feature_columns,
        engagement_column,
        performance_column,
        outcome_column,
        stratification_column,
        display_columns,
    )

    started = time.time()
    print(f"110 Load dataset via {adapter.name}...")
    raw = adapter.load()

    print("120 Build learner features...")
    feature_matrix, schema = adapter.build_features(raw)
    feature_matrix = _sample_feature_matrix(feature_matrix, sample_n, schema.id_col)
    if feature_matrix.empty:
        raise ValueError("No learners remain after scope filtering")

    print(f"130 Preprocess ({len(feature_matrix)} learners)...")
    prep = preprocess.preprocess(feature_matrix, schema=schema)
    ids, X_scaled, columns = prep.ids, prep.X, prep.feature_names
    features_for_output = feature_matrix.reset_index(drop=True).copy()

    print("160 Multi-config sweep (12 configs)...")
    if min_k is not None or max_k is not None:
        from .config import K_SWEEP as _DEFAULT_K
        lo = min_k if min_k is not None else _DEFAULT_K[0]
        hi = max_k if max_k is not None else _DEFAULT_K[-1]
        if lo > hi:
            raise ValueError(f"min_k ({lo}) must be <= max_k ({hi})")
        k_sweep = list(range(lo, hi + 1))
        print(f"  Constrained k_sweep = {k_sweep}")
    else:
        k_sweep = None
    metrics_df, labels_by_config, reductions = multi_config.run_all(X_scaled, k_sweep=k_sweep)

    print(f"170 Bootstrap stability (B={bootstrap_b})...")
    stability_df = stability.run_all(X_scaled, B=bootstrap_b, frac=bootstrap_frac, n_jobs=n_jobs)

    print("180 Select winning config...")
    winner, ranked = selector.select_winning(metrics_df, stability_df)
    winner = _with_dataset_metadata(winner, schema)
    print(f"  Winner: {winner['config_id']} ({winner['reducer']} + {winner['clusterer']})")

    print("Refit winning configuration...")
    X_red = reductions[winner["reducer"]]
    winner_labels_raw = labels_by_config[winner["config_id"]].copy()

    primary, secondary = _canonical_order_cols(schema, columns)
    remap = cluster_interpret.canonical_cluster_order(
        X_scaled,
        winner_labels_raw,
        columns,
        primary=primary,
        secondary=secondary,
    )
    winner_labels = cluster_interpret.apply_remap(winner_labels_raw, remap)
    labels_by_config[winner["config_id"]] = winner_labels
    print(f"  Canonical remap: {remap}")

    print("Compute 2D visualization embedding...")
    try:
        reductions["umap_2d"], _ = reduce_umap_2d(X_scaled)
    except Exception as exc:
        print(f"  UMAP 2D failed ({exc}); falling back to PCA coordinates.")
        reductions["umap_2d"] = reductions.get("pca", X_scaled[:, :2])[:, :2]

    print("185 Build live-predict artifacts (transformer + reducer + clusterer)...")
    try:
        predict_art = predict_module.build_predict_artifacts(
            feature_matrix=feature_matrix,
            schema=schema,
            transformer=prep.transformer,
            keep_mask=prep.keep_mask,
            feature_names=columns,
            X_scaled=X_scaled,
            winner_config_id=str(winner["config_id"]),
            winner_reducer=str(winner["reducer"]),
            winner_clusterer=str(winner["clusterer"]),
            winner_labels=winner_labels_raw,
            raw_to_canonical=remap,
        )
        predict_module.save_artifacts(predict_art, cache_path)
    except Exception as exc:
        print(f"  Live-predict artifact build failed ({exc}); demo predict page will be disabled.")

    print("190 Group formation (Mode A + Mode B)...")
    groups_a = group_former.form_homogeneous(X_red, winner_labels, GROUP_SIZE)
    groups_b = group_former.form_heterogeneous(winner_labels, GROUP_SIZE)
    groups_a, constraints_a = constraints.refine_demo(
        groups_a,
        winner_labels,
        GROUP_SIZE,
        feature_df=features_for_output,
        schema=schema,
        max_swap_iters=max_swap_iters,
        refinement_time_budget=refinement_time_budget,
        enforce_complementarity=False,
        soft_refine=False,
    )
    groups_b, constraints_b = constraints.refine_demo(
        groups_b,
        winner_labels,
        GROUP_SIZE,
        feature_df=features_for_output,
        schema=schema,
        max_swap_iters=max_swap_iters,
        refinement_time_budget=refinement_time_budget,
    )

    print("194 Group evaluation...")
    groups_random = group_former.form_random(len(ids), GROUP_SIZE)
    groups_stratified = group_former.form_stratified(
        features_for_output,
        GROUP_SIZE,
        attr=_stratification_col(schema),
        performance_col=schema.performance_col,
    )
    evals = {
        "random": group_eval.evaluate_all(X_red, winner_labels, groups_random, features_for_output, GROUP_SIZE, schema=schema),
        "stratified": group_eval.evaluate_all(
            X_red,
            winner_labels,
            groups_stratified,
            features_for_output,
            GROUP_SIZE,
            schema=schema,
        ),
        "mode_a": group_eval.evaluate_all(X_red, winner_labels, groups_a, features_for_output, GROUP_SIZE, schema=schema),
        "mode_b": group_eval.evaluate_all(X_red, winner_labels, groups_b, features_for_output, GROUP_SIZE, schema=schema),
    }
    group_metrics = pd.DataFrame.from_dict(evals, orient="index").rename_axis("strategy").reset_index()

    print(f"195 Random-baseline significance ({random_baselines} partitions)...")
    random_distribution = significance.random_baseline_distribution(
        X_red,
        winner_labels,
        features_for_output,
        GROUP_SIZE,
        n_runs=random_baselines,
        schema=schema,
    )
    group_significance = significance.compare_to_random(group_metrics, random_distribution)
    cluster_summary = cluster_interpret.characterize_clusters(
        X_scaled, winner_labels, columns, schema=schema
    )

    print("196 Write outputs...")
    constraint_summary = {"mode_a": constraints_a, "mode_b": constraints_b}
    run_metadata = {
        "bootstrap_b": bootstrap_b,
        "bootstrap_frac": bootstrap_frac,
        "random_baselines": random_baselines,
        "sample_n": sample_n,
        "fast_run": bool(fast_run),
        "cache_dir": str(cache_path),
        "max_swap_iters": max_swap_iters,
        "refinement_time_budget": refinement_time_budget,
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
        cache_dir=cache_path,
        run_metadata=run_metadata,
        columns=columns,
        schema=schema,
    )

    elapsed = time.time() - started
    print(f"Done in {elapsed:.1f}s.")
    return {
        "winner": winner,
        "ranked": ranked,
        "group_metrics": group_metrics,
        "group_significance": group_significance,
        "cluster_summary": cluster_summary,
        "cache_dir": cache_path,
        "schema": schema,
        "columns": columns,
        "elapsed_seconds": elapsed,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the collaborative learner grouping pipeline")
    parser.add_argument("--presentation", default="AAA_2014J", help="OULAD presentation, e.g. AAA_2014J")
    parser.add_argument("--csv", default=None, help="Generic learner-level CSV input")
    parser.add_argument("--id-column", default=None, help="Learner ID column for --csv")
    parser.add_argument("--feature-column", action="append", default=None, help="Feature column for --csv; repeatable")
    parser.add_argument("--fairness-column", action="append", default=None, help="Fairness column for --csv; repeatable")
    parser.add_argument("--engagement-column", default=None, help="Engagement column for balance evaluation")
    parser.add_argument("--performance-column", default=None, help="Performance column for stratified fallback")
    parser.add_argument("--outcome-column", default=None, help="Outcome column for post-hoc evaluation only")
    parser.add_argument("--stratification-column", default=None, help="Explicit stratification column")
    parser.add_argument("--display-column", action="append", default=None, help="Group table display column; repeatable")
    parser.add_argument("--cache-dir", default=None, help="Output cache directory")
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
        "--max-swap-iters",
        type=int,
        default=constraints.MAX_SWAP_ITERS,
        help="Maximum greedy swap-refinement iterations",
    )
    parser.add_argument(
        "--refinement-time-budget",
        type=float,
        default=None,
        help="Optional wall-clock budget in seconds for soft swap refinement",
    )
    parser.add_argument(
        "--min-k",
        type=int,
        default=None,
        help="Lower bound on the k-sweep used by KMeans/GMM/Agglo (default: 3)",
    )
    parser.add_argument(
        "--max-k",
        type=int,
        default=None,
        help="Upper bound on the k-sweep used by KMeans/GMM/Agglo (default: 10). "
             "Use --max-k 5 for cohorts where 'small number of profiles' must hold.",
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Smoke mode: sample 300 learners and use B=2 bootstrap resamples",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Override the --fast safety guard that protects existing full-fidelity caches",
    )
    return parser


def _guard_fast_cache(cache_dir: Path, force: bool) -> None:
    if force:
        return
    meta_path = cache_dir / "meta.json"
    if not meta_path.exists():
        return
    try:
        existing = json.loads(meta_path.read_text())
    except json.JSONDecodeError:
        existing = {}
    if "bootstrap_b" not in existing:
        print(
            "ERROR: --fast would overwrite an existing cache whose fidelity is unknown "
            "(missing bootstrap_b). Pass --force to override.",
            file=sys.stderr,
        )
        raise SystemExit(2)
    if existing.get("bootstrap_b", 0) >= 30:
        print(
            f"ERROR: --fast would overwrite full-fidelity cache "
            f"(bootstrap_b={existing['bootstrap_b']}). Pass --force to override, "
            f"or run without --fast.",
            file=sys.stderr,
        )
        raise SystemExit(2)


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    cache_dir = Path(args.cache_dir) if args.cache_dir else DEMO_CACHE

    if args.fast:
        _guard_fast_cache(cache_dir, args.force)
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
        csv_path=args.csv,
        id_column=args.id_column,
        fairness_columns=args.fairness_column,
        feature_columns=args.feature_column,
        engagement_column=args.engagement_column,
        performance_column=args.performance_column,
        outcome_column=args.outcome_column,
        stratification_column=args.stratification_column,
        display_columns=args.display_column,
        cache_dir=cache_dir,
        max_swap_iters=args.max_swap_iters,
        refinement_time_budget=args.refinement_time_budget,
        min_k=args.min_k,
        max_k=args.max_k,
    )


if __name__ == "__main__":
    main()
