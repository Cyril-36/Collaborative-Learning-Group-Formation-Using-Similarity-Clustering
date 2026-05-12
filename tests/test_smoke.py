from __future__ import annotations

import numpy as np
import pandas as pd

from src import cluster_interpret, constraints, features, group_eval, group_former, output, pipeline, preprocess, significance
from src.adapters import DatasetSchema, GenericCsvAdapter, OuladAdapter


def test_preprocess_drops_constant_and_scales():
    df = pd.DataFrame(
        {
            "id_student": [1, 2, 3, 4],
            "feature": [1.0, 2.0, None, 4.0],
            "constant": [5, 5, 5, 5],
        }
    )
    ids, X, _, _, cols, clean = preprocess.preprocess(df)
    assert ids.tolist() == [1, 2, 3, 4]
    assert cols == ["feature"]
    assert X.shape == (4, 1)
    assert clean.shape == (4, 1)
    assert np.isfinite(X).all()


def test_generic_csv_adapter_and_schema_preprocess():
    adapter = GenericCsvAdapter(
        "tests/fixtures/synthetic_learners.csv",
        "learner_id",
        fairness_cols=["gender"],
        engagement_col="login_count",
        performance_col="quiz_average",
        outcome_col="final_grade",
        feature_cols=["login_count", "quiz_average", "forum_posts", "learning_style"],
    )
    df, schema = adapter.build_features(adapter.load())
    assert "id_student" in df.columns
    assert schema.source_id_col == "learner_id"
    assert schema.fairness_cols == ["gender"]
    assert schema.outcome_col not in schema.clustering_feature_cols()

    result = preprocess.preprocess(df, schema)
    assert result.X.shape[0] == len(df)
    assert any(name.startswith("learning_style_") for name in result.feature_names)
    assert "final_grade" not in result.feature_names


def test_generic_csv_inference_excludes_stratification_metadata():
    adapter = GenericCsvAdapter(
        "tests/fixtures/synthetic_learners.csv",
        "learner_id",
        stratification_col="socioeconomic_band",
    )
    _, schema = adapter.build_features(adapter.load())
    assert schema.stratification_col == "socioeconomic_band"
    assert "socioeconomic_band" not in schema.feature_cols


def test_group_metrics_return_numbers():
    X = np.array([[0.0, 0.0], [0.1, 0.0], [3.0, 3.0], [3.1, 3.0]])
    labels = np.array([0, 0, 1, 1])
    groups = [[0, 2], [1, 3]]
    feature_df = pd.DataFrame({"total_clicks": [1, 2, 3, 4], "imd_band_ord": [0, 1, 0, 1]})
    schema = DatasetSchema(
        dataset_name="metrics",
        adapter_name="test",
        source_id_col="id_student",
        engagement_col="total_clicks",
        fairness_cols=["imd_band_ord"],
    )
    metrics = group_eval.evaluate_all(X, labels, groups, feature_df, G=2, schema=schema)
    assert {
        "intra_group_distance",
        "inter_group_variance",
        "complementarity",
        "engagement_balance",
        "demographic_fairness",
        "cluster_coverage",
    }.issubset(metrics)
    assert all(np.isfinite(value) for value in metrics.values())


def test_outcome_metrics_are_posthoc_group_checks():
    X = np.array([[0.0, 0.0], [0.1, 0.0], [3.0, 3.0], [3.1, 3.0]])
    labels = np.array([0, 0, 1, 1])
    groups = [[0, 2], [1, 3]]
    feature_df = pd.DataFrame(
        {
            "total_clicks": [1, 2, 3, 4],
            "imd_band_ord": [0, 1, 0, 1],
            "final_result": ["Pass", "Fail", "Distinction", "Withdrawn"],
        }
    )
    schema = DatasetSchema(
        dataset_name="outcomes",
        adapter_name="test",
        source_id_col="id_student",
        engagement_col="total_clicks",
        fairness_cols=["imd_band_ord"],
        outcome_col="final_result",
    )
    metrics = group_eval.evaluate_all(X, labels, groups, feature_df, G=2, schema=schema)
    for key in ["outcome_diversity", "at_risk_concentration", "high_risk_group_rate", "outcome_balance"]:
        assert key in metrics
        assert np.isfinite(metrics[key])


def test_numeric_outcome_balance_uses_scores_directly():
    outcomes = pd.Series([0.0, 20.0, 10.0, 10.0])
    balanced = group_eval.outcome_balance(outcomes, [[0, 1], [2, 3]])
    imbalanced = group_eval.outcome_balance(outcomes, [[0, 2], [1, 3]])

    assert balanced == 0.0
    assert imbalanced > balanced


def test_mapped_categorical_outcome_balance_uses_outcome_score():
    outcomes = pd.Series(["Withdrawn", "Distinction", "Fail", "Pass"])
    balanced = group_eval.outcome_balance(outcomes, [[0, 1], [2, 3]])
    imbalanced = group_eval.outcome_balance(outcomes, [[0, 2], [1, 3]])

    assert balanced < 1e-12
    assert imbalanced > balanced


def test_unmapped_categorical_outcome_balance_stays_zero():
    outcomes = pd.Series(["low", "medium", "high", "medium"])
    assert group_eval.outcome_balance(outcomes, [[0, 1], [2, 3]]) == 0.0


def test_schema_driven_group_eval_skips_missing_roles():
    X = np.array([[0.0, 0.0], [0.1, 0.0], [3.0, 3.0], [3.1, 3.0]])
    labels = np.array([0, 0, 1, 1])
    groups = [[0, 2], [1, 3]]
    feature_df = pd.DataFrame({"id_student": [1, 2, 3, 4], "feature": [1, 2, 3, 4]})
    schema = DatasetSchema(
        dataset_name="minimal",
        adapter_name="test",
        source_id_col="id_student",
        numeric_feature_cols=["feature"],
    )
    metrics = group_eval.evaluate_all(X, labels, groups, feature_df, G=2, schema=schema)
    assert "cluster_coverage" in metrics
    assert "engagement_balance" not in metrics
    assert "demographic_fairness" not in metrics


def test_penalty_normalization_and_refinement_preserve_members():
    labels = np.array([0, 0, 1, 1, 0, 1])
    groups = [[0, 1, 2], [3, 4, 5]]
    feature_df = pd.DataFrame(
        {
            "id_student": [1, 2, 3, 4, 5, 6],
            "gender": ["F", "F", "M", "M", "F", "M"],
            "login_count": [10, 12, 100, 98, 14, 95],
        }
    )
    schema = DatasetSchema(
        dataset_name="penalty",
        adapter_name="test",
        source_id_col="id_student",
        fairness_cols=["gender"],
        engagement_col="login_count",
    )
    terms = constraints.normalized_penalty_terms(groups, labels, feature_df, schema, G=3)
    assert all(0.0 <= value <= 1.0 for value in terms.values())
    refined, summary = constraints.refine_demo(groups, labels, G=3, feature_df=feature_df, schema=schema)
    assert sorted(idx for group in refined for idx in group) == list(range(len(labels)))
    assert summary["size_violations"] == 0


def test_minority_first_heterogeneous_seeds_smallest_cluster():
    labels = np.array([0] * 8 + [1] * 8 + [2] * 3)
    groups = group_former.form_heterogeneous_minority_first(labels, G=3, seed=7)

    assert sorted(idx for group in groups for idx in group) == list(range(len(labels)))
    minority_groups = sum(2 in set(labels[group].tolist()) for group in groups)
    assert minority_groups == min(len(groups), int(np.sum(labels == 2)))
    assert all(len(group) <= 3 for group in groups)


def test_refinement_guard_preserves_max_diversity_groups():
    labels = np.array([0, 1, 2, 1, 0, 0, 1, 1])
    assert constraints._would_reduce_max_diversity(
        old_unique=3,
        new_unique=2,
        group=[0, 1, 2, 3],
        n_clusters=3,
        G=4,
    )
    assert not constraints._would_reduce_max_diversity(
        old_unique=2,
        new_unique=2,
        group=[4, 5, 6, 7],
        n_clusters=3,
        G=4,
    )


def test_engagement_features_include_collaboration_signals():
    vle = pd.DataFrame(
        {
            "id_student": [1, 1, 1, 2],
            "id_site": [10, 11, 12, 10],
            "date": [1, 2, 3, 1],
            "sum_click": [5, 7, 11, 2],
        }
    )
    vle_meta = pd.DataFrame(
        {
            "id_site": [10, 11, 12],
            "activity_type": ["forumng", "oucollaborate", "resource"],
        }
    )
    out = features.engagement_features(vle, vle_meta)
    row = out[out["id_student"].eq(1)].iloc[0]
    assert row["collaborative_clicks"] == 12
    assert row["forum_clicks"] == 5
    assert row["live_collab_clicks"] == 7
    assert row["collaborative_active_days"] == 2
    assert row["collaboration_click_ratio"] == 12 / 23


def test_oulad_adapter_schema_from_tables():
    raw = {
        "info": pd.DataFrame(
            {
                "id_student": [1, 2],
                "age_band": ["0-35", "35-55"],
                "imd_band": ["0-10%", "10-20%"],
                "highest_education": ["A Level or Equivalent", "HE Qualification"],
                "disability": ["N", "Y"],
                "gender": ["M", "F"],
                "num_of_prev_attempts": [0, 1],
                "studied_credits": [60, 60],
                "region": ["x", "y"],
                "final_result": ["Pass", "Fail"],
            }
        ),
        "registration": pd.DataFrame({"id_student": [1, 2], "date_registration": [-10, -5]}),
        "vle": pd.DataFrame(columns=["id_student", "id_site", "date", "sum_click"]),
        "vle_meta": pd.DataFrame(columns=["id_site", "activity_type"]),
        "assessment": pd.DataFrame(columns=["id_assessment", "id_student", "date_submitted", "score"]),
        "assessments": pd.DataFrame(columns=["id_assessment", "date", "weight", "assessment_type"]),
    }
    matrix, schema = OuladAdapter("AAA_2014J").build_features(raw)
    assert "id_student" in matrix.columns
    assert schema.adapter_name == "oulad"
    assert "imd_band_ord" in schema.fairness_cols
    assert schema.outcome_col == "final_result"


def test_oulad_adapter_respects_feature_subset():
    raw = {
        "info": pd.DataFrame(
            {
                "id_student": [1, 2],
                "age_band": ["0-35", "35-55"],
                "imd_band": ["0-10%", "10-20%"],
                "highest_education": ["A Level or Equivalent", "HE Qualification"],
                "disability": ["N", "Y"],
                "gender": ["M", "F"],
                "num_of_prev_attempts": [0, 1],
                "studied_credits": [60, 60],
                "region": ["x", "y"],
                "final_result": ["Pass", "Fail"],
            }
        ),
        "registration": pd.DataFrame({"id_student": [1, 2], "date_registration": [-10, -5]}),
        "vle": pd.DataFrame(columns=["id_student", "id_site", "date", "sum_click"]),
        "vle_meta": pd.DataFrame(columns=["id_site", "activity_type"]),
        "assessment": pd.DataFrame(columns=["id_assessment", "id_student", "date_submitted", "score"]),
        "assessments": pd.DataFrame(columns=["id_assessment", "date", "weight", "assessment_type"]),
    }
    _, schema = OuladAdapter(
        "AAA_2014J",
        feature_cols=["total_clicks", "weighted_score"],
    ).build_features(raw)
    assert schema.feature_cols == ["total_clicks", "weighted_score"]
    assert schema.numeric_feature_cols == ["total_clicks", "weighted_score"]
    assert schema.outcome_col == "final_result"


def test_cluster_interpretation_and_significance_shapes():
    X = np.array([[0.0, 2.0], [0.1, 2.2], [3.0, -1.0], [3.1, -0.8]])
    labels = np.array([0, 0, 1, 1])
    summary = cluster_interpret.characterize_clusters(X, labels, ["collaborative_clicks", "weighted_score"])
    assert list(summary["cluster"]) == [0, 1]
    feature_df = pd.DataFrame(
        {
            "total_clicks": [1, 2, 3, 4],
            "imd_band_ord": [0, 1, 0, 1],
            "final_result": ["Pass", "Fail", "Distinction", "Withdrawn"],
        }
    )
    schema = DatasetSchema(
        dataset_name="significance",
        adapter_name="test",
        source_id_col="id_student",
        engagement_col="total_clicks",
        fairness_cols=["imd_band_ord"],
        outcome_col="final_result",
    )
    random_dist = significance.random_baseline_distribution(X, labels, feature_df, G=2, n_runs=3, schema=schema)
    metrics = pd.DataFrame(
        [
            {"strategy": "mode_a", **group_eval.evaluate_all(X, labels, [[0, 1], [2, 3]], feature_df, G=2, schema=schema)},
            {"strategy": "mode_b", **group_eval.evaluate_all(X, labels, [[0, 2], [1, 3]], feature_df, G=2, schema=schema)},
        ]
    )
    compared = significance.compare_to_random(metrics, random_dist)
    assert {"strategy", "metric", "preferred_p_value"}.issubset(compared.columns)
    mode_a_intra = compared[
        compared["strategy"].eq("mode_a") & compared["metric"].eq("intra_group_distance")
    ].iloc[0]
    mode_b_intra = compared[
        compared["strategy"].eq("mode_b") & compared["metric"].eq("intra_group_distance")
    ].iloc[0]
    assert mode_a_intra["preferred_direction"] == "lower"
    assert mode_b_intra["preferred_direction"] == "higher"


def test_group_formers_cover_all_learners():
    X = np.random.default_rng(42).normal(size=(13, 3))
    labels = np.array([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 2])
    for groups in [
        group_former.form_homogeneous(X, labels, G=4),
        group_former.form_heterogeneous(labels, G=4),
        group_former.form_random(len(labels), G=4),
    ]:
        members = sorted(idx for group in groups for idx in group)
        assert members == list(range(len(labels)))


def test_generic_csv_pipeline_smoke(tmp_path):
    cache_dir = tmp_path / "generic-cache"
    result = pipeline.run(
        csv_path="tests/fixtures/synthetic_learners.csv",
        id_column="learner_id",
        fairness_columns=["gender"],
        engagement_column="login_count",
        performance_column="quiz_average",
        outcome_column="final_grade",
        feature_columns=["login_count", "quiz_average", "forum_posts", "learning_style"],
        cache_dir=cache_dir,
        bootstrap_b=2,
        random_baselines=3,
        n_jobs=1,
        fast_run=True,
    )
    assert result["schema"].adapter_name == "generic_csv"
    groups = pd.read_parquet(cache_dir / "groups_mode_b.parquet")
    assert sorted(groups["id_student"].tolist()) == list(range(1001, 1025))
    assert (cache_dir / "schema.json").exists()
    meta = pd.read_json(cache_dir / "meta.json", typ="series")
    assert meta["adapter_name"] == "generic_csv"


def test_json_output_is_strict(tmp_path):
    path = tmp_path / "payload.json"
    output._write_json(
        path,
        {
            "python_nan": float("nan"),
            "numpy_nan": np.float64(np.nan),
            "array_nan": np.array([1.0, np.nan]),
            "pandas_na": pd.NA,
            "ok": np.int64(7),
        },
    )
    text = path.read_text()
    assert "NaN" not in text
    assert '"python_nan": null' in text
    assert '"numpy_nan": null' in text
    assert '"array_nan": [\n    1.0,\n    null\n  ]' in text
    assert '"pandas_na": null' in text


def test_performance_features_ignore_missing_scores_for_weighted_score_and_slope():
    assessment = pd.DataFrame(
        {
            "id_assessment": [1, 2, 3],
            "id_student": [10, 10, 10],
            "date_submitted": [10, 20, 30],
            "score": [80, np.nan, 100],
        }
    )
    assessments = pd.DataFrame(
        {
            "id_assessment": [1, 2, 3],
            "date": [10, 20, 30],
            "weight": [10, 10, 20],
            "assessment_type": ["TMA", "TMA", "TMA"],
        }
    )
    out = features.performance_features(assessment, assessments)
    row = out.iloc[0]
    assert row["weighted_score"] == (80 * 10 + 100 * 20) / 30
    assert np.isfinite(row["score_trajectory_slope"])


def test_performance_slope_is_nan_when_insufficient_points():
    assessment = pd.DataFrame(
        {
            "id_assessment": [1],
            "id_student": [10],
            "date_submitted": [10],
            "score": [80],
        }
    )
    assessments = pd.DataFrame(
        {
            "id_assessment": [1],
            "date": [10],
            "weight": [10],
            "assessment_type": ["TMA"],
        }
    )
    out = features.performance_features(assessment, assessments)
    assert np.isnan(out.iloc[0]["score_trajectory_slope"])
