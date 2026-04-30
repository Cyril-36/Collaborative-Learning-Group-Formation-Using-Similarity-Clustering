from __future__ import annotations

import numpy as np
import pandas as pd

from src import cluster_interpret, features, group_eval, group_former, output, preprocess, significance


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


def test_group_metrics_return_numbers():
    X = np.array([[0.0, 0.0], [0.1, 0.0], [3.0, 3.0], [3.1, 3.0]])
    labels = np.array([0, 0, 1, 1])
    groups = [[0, 2], [1, 3]]
    features = pd.DataFrame({"total_clicks": [1, 2, 3, 4], "imd_band_ord": [0, 1, 0, 1]})
    metrics = group_eval.evaluate_all(X, labels, groups, features, G=2)
    assert set(metrics) == {
        "intra_group_distance",
        "inter_group_variance",
        "complementarity",
        "engagement_balance",
        "demographic_fairness",
        "cluster_coverage",
    }
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
    metrics = group_eval.evaluate_all(X, labels, groups, feature_df, G=2)
    for key in ["outcome_diversity", "at_risk_concentration", "high_risk_group_rate", "outcome_balance"]:
        assert key in metrics
        assert np.isfinite(metrics[key])


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
    random_dist = significance.random_baseline_distribution(X, labels, feature_df, G=2, n_runs=3)
    metrics = pd.DataFrame(
        [
            {"strategy": "mode_a", **group_eval.evaluate_all(X, labels, [[0, 1], [2, 3]], feature_df, G=2)},
            {"strategy": "mode_b", **group_eval.evaluate_all(X, labels, [[0, 2], [1, 3]], feature_df, G=2)},
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
