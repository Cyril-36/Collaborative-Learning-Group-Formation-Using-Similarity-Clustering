"""Full OULAD snapshot regression test.

This test is gated behind environment variables because it runs the entire
pipeline against the real OULAD case study (~minutes, requires the OULAD
data to be available locally).

Modes
-----
Run in compare mode (default when enabled)::

    INT396_RUN_OULAD_SNAPSHOT=1 .venv/bin/python -m pytest tests/test_oulad_snapshot.py

Run in update mode (regenerates the committed reference cache after a
deliberate refactor)::

    INT396_UPDATE_OULAD_SNAPSHOT=1 .venv/bin/python -m pytest tests/test_oulad_snapshot.py

Update mode overwrites the reference parquet/json files in
``pipeline.DEMO_CACHE`` with the freshly produced artifacts and skips the
strict comparison. Commit the diff to the reference cache afterwards.

Compare mode runs the pipeline into a temp cache and compares against the
committed reference. Numeric metric values are compared with ``atol=1e-6``.
The ``interpretive_label`` column on ``cluster_summary.parquet`` is treated
as a display string (must be present and non-empty) rather than checked for
exact-string equality, because it is derived from schema role columns and
will legitimately evolve as label heuristics improve.
"""

from __future__ import annotations

import json
import math
import os
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src import pipeline


RUN_FLAG = "INT396_RUN_OULAD_SNAPSHOT"
UPDATE_FLAG = "INT396_UPDATE_OULAD_SNAPSHOT"


def _enabled() -> bool:
    return os.getenv(RUN_FLAG) == "1" or os.getenv(UPDATE_FLAG) == "1"


pytestmark = pytest.mark.skipif(
    not _enabled(),
    reason=(
        "Full OULAD snapshot is expensive; set INT396_RUN_OULAD_SNAPSHOT=1 to "
        "compare against the committed reference cache, or "
        "INT396_UPDATE_OULAD_SNAPSHOT=1 to regenerate it."
    ),
)


# Columns that are display-only / derived strings and may legitimately drift
# across refactors of label heuristics. We require them to exist and be
# non-empty in the actual output, but do not require exact-string equality.
_DERIVED_STRING_COLS: dict[str, set[str]] = {
    "cluster_summary.parquet": {"interpretive_label"},
}

_PARQUET_FILES = (
    "group_metrics.parquet",
    "cluster_summary.parquet",
    "cluster_labels.parquet",
    "groups_mode_a.parquet",
    "groups_mode_b.parquet",
)

_JSON_FILES = ("winner.json",)


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _assert_frame_close(
    actual: pd.DataFrame,
    expected: pd.DataFrame,
    *,
    ignore_string_cols: set[str] | None = None,
) -> None:
    ignore_string_cols = ignore_string_cols or set()
    actual = actual.reset_index(drop=True)
    expected = expected.reset_index(drop=True)
    assert list(actual.columns) == list(expected.columns), (
        f"column mismatch: {list(actual.columns)} vs {list(expected.columns)}"
    )
    assert actual.shape == expected.shape, (
        f"shape mismatch: {actual.shape} vs {expected.shape}"
    )
    for col in actual.columns:
        if col in ignore_string_cols:
            # Display-only column: just require presence and non-empty values.
            values = actual[col].astype(str).tolist()
            assert all(v and v != "nan" for v in values), (
                f"derived display column {col!r} contains empty/NaN entries: {values}"
            )
            continue
        if pd.api.types.is_numeric_dtype(actual[col]) and pd.api.types.is_numeric_dtype(expected[col]):
            np.testing.assert_allclose(
                actual[col].to_numpy(),
                expected[col].to_numpy(),
                rtol=0,
                atol=1e-6,
                err_msg=f"numeric drift in column {col!r}",
            )
        else:
            assert actual[col].astype(str).tolist() == expected[col].astype(str).tolist(), (
                f"non-numeric mismatch in column {col!r}"
            )


def _update_reference(actual_dir: Path, reference_dir: Path) -> None:
    reference_dir.mkdir(parents=True, exist_ok=True)
    for name in _PARQUET_FILES + _JSON_FILES:
        src = actual_dir / name
        if not src.exists():
            raise AssertionError(
                f"expected pipeline to produce {src}, but it was not written"
            )
        shutil.copy2(src, reference_dir / name)


def test_oulad_full_snapshot_regression(tmp_path):
    cache_dir = tmp_path / "oulad-cache"
    pipeline.run(
        presentation="AAA_2014J",
        cache_dir=cache_dir,
        bootstrap_b=30,
        random_baselines=100,
        n_jobs=1,
    )

    reference = pipeline.DEMO_CACHE

    if os.getenv(UPDATE_FLAG) == "1":
        _update_reference(cache_dir, reference)
        pytest.skip(
            f"Snapshot reference at {reference} regenerated. "
            f"Inspect the diff and commit if the change is intentional."
        )

    # Compare mode.
    actual_winner = _json(cache_dir / "winner.json")
    expected_winner = _json(reference / "winner.json")
    for key in ["config_id", "reducer", "clusterer", "k"]:
        assert actual_winner[key] == expected_winner[key], (
            f"winner.{key} drift: {actual_winner[key]!r} vs {expected_winner[key]!r}"
        )
    for key in ["silhouette", "bootstrap_ari_mean"]:
        assert math.isclose(
            actual_winner[key], expected_winner[key], rel_tol=0.0, abs_tol=1e-6
        ), f"winner.{key} numeric drift: {actual_winner[key]} vs {expected_winner[key]}"

    for name in _PARQUET_FILES:
        _assert_frame_close(
            pd.read_parquet(cache_dir / name),
            pd.read_parquet(reference / name),
            ignore_string_cols=_DERIVED_STRING_COLS.get(name),
        )
