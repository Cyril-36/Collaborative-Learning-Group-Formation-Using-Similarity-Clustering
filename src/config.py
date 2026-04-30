"""Project configuration and default hyperparameters."""

from __future__ import annotations

import os
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = ROOT / "data" / "raw"
DATA_ANONYMISED = ROOT / "anonymisedData"
DATA_PROCESSED = ROOT / "data" / "processed"
RESULTS = ROOT / "results"
FIGURES = RESULTS / "figures"
TABLES = RESULTS / "tables"
DEMO_CACHE = ROOT / "demo" / "demo_cache"

PRESENTATION = ("AAA", "2014J")
PRESENTATION_LENGTH = 270
MIN_ENGAGEMENT_DAYS = 30

USE_REGION_ONEHOT = False
ACTIVITY_TYPES_TOP_N = 9
DEFAULT_ACTIVITY_TYPES = [
    "oucontent",
    "subpage",
    "homepage",
    "resource",
    "url",
    "quiz",
    "forumng",
    "oucollaborate",
    "ouwiki",
]

PCA_N_COMPONENTS = 10
UMAP_N_COMPONENTS = 10
UMAP_2D_COMPONENTS = 2
UMAP_N_NEIGHBORS = 15
UMAP_MIN_DIST = 0.1

K_SWEEP = list(range(3, 11))
HDBSCAN_MIN_CLUSTER_SIZE = 30
HDBSCAN_MIN_SAMPLES = 10

BOOTSTRAP_B = int(os.getenv("INT396_BOOTSTRAP_B", "30"))
BOOTSTRAP_FRAC = float(os.getenv("INT396_BOOTSTRAP_FRAC", "0.80"))
N_JOBS = int(os.getenv("INT396_N_JOBS", "-1"))

GROUP_SIZE = 4
SIZE_TOLERANCE = 1

FAIRNESS_ATTR = "imd_band_ord"
FAIRNESS_TVD_MAX = 0.20
COMPLEMENTARITY_PERCENTILE = 60
ENGAGEMENT_BALANCE_SIGMA = 1.0
MAX_SWAP_ITERS = 500

MIN_STABILITY = 0.40
SEED = 42


def ensure_dirs() -> None:
    """Create runtime output directories."""
    for path in [DATA_RAW, DATA_PROCESSED, DEMO_CACHE, RESULTS, FIGURES, TABLES]:
        path.mkdir(parents=True, exist_ok=True)


def raw_data_dir() -> Path:
    """Return the directory containing OULAD CSV files.

    The documented project uses data/raw, but this workspace ships the dataset in
    anonymisedData. Prefer data/raw if populated, otherwise use the bundled path.
    """
    expected = [
        "studentInfo.csv",
        "studentVle.csv",
        "studentAssessment.csv",
        "studentRegistration.csv",
        "vle.csv",
        "assessments.csv",
    ]
    if all((DATA_RAW / name).exists() for name in expected):
        return DATA_RAW
    if all((DATA_ANONYMISED / name).exists() for name in expected):
        return DATA_ANONYMISED
    missing = [name for name in expected if not (DATA_RAW / name).exists()]
    raise FileNotFoundError(
        "Missing OULAD CSVs in data/raw. Also checked anonymisedData. "
        f"Missing from data/raw: {missing}"
    )


def parse_presentation(value: str | None) -> tuple[str, str]:
    if not value:
        return PRESENTATION
    parts = value.replace("-", "_").split("_")
    if len(parts) != 2:
        raise ValueError("Presentation must look like AAA_2014J")
    return parts[0], parts[1]
