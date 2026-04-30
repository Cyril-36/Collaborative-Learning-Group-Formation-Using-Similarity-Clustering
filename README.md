# CollabLearn — Collaborative Learning Group Formation Using Similarity Clustering

> A robust, stability-validated unsupervised learning pipeline for forming explainable collaborative learning groups from Virtual Learning Environment (VLE) data.

[![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Demo-red?logo=streamlit)](https://streamlit.io/)
[![License: CC-BY](https://img.shields.io/badge/Dataset-CC--BY-green)](https://analyse.kmi.open.ac.uk/open_dataset)
[![Course](https://img.shields.io/badge/Course-INT--396-purple)](https://github.com/Cyril-36/INT-396)

---

## Overview

This repository contains the **INT-396 Unsupervised Learning** project — a complete end-to-end system that transforms raw Open University Learning Analytics Dataset (OULAD) tables into validated, explainable collaborative learning group assignments.

The validated study uses the **OULAD AAA 2014J cohort**, yielding a final analysis set of **344 learners** represented through **35 engineered features** spanning demographics, VLE engagement, collaboration activity, and assessment performance.

Unlike single-algorithm clustering approaches, **CollabLearn** evaluates 12 reducer–clusterer configurations in parallel, subjects each to a bootstrap stability filter, and only uses configurations that survive the ARI threshold for downstream group formation. The core contribution is methodological: the system verifies whether clustering is *reliably stable* before trusting it for educational decisions.

---

## Key Result

The winning configuration on OULAD AAA 2014J was:

**🏆 C05 — UMAP + K-Means, k = 3**

After canonical cluster remapping, three learner profiles were identified:

| Cluster | Size | Profile |
|:---:|---:|:---|
| 0 | 128 | **High-engagement** — high active days, high VLE clicks, strong `oucontent` activity |
| 1 | 159 | **High-performing** — more assessments submitted, early first clicks, sustained last clicks |
| 2 | 57  | **At-risk / low-engagement** — fewer submissions, greater lateness, weaker engagement |

For **heterogeneous group formation**, Mode B outperformed the random baseline:

| Metric | Mode B | Random Baseline | p-value |
|:---|:---:|:---:|:---:|
| Complementarity | 0.666 | 0.602 | **0.010** |
| Cluster Coverage | 0.888 | 0.802 | **0.010** |
| At-risk Concentration | 0.270 | 0.270 | — |

---

## Method

### 1 · Feature Engineering

A **35-feature learner profile** is built from four OULAD tables, covering:

- **Demographics** — age band, IMD band, highest education, disability, gender, region
- **Engagement** — total VLE clicks, active days, engagement span, weekend ratio, first/last click day, per-activity clicks (`forumng`, `oucontent`, `resource`, `quiz`, `subpage`, `externalquiz`, `oucollaborate`, `homepage`), collaborative active days
- **Performance** — mean score, weighted score, assessment count, mean submission lateness, score slope, non-submission count

All features are imputed, ordinal-encoded, and scaled to zero-mean unit-variance via `StandardScaler`.

---

### 2 · Multi-Configuration Clustering

The pipeline evaluates **12 configurations** by crossing:

| Representation | Clusterers |
|:---|:---|
| PCA | K-Means |
| UMAP | Gaussian Mixture Models (GMM) |
| Identity (no reduction) | Agglomerative Clustering |
| | HDBSCAN |

Each configuration is scored on:

- Silhouette Index
- Davies–Bouldin Index
- Calinski–Harabasz Index
- Noise ratio / DBCV (density methods)

> **Note on HDBSCAN:** Several HDBSCAN configurations on this cohort collapsed almost entirely into noise or failed the ARI threshold. Results are reported transparently but HDBSCAN was not selected for group formation.

---

### 3 · Bootstrap Stability Validation

To prevent over-trusting a single visually appealing partition:

1. Draw **B = 30** subsamples at 80% of learners
2. Re-run preprocessing → reduction → clustering for each subsample
3. Compute pairwise **Adjusted Rand Index** across all 435 pairs
4. Record mean, std, min, max ARI
5. **Reject** configurations with mean ARI < 0.40 — regardless of internal scores

The composite winner is chosen by ranking surviving configurations across all internal validity metrics plus bootstrap ARI.

---

## Group Formation Modes

Once a stable configuration is selected, groups are formed with **default size G = 4 (±1)** — yielding **86 groups** over 344 learners in the validated run.

### Mode A — Homogeneous (Pace-Matched)

- Groups are formed **within clusters**
- Learners are ordered by nearest-neighbour distance in the UMAP representation space
- Produces pace-matched groups with minimal intra-group distance (0.550 vs 2.913 random)
- Best suited for self-directed or competency-matched activities

### Mode B — Heterogeneous (Scaffolded)

- Learners are drawn **round-robin across clusters**, ensuring ≥ 2 distinct profiles per group
- Final seat filled from the largest remaining cluster
- Produces scaffolded groups with higher **complementarity** and **cluster coverage**
- Best suited for peer-teaching, project teams, or diverse collaborative tasks

A **constraint subsystem** enforces group-size and complementarity rules with an optional greedy swap refinement pass to repair violations.

---

## Evaluation

Groups are evaluated against **100 random partitions** and a **demographic-stratified baseline** using 10 group-level metrics:

> Intra-group distance · Inter-group variance · Complementarity · Engagement balance · Demographic fairness · Cluster coverage · Outcome diversity · At-risk concentration · High-risk group rate · Outcome balance

Statistical significance is computed via a **randomization test** (one-sided, 100 partitions).

---

## Repository Structure

```text
INT-396/
├── src/
│   ├── pipeline.py           # End-to-end orchestration + CLI entry point
│   ├── ingest.py             # Multi-table OULAD ingestion & integrity checks
│   ├── features.py           # 35-feature learner profile engineering
│   ├── preprocess.py         # Imputation, encoding, scaling
│   ├── multiconfig.py        # 12-configuration clustering evaluation loop
│   ├── stability.py          # B=30 bootstrap ARI stability validation
│   ├── selector.py           # Winner selection with ARI thresholding
│   ├── clusterinterpret.py   # Canonical cluster remapping & profile characterisation
│   ├── groupformer.py        # Dual-mode group formation engine
│   ├── constraints.py        # Constraint checking & greedy swap refinement
│   ├── groupeval.py          # Group-level metric computation
│   ├── significance.py       # Randomization significance testing
│   └── output.py             # CSV, figure & report export
│
├── demo/
│   └── app.py                # Streamlit instructor dashboard
│
├── notebooks/                # Exploratory analysis & development runs
│
├── results/
│   ├── figures/              # 10 polished result figures
│   └── tables/               # Exported metric tables
│
├── scripts/
│   └── regen_figures.py      # Figure regeneration from cached artifacts
│
├── tests/
│   └── test_smoke.py         # Basic pipeline smoke tests
│
├── .streamlit/               # Streamlit configuration
├── assets/                   # Static assets
├── data/                     # OULAD CSVs (not committed — see Dataset section)
├── requirements.txt
└── README.md
```

---

## Installation

**Requirements:** Python 3.11+

```bash
# 1. Clone the repository
git clone https://github.com/Cyril-36/INT-396.git
cd INT-396

# 2. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

Key dependencies include `scikit-learn`, `umap-learn`, `pandas`, `numpy`, `plotly`, `streamlit`, and `matplotlib`.

---

## How To Run

### Streamlit Dashboard (Recommended)

```bash
streamlit run demo/app.py
```

The dashboard lets instructors inspect:
- Per-configuration validity scores and stability plots
- UMAP embedding with cluster centroids
- Learner profile summaries per cluster
- Group assignments and metric comparisons vs baselines

### Full Pipeline from Raw Data

```bash
python -m src.pipeline \
    --dataset oulad_aaa_2014j \
    --group_size 4 \
    --mode both \
    --output_dir results/run_aaa_2014j
```

This will ingest → engineer features → evaluate 12 configurations → run bootstrap stability → select the winner → form Mode A & B groups → compute metrics → export CSVs and figures to `results/`.

### Tests

```bash
pytest -q
```

### Regenerate Figures

```bash
python scripts/regen_figures.py
```

---

## Dataset

This project uses the **Open University Learning Analytics Dataset (OULAD)**, AAA 2014J presentation, released under a **CC-BY licence**.

> The raw dataset is **not bundled** with this repository. Download it from [analyse.kmi.open.ac.uk](https://analyse.kmi.open.ac.uk/open_dataset) and place the following CSVs in `data/`:

```
assessments.csv
courses.csv
studentAssessment.csv
studentInfo.csv
studentRegistration.csv
studentVle.csv
vle.csv
```

---

## Why This Matters

Most digital learning platforms still rely on one of four simplistic group-allocation strategies: random assignment, manual instructor sorting, single-attribute sorting by grade, or a single unvalidated clustering result. Each ignores the richness of learner behaviour logs and provides no robustness guarantees.

**CollabLearn addresses this gap by:**
- Building rich, multi-dimensional learner profiles from real VLE traces
- Evaluating a full matrix of reducer–clusterer combinations in parallel
- Using bootstrap ARI stability as a hard filter before any group formation
- Turning the result into two explainable group modes with documented trade-offs

The project is **academically honest**: it demonstrates structural improvements in group-quality metrics, but deliberately does not claim causal improvement in learning outcomes without intervention-based validation.

---

## Extensibility

The framework is designed to be adapted beyond OULAD:

- **Pluggable feature engineering** — demographic, engagement, and performance column lists are configurable without touching core logic
- **Configurable configuration matrix** — number and types of reducers/clusterers, k-ranges, UMAP parameters, and HDBSCAN hyperparameters are all parameterised
- **Adjustable robustness thresholds** — ARI threshold, resample fraction, and B are tuneable
- **Flexible grouping** — target group size, mode selection, and complementarity thresholds can be set per course context
- **Scalable evaluation** — number of random baseline partitions and active metric set are configurable

---

## Licence

- **Code:** Open-source Python stack (scikit-learn, UMAP-learn, pandas, numpy); see each library's respective licence.
- **Dataset:** OULAD is released under [CC-BY](https://creativecommons.org/licenses/by/4.0/) by The Open University. It is not redistributed in this repository.

---

*Developed for INT-396 Unsupervised Learning · Lovely Professional University*
