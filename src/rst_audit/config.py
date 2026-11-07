"""Paths and frozen analysis parameters.

The repository ships only derived aggregates and figures.  Raw CITES records
are not redistributed; download them from https://trade.cites.org/ and point
``CITES_DB`` at a DuckDB file containing a table named ``trade``.
"""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _path(env: str, default: Path) -> Path:
    value = os.environ.get(env)
    return Path(value).expanduser().resolve() if value else default


DATA_DIR = _path("RST_DATA_DIR", ROOT / "data")
OUT_DIR = _path("RST_OUT_DIR", ROOT / "output")
FIG_DIR = _path("RST_FIG_DIR", ROOT / "figures")
RES_DIR = _path("RST_RES_DIR", ROOT / "results")
LOG_DIR = _path("RST_LOG_DIR", OUT_DIR / "logs")

CITES_DB = _path("CITES_DB", DATA_DIR / "cites.duckdb")
PANEL = DATA_DIR / "panel.parquet"
LABELS = DATA_DIR / "labels.parquet"
SCORE_DIR = OUT_DIR / "scores"
AUDIT_DIR = OUT_DIR / "audit"

# Analysis window.  2024 is context only; the 2025 annual file is incomplete
# at the study snapshot and is excluded.
YEAR_MIN = 2000
YEAR_MAX = 2024
TRAIN_START = 2005
LABEL_YEARS = list(range(TRAIN_START, YEAR_MAX + 1))
TEST_YEARS = [2018, 2019, 2020, 2021, 2022]
EXT_YEARS = [2016, 2017, 2023]
ALL_YEARS = [2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023]
SEEDS = [0, 1, 2]
N_THREADS = 4

# Sharp-increase rule, fixed before modelling.
SI_MULT = 4.0
SI_MIN_MEAN = 100.0
GLOBAL_SHARE = 0.10

# Audit definitions.
BASE_LEN = 5              # five-year reference window
MIN_BASE_POSITIVE = 2     # "multi-year record support"
MIN_RECORD_RATIO = 0.5    # trigger-year records / median of support years
TOP_K = 100               # fixed candidate list
DIVERGENCE_TOP = 50       # pooled and unfiltered divergence sets
PER_YEAR_TOP = 10         # per-year divergence set

SOURCES = ["W", "C", "F", "R", "D", "A", "U", "I", "O"]
SOURCE_KEYS = SOURCES + ["UNK"]
LIVE_TERMS = ("live", "eggs (live)")

# Model protocol.
LABEL = "B"
FEATURE_SET = "no_leak_strict"
N_ROUNDS = 300
# XGBoost's histogram method builds different trees for different thread
# counts, so the thread count is part of the model definition.  It is pinned to
# the value used for the published model-family table; random forests are
# thread invariant.
XGB_THREADS = 20
LGB_PARAMS = dict(
    objective="binary",
    metric="binary_logloss",
    learning_rate=0.05,
    num_leaves=31,
    min_data_in_leaf=50,
    feature_fraction=0.8,
    bagging_fraction=0.8,
    bagging_freq=1,
    verbose=-1,
)


def ensure_dirs() -> None:
    for d in (DATA_DIR, OUT_DIR, FIG_DIR, RES_DIR, LOG_DIR, SCORE_DIR, AUDIT_DIR):
        d.mkdir(parents=True, exist_ok=True)


def score_path(label: str, feature_set: str, year: int, seed: int) -> Path:
    return SCORE_DIR / f"{label}__{feature_set}__T{year}__s{seed}.parquet"
