# Reproducibility guide

There are two independent reproduction paths. Choose the first if you want to
verify the manuscript's numbers and figures; choose the second if you want to
rebuild the analysis from the official CITES data.

## Path A — verify the published aggregates (no data download)

This path uses only files committed to the repository.

```bash
python -m pip install -e ".[dev]"
rst-audit reproduce
```

What happens:

1. `figures/figA0_flagship.{pdf,png}` … `figures/figA5_country_coverage.{pdf,png}`
   are redrawn from `results/tables/` and `results/figure_data/`.
2. Each of the 66 expectations in `results/checks/expected.json` is recomputed
   from the same tables and compared at tolerance `5e-4`.
3. `results/checks/checks.csv` and `checks.json` are rewritten.
4. Any mismatch exits with status 1 and prints the failing check name, the
   expected value and the observed value.

No network access, no DuckDB file, no raw records. This is the path used by CI
and by reviewers who want to confirm that the shipped tables still support the
sentences in the paper.

Individual steps:

```bash
rst-audit figures     # redraw figures only
rst-audit checks      # numeric checks only
rst-audit checks --no-strict   # report failures but exit 0 (diagnostics)
```

`--freeze` regenerates `expected.json` from the current tables. Do not use it
to make a failing check pass; use it only after a deliberate manuscript
revision, and record the change.

## Path B — rebuild from the official CITES export

### 1. Environment

Python ≥ 3.10. The pinned lower bounds are in `requirements.txt`; the exact
environment used for the manuscript is described in `environment.yml`.

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
python -m pip install -e ".[dev]"
```

### 2. Data

Follow `data/README.md`:

```bash
python scripts/prepare_cites_db.py \
  --csv /path/to/cites_trade_2026_1.csv \
  --out data/cites.duckdb
```

The loader checks the header, normalizes `App.`/`Appendix` and
`Reporter type`/`Reporter.type`, streams the CSV in 500,000-row chunks, and
creates indexes on year and unit. It never silently drops a column.

### 3. Run the chain

```bash
CITES_DB=data/cites.duckdb rst-audit pipeline
```

Stages and outputs:

| Stage | Output | Notes |
|---|---|---|
| `panel` | `data/panel.parquet` | exporter × taxon × year; 301,914 rows at the frozen snapshot |
| `labels` | `data/labels.parquet` | 421,354 rows; 4,998 one-year positives; 11,759 three-year positives |
| `features` | `data/feat_T2004.parquet` … `feat_T2023.parquet` | 20 label years; 60 strict features |
| `models` | `output/scores/B__no_leak_strict__T{year}__s{seed}.parquet` | 5 years × 3 seeds |
| `audit` | `output/audit/` and `results/figure_data/` | cases, flags, window comparison |
| `sensitivity` | `output/audit/` tables | threshold, model family, import side |
| `figures` | `figures/*.pdf`, `figures/*.png` | five figures |
| `checks` | `results/checks/` | 66 checks |

Useful flags:

```bash
rst-audit pipeline --skip-models        # data stages only; reuse existing scores
rst-audit pipeline --skip-sensitivity   # skip the model-family reruns
rst-audit pipeline --seeds 0            # one seed for a quick pass
rst-audit pipeline --force              # rebuild every stage
```

Every stage is idempotent: if the expected artefact exists, the stage logs
`exists, reuse` and returns. This makes interrupted runs resumable.

### 4. Compare with the frozen results

The manuscript's panels, labels and scores were produced with the same code.
After a rebuild, compare:

```bash
python - <<'PY'
import pandas as pd
for name in ["panel", "labels"]:
    d = pd.read_parquet(f"data/{name}.parquet")
    print(name, len(d))
PY
```

Expected at database version 2026.1:

- panel: 301,914 rows;
- labels: 421,354 rows, `y_B.sum() == 4998`, `y_B3.sum() == 11759`.

Then run `rst-audit audit`, `rst-audit sensitivity`, `rst-audit figures` and
`rst-audit checks`. A later CITES version may change counts; in that case the
checks will fail until `expected.json` is deliberately refrozen, which is the
intended behaviour.

### 5. Validation performed for this release

The ported pipeline was checked against the frozen upstream artefacts:

| Object | Result |
|---|---|
| Labels | 421,354 rows; 4,998 one-year positives; 11,759 three-year positives |
| Core label columns | `v_cnt`, `m5`, `si_c`, `si3_c`, `si_g_u`, `y_B`, `y_B3`, `active5` equal to upstream at precision 1.0 |
| Figures | all five regenerated and visually inspected; numbers match the frozen tables |
| Checks | 66/66 pass |
| Tests | all unit, smoke, anonymity and data-hygiene tests pass |

## Environment variables

| Variable | Default | Purpose |
|---|---|---|
| `CITES_DB` | `data/cites.duckdb` | DuckDB file containing table `trade` |
| `RST_DATA_DIR` | `data/` | panel, labels, feature files |
| `RST_OUT_DIR` | `output/` | scores, audit intermediates, logs |
| `RST_RES_DIR` | `results/` | shipped tables, figure data, checks |
| `RST_FIG_DIR` | `figures/` | figure output |
| `RST_LOG_DIR` | `output/logs/` | stage logs |

The defaults write derived data into gitignored directories, so a local run
does not dirty the repository.

## Expected cost

On a four-core laptop:

- `reproduce`: seconds to a minute (figure rendering dominates);
- `panel` + `labels`: a few minutes, dominated by the DuckDB scan;
- `features`: about 30 seconds per label year, so roughly 10 minutes for 20
  years;
- `models`: a few minutes for 5 years × 3 seeds on the 60-feature matrix;
- `audit` + `sensitivity`: seconds to a few minutes, depending on whether model
  families are refit;
- figures + checks: seconds.

The full chain is therefore on the order of tens of minutes, not hours. Disk
use is dominated by the feature files (20 × ~3 MB) and score files (15 × small).

## Seeds and determinism

- Main audit: seed 0.
- Seed checks: seeds 1 and 2.
- LightGBM is deterministic given the same data, features, and seed on the same
  platform. Small numerical differences across platforms or library versions
  can change the ordering near the top-100 cutoff. The checks tolerate `5e-4`;
  a changed candidate list is a substantive difference and should be
  investigated, not silently accepted.
- Model-family checks (XGBoost, random forest) are reported as ranges, not as
  exact reproductions. Random forests are thread invariant; XGBoost's
  histogram method is not — the same data and seed give different trees under
  different thread counts (on a 20-core machine, 4 versus 20 threads changes
  individual scores by up to 0.29). The thread count is therefore part of the
  model definition: `config.XGB_THREADS` is pinned to 20, the setting used for
  the published table.

## A note on the exporter code `NA`

`NA` is Namibia's ISO code, not a missing value. The panel keeps exporter codes
as strings, and the trigger split is computed in memory from that panel, so
Namibian units keep their identity through every merge. A pipeline that sends
the pool through a CSV round-trip and lets pandas convert `NA` to missing loses
the `T−1` sharp-increase lookup for those units: the count of units that fire
only at *T−2* then becomes 881 instead of 879. Both values are 28.4% of 3,098
and no other number in the package changes.

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `CITES database not found` | `CITES_DB` not set or file absent | run `scripts/prepare_cites_db.py` and set `CITES_DB` |
| `missing required column` | CSV header variant not recognised | inspect the header; add an alias in `scripts/prepare_cites_db.py` |
| `empty candidate universe at T` | panel truncated or wrong year range | check the export years and `config.YEAR_MIN/YEAR_MAX` |
| checks fail after a data update | CITES version changed | rebuild, inspect the drift, refreeze only with a manuscript revision |
| figures missing | `reproduce` not run | `rst-audit figures` |
| `ModuleNotFoundError: rst_audit` | package not installed | `python -m pip install -e .` or set `PYTHONPATH=src` |

## What is intentionally not reproducible here

- The raw CITES records are not redistributed. Anyone can obtain them from the
  official source, but the repository cannot guarantee that a future CITES
  release is byte-identical to version 2026.1.
- Importer-reported volumes are a separate reporting stream. They are used for
  the data-boundary analysis only and are not a substitute for export records.
- Post-trigger review outcomes, non-detriment findings, and compliance
  responses are outside the database and outside the audit.
