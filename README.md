# rst-window-audit

Reproducibility package for an audit of the **2026 CITES Review of Significant
Trade (RST) three-year sharp-increase window**.

The study asks a narrow question: when the sharp-increase test is applied only
to the focus year, which record-supported surges fall outside the detection
layer? It then compares, on a fixed candidate list, extending the look-back
window with lowering the threshold. The package contains the full analysis
pipeline (panel construction → labels → features → rolling-origin ranking →
audit → sensitivity → figures → numeric checks), the frozen aggregate tables,
the five manuscript figures, and machine checks for every headline number.

The manuscript is under review. Author information is withheld for
double-blind review. Raw CITES trade records are **not** redistributed; the
official download link and the exact build steps are in [`data/README.md`](data/README.md).

---

## What is in the box

| Path | Contents |
|---|---|
| [`src/rst_audit/`](src/rst_audit) | Analysis package: panel, labels, features, model, audit, sensitivity, figures, checks, CLI |
| [`results/tables/`](results/tables) | Frozen aggregate tables behind the manuscript tables (16 files) |
| [`results/figure_data/`](results/figure_data) | Per-case series and matrices used to draw the figures |
| [`results/checks/`](results/checks) | 66 frozen expectations, plus the latest pass/fail record |
| [`figures/`](figures) | Five manuscript figures, PDF (vector) and 600 dpi PNG |
| [`tests/`](tests) | Unit tests, figure/check smoke tests, anonymity and data-hygiene scans |
| [`docs/`](docs) | Methods, data dictionary, reproducibility guide, paper-to-code map |
| [`scripts/`](scripts) | Helper to turn the official CITES bulk export into the DuckDB table expected by the pipeline |

## Quick start (no data download)

```bash
python -m pip install -e ".[dev]"
rst-audit reproduce
```

`reproduce` redraws all five figures from the shipped tables and re-runs the
numeric checks. It does **not** need the CITES database. A successful run ends
with `all N checks passed` and writes `results/checks/checks.csv`.

Run the test suite (no network, no raw data):

```bash
pytest
```

## Full pipeline (from the official CITES export)

1. Download the CITES Trade Database bulk export (version 2026.1) from
   <https://trade.cites.org/>. The download and licence notes are in
   [`data/README.md`](data/README.md).
2. Convert the export into the DuckDB table `trade`:

   ```bash
   python scripts/prepare_cites_db.py --csv /path/to/cites_trade.csv --out data/cites.duckdb
   ```

3. Run the chain:

   ```bash
   CITES_DB=data/cites.duckdb rst-audit pipeline
   ```

   Or run stages separately:

   ```bash
   rst-audit panel       # exporter × taxon × year panel (2000–2024)
   rst-audit labels      # sharp-increase states, one-year and three-year labels
   rst-audit features    # 60 rule-free features, one file per label year
   rst-audit models      # rolling-origin LightGBM, five test years, seeds 0–2
   rst-audit audit       # divergence set, case verification, fixed-list comparison
   rst-audit sensitivity # threshold, model family, import-side and lag checks
   rst-audit figures
   rst-audit checks
   ```

All stages are idempotent: existing artefacts are reused unless `--force` is
given. Paths are controlled by `RST_DATA_DIR`, `RST_OUT_DIR`, `RST_RES_DIR`,
`RST_FIG_DIR`, `RST_LOG_DIR`, and `CITES_DB`.

## The analysis in one page

**Unit and volume.** A unit is an exporter × taxon pair. The primary volume is
exporter-reported direct trade in `Number of specimens`, summed over records
whose `Origin` is empty or equal to the exporter (the standard export-side
convention). The panel covers 2000–2024 and contains 301,914 unit-years;
32,062 exporter–taxon pairs appear in the 2018–2022 test period.

**Sharp-increase rule.** A unit fires at year *t* when

```
V(t) > 4 × mean(V(t−5 … t−1))   and   mean(V(t−5 … t−1)) ≥ 100.
```

A global variant additionally requires the species total to pass the same test
and the unit to hold at least 10% of world exports. The one-year label uses
year *T*; the three-year label uses *T*, *T−1*, or *T−2*.

**Independent ranking.** To avoid auditing the rule with itself, the model is
trained on a feature set that excludes the rule and its near proxies: no
sharp-increase flags, no rule ratios, no five-year mean or five-year total, no
endangerment flag. Sixty features remain (trade history, network position,
trade form, taxonomy, record quality). A LightGBM classifier is fit with
rolling-origin evaluation: for each test year, the model sees only earlier
years and is scored on the current year. The score is used only to produce
candidates; the audit depends on the top of the ranking, not on overall
predictive performance.

**Divergence set.** Units ranked near the top by the model but not fired by the
one-year rule. Three selection rules are reported: pooled top 50, top 10 per
test year (50 in total), and unfiltered top 50.

**Case verification.** Each divergence case is recomputed from the raw panel.
A case passes internal verification when (i) its support window — the five
years before *T*, excluding trigger years — has at least two years with
positive trade records, and (ii) the record count in the most recent trigger
year is at least half the median record count of those support years. A case
with exactly one support year is placed on a watchlist; the verdict is a
statement about internal record consistency, not about biological
sustainability.

**Fixed-list comparison.** The same top-100 model candidates are held fixed,
and alternative rules (one/two/three-year window; 2×/3×/4×/5× threshold) are
applied to that list. This answers the policy question — how many cases enter a
fixed review budget — rather than the unconstrained question of how many units
fire anywhere in the panel.

## Headline results (all machine-checked)

- All 34 pooled divergence cases fire at *T−1*; 10 also at *T−2*; none at *T*.
  The blind spot is driven by surges that do not repeat in the focus year, not
  by units just below the threshold (near-threshold cases: 0 in all three
  selection rules).
- 18 of the 34 cases have multi-year record support and pass verification;
  16 have a single support year and form the watchlist.
- On the fixed top-100 list, the one-year rule flags 34.2 units per year. The
  three-year window flags 60.4 and adds 15.0 verified cases; the 3× threshold
  adds 6.2 and 2.6 verified cases. The window adds about **5.8×** more verified
  cases than the threshold change.
- The two-year window already delivers almost all of the list-level gain
  (14.8 verified cases); the third year mainly extends coverage of the full
  candidate universe, where 879 of 3,098 three-year positives (28.4%) fire only
  at *T−2* and are outside the top of the list.
- The blind spot concentrates in Guatemala and in plants, but is not exclusive
  to them: removing Guatemala leaves 30 of the 34 pooled cases; removing all
  exporters whose reporting footprint at least doubled in 2017 leaves 28.
- Importer-side records pair for 29 of 34 cases, but 28 of those 29 have no
  origin country and the median quantity ratio is 0.15. Importer records
  cannot verify the magnitude of a surge; this is a data boundary, not a
  retrieval failure.

## Design choices worth knowing

**Why exclude rule features?** A model that can see `V/M` or the five-year mean
can reproduce the rule. Divergences produced by such a model would mostly be
near-threshold duplicates, not independent evidence about temporal coverage.
The paper reports a full-feature contrast for transparency (26 cases, 4 of them
near-threshold), but the audit uses the rule-free ranking.

**Why a fixed list?** A rule can flag many units in the full universe and still
change nothing about a review team that can only examine 100 candidates per
year. Holding the model's top-100 fixed isolates the effect of the rule on the
list that would actually be reviewed.

**Why call it an audit?** The model is a candidate generator, not a causal
model. "Verified" means internally consistent records, not confirmed illegal or
unsustainable trade. The manuscript is explicit about this boundary and about
the limits of a single database snapshot.

## Reproducibility contract

`results/checks/expected.json` stores 66 quantities computed from the shipped
tables: case counts, trigger-year splits, list-level arms, concentration
shares, import matching, lag fields, and model-family ranges. `rst-audit
checks` recomputes them and fails with a non-zero exit status on any mismatch
(tolerance `5e-4`). The expected file is a contract, not a convenience: if a
table changes, the check fails until the manuscript number and the expectation
are reconciled deliberately.

The pipeline was validated against the frozen upstream artefacts used for the
manuscript:

- labels: 421,354 rows, 4,998 one-year positives, 11,759 three-year positives;
- `v_cnt`, `m5`, `si_c`, `si3_c`, `si_g_u`, `y_B`, `y_B3`, `active5` equal to
  the upstream labels at machine precision;
- every model matrix (`feat_T*.parquet`, 71 columns) equals the upstream file;
- the seed-0 LightGBM scores for all five test years reproduce the manuscript
  scores exactly (max |Δ| = 0, top-100 lists identical), so the fixed
  candidate list, the case table and every downstream number are regenerable;
- the model-family table reproduces with the XGBoost thread count pinned in
  `config.XGB_THREADS` (the histogram method builds different trees for
  different thread counts; random forests are thread invariant);
- `rst-audit reproduce` passes all 66 checks and the test suite passes.

## Data and code availability

- CITES Trade Database (2026.1): <https://trade.cites.org/>. Not redistributed
  here; see [`data/README.md`](data/README.md) for the exact export settings.
- Code: MIT licence. Figures: CC BY 4.0. Derived tables: CC BY 4.0.
- No author-identifying metadata is included in this repository. Commit
  history, if any, is scrubbed before public release.

## Citation

See [`CITATION.cff`](CITATION.cff). While the manuscript is under review,
please cite it as an anonymous preprint/repository and do not add author names.

## Layout

```
rst-window-audit/
├── configs/default.yaml        # frozen parameter snapshot
├── data/README.md              # how to obtain the official data
├── docs/
│   ├── methods.md              # definitions, formulas, audit protocol
│   ├── data_dictionary.md      # every shipped column
│   ├── reproducibility.md      # two reproduction paths and expected runtimes
│   └── paper_mapping.md        # manuscript figure/table → file → claim
├── figures/                    # figA0, figA2–figA5 (PDF + PNG)
├── results/
│   ├── tables/                 # 16 aggregate tables
│   ├── figure_data/            # 6 per-case / per-year tables
│   ├── checks/                 # expectations and check record
│   ├── model_results.csv       # rolling-origin ranking metrics per year and seed
│   └── paper_numbers.json      # headline numbers as a single JSON
├── scripts/prepare_cites_db.py # official CSV → DuckDB `trade` table
├── src/rst_audit/              # the analysis package
└── tests/                      # unit, smoke, anonymity, data-hygiene tests
```

## Licence

Code: MIT (see [`LICENSE`](LICENSE)). Figures and derived tables: CC BY 4.0.
The CITES data remain subject to the terms of use of the CITES Trade Database.
