# Data access

The repository does **not** redistribute the CITES trade records. The raw
records are public and must be obtained from the official source; this page
records the exact version, export settings and table schema used by the
manuscript so that the pipeline can be rebuilt from scratch.

## Official source

- **Database:** CITES Trade Database
- **URL:** <https://trade.cites.org/>
- **Version used:** 2026.1
- **Terms:** the database is published by the CITES Secretariat for
  non-commercial use. Check the terms of use on the download page before
  redistributing any extract; this repository deliberately does not.

## Export settings

On <https://trade.cites.org/>, choose the bulk download ("download all data")
and export a single CSV file with:

| Setting | Value |
|---|---|
| Years | 2000–2024 |
| Taxa | all |
| Reporters | all countries |
| Trade flows | exports **and** imports |
| Terms / purposes / sources | all |
| Format | comma-separated values |

The analysis filters the table in SQL; no filtering is needed at download time.
The 2025 annual file was incomplete at the study snapshot and is excluded. If
you download a later database version, rerun the full pipeline and expect the
counts to shift slightly.

## Expected table schema

`scripts/prepare_cites_db.py` normalizes the CSV into a DuckDB table named
`trade` with these columns:

| Column | Type | Notes |
|---|---|---|
| `Year` | INTEGER | 2000–2024 |
| `Exporter` | VARCHAR | ISO-2 country code |
| `Importer` | VARCHAR | ISO-2 country code |
| `Origin` | VARCHAR | ISO-2 code, may be NULL |
| `Taxon` | VARCHAR | species or higher taxon |
| `Genus`, `Family`, `Order`, `Class` | VARCHAR | taxonomy as reported |
| `Appendix` | VARCHAR | I, II, III or N |
| `Reporter.type` | VARCHAR | `E` = exporter-reported, `I` = importer-reported |
| `Purpose` | VARCHAR | e.g. T, P, H, S |
| `Source` | VARCHAR | e.g. W, C, F, R, D, A, U, I, O |
| `Term` | VARCHAR | e.g. live, specimens, derivatives |
| `Unit` | VARCHAR | e.g. `Number of specimens` |
| `Quantity` | BIGINT | as reported |

The loader accepts the common header variants (`App.` → `Appendix`,
`Reporter type` → `Reporter.type`, case-insensitive matching) and fails loudly
if a required column is missing. It never silently drops a column.

## Build the DuckDB file

```bash
python scripts/prepare_cites_db.py \
  --csv /path/to/cites_trade.csv \
  --out data/cites.duckdb
```

Then point the pipeline at it:

```bash
CITES_DB=data/cites.duckdb rst-audit pipeline
```

The full chain needs several GB of disk space for intermediate parquet files
(panel, labels, 20 feature-year files, scores). `data/`, `output/` and
`*.parquet` are gitignored; nothing derived from the raw records is committed.

## What is shipped instead

The repository ships only **aggregates** that cannot be traced back to
individual shipment records:

- `results/tables/*.csv` — case-level and list-level summaries (country codes,
  taxon names and counts; no shipment rows);
- `results/figure_data/*.csv` — per-case yearly volumes for the 34 audited
  cases, the six timelines shown in Figure 4, and the trigger-year flag matrix;
- `results/checks/expected.json` — 66 headline quantities;
- `results/paper_numbers.json` — the manuscript's headline numbers in one file.

These are sufficient to redraw every figure and re-run every numeric check
without downloading the database. They are not sufficient to rebuild the
panel, which is why the full pipeline exists.

## Version and fingerprint

The manuscript used CITES Trade Database **2026.1**. The derived panel has
301,914 unit-years, 32,062 exporter–taxon pairs in the 2018–2022 test period,
and 421,354 label rows. `rst-audit panel` prints the row count, volume totals
and a SHA-1 fingerprint of the generated panel; record the fingerprint in your
own run if you compare results across database versions.

## Licence note

CITES data: subject to the CITES Trade Database terms of use.
Code: MIT. Figures and derived aggregates: CC BY 4.0.
