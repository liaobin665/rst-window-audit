# Data dictionary

This page documents every file shape and column shipped in `results/`, and the
intermediate files produced by the pipeline. Raw CITES records are not part of
the repository; their schema is documented in `data/README.md`.

---

## 1. Panel (`data/panel.parquet`, rebuilt)

One row per exporter × taxon × year. 64 columns, 301,914 rows at database
version 2026.1.

### Keys and primary volume

| Column | Type | Meaning |
|---|---|---|
| `year` | int | 2000–2024 |
| `exporter` | str | ISO-2 exporter code |
| `taxon` | str | species or higher taxon, as reported |
| `v_cnt` | float | primary volume: count-unit, exporter-reported, direct exports |
| `records` | int | number of records behind `v_cnt` |

### Sensitivity volumes

| Column | Meaning |
|---|---|
| `v_wild` | volume with `Source = W` (wild) |
| `v_wlive` | wild volume with term `live` or `eggs (live)` |
| `v_allunit` | volume across all units (no count-unit filter) |
| `records_allunit` | records behind `v_allunit` |
| `v_live` | volume of live terms |
| `live_share` | `v_live / v_cnt` |
| `v_reexport` | volume where `Origin` is filled and differs from exporter |
| `top_importer_share` | largest single-importer share of the unit's volume |

### Source shares

`src_W`, `src_C`, `src_F`, `src_R`, `src_D`, `src_A`, `src_U`, `src_I`,
`src_O`, `src_UNK` — volume by CITES source code (W wild, C captive-bred,
F born in captivity, R ranched, D bred in captivity, A artificially
propagated, U unknown, I seized/confiscated, O pre-convention).

### Purpose shares

`p_T`, `p_P`, `p_H`, `p_M`, `p_S`, `p_Q`, `p_Z`, `p_B` — share of volume by
CITES purpose code (T commercial, P personal, H hunting trophy, M medical,
S scientific, Q circus, Z zoo, B breeding).

### Term and taxonomy

| Column | Meaning |
|---|---|
| `top_term` | most frequent term by volume |
| `top_term_share` | its share |
| `term_entropy` | Shannon entropy of term shares |
| `terms_n` | distinct terms |
| `importers_n` | distinct importers |
| `origin_n` | distinct origins |
| `appendix`, `cls`, `ord`, `family`, `genus` | taxonomy as reported (`cls`, `ord` avoid Python/SQL keyword clashes) |

### Importer-reported volumes

Two families, both are separate reporting streams and are **not** substituted
for export records:

- `imp_*` — strict importer-reported volume (`Reporter.type = I`,
  `Origin = Exporter`);
- `impn_*` — relaxed importer-reported volume (`Origin` NULL or equal to
  exporter);
- suffixes are the same source codes plus `UNK`;
- `v_imp_total`, `v_impn_total` — row sums.

The strict importer series is too sparse for case-level matching; the relaxed
series is used for the data-boundary analysis and is labelled as such.

---

## 2. Labels (`data/labels.parquet`, rebuilt)

One row per candidate unit-year with a constructible label (2005–2024).
17 columns, 421,354 rows; 4,998 one-year positives; 11,759 three-year
positives.

| Column | Type | Meaning |
|---|---|---|
| `exporter`, `taxon`, `year` | key | unit and label year |
| `v_cnt` | float | primary volume |
| `m5` | float | mean volume of the previous five years |
| `active5` | float | number of positive years in the previous five |
| `en` | bool | endangerment marker (not used by this paper) |
| `si_c` | bool | unit rule `V > 4·M5` and `M5 ≥ 100` |
| `si3_c` | bool | unit rule in `T`, `T−1` or `T−2` |
| `si_g_u` | bool | global rule with ≥10% world share |
| `si_3x` | bool | unit rule at 3× |
| `si3x_3y` | bool | 3× unit rule in `T`, `T−1` or `T−2` |
| `hv200` | bool | `M5 > 200` |
| `source_shift` | bool | one source surges while another declines, `M5 ≥ 50` |
| `report_disc_relax` | bool | relaxed exporter/importer discrepancy |
| `y_B` | int | one-year label (`si_c` or `si_g_u`) |
| `y_B3` | int | three-year label (`si3_c` or `si_g_u`) |

`source_shift` and `report_disc_relax` are classification aids for divergence
cases; they are not used as training labels.

---

## 3. Features (`data/feat_T<year>.parquet`, rebuilt)

One row per candidate unit in label year `T`. 71 columns: two keys, 68
features, one year stamp.

| Column | Meaning |
|---|---|
| `exporter`, `taxon` | unit keys |
| `feat_year` | label year `T` |

Feature groups (68 total):

| Group | n | Columns |
|---|---|---|
| Volume dynamics | 17 | `v_log1p`, `v_lag2_log1p`, `m5_log1p`, `log5sum`, `active5`, `recency`, `cv5`, `slope5`, `max_over_mean5`, `zero_years5`, `log_records`, `qty_per_record`, `last_share5`, `v_allunit_ratio`, `v_lag3_log1p`, `v_sum_lag3_log1p`, `v_std_log` |
| Network | 18 | `deg_exp`, `deg_sp`, `share_global`, `hhi_sp`, `peer_vol_log`, `peer_growth`, `exp_total_log`, `exp_entropy`, `exp_growth`, `jac_max`, `jac_mean`, `aa_peer`, `genus_countries`, `genus_vol_share`, `partner_country_n`, `growth_rank_sp`, `deg_exp_chg`, `sp_growth` |
| Trade form | 27 | `term_entropy`, `live_share`, `top_term_share`, `terms_n`, `importers_n`, `top_importer_share`, `origin_n`, `src_W`, `src_C`, `src_F`, `src_R`, `src_D`, `src_A`, `src_U`, `v_wild_share`, `p_T`, `p_P`, `p_H`, `reexport_share`, `appendix_code`, `class_code`, `order_code`, `family_code`, `genus_code`, `is_spp`, `is_hybrid`, `taxon_n_tokens` |
| Rule state | 6 | `r1`, `r_prev`, `max_r3`, `si_lag1`, `si_lag2`, `hv200_lag` |

The **strict audit set** (`no_leak_strict`) is 60 features: all volume features
except `m5_log1p` and `log5sum`, all network features, all form features except
the eight static taxonomy fields, plus the eight static fields. In set algebra:

```
15 volume + 18 network + 19 non-static form + 8 static = 60.
```

The six rule fields are excluded. The full set (69 including `en_flag`) is
used only for the transparency contrast.

Every dynamic feature for label year `T` uses data up to year `T−1`.

---

## 4. Scores (`output/scores/*.parquet`, rebuilt)

Filename: `B__no_leak_strict__T<year>__s<seed>.parquet`.

| Column | Meaning |
|---|---|
| `exporter`, `taxon` | unit keys |
| `score` | LightGBM predicted probability for `y_B` |
| `y` | the one-year label (a copy; the audit drops it before merging the authoritative labels) |

Five test years (2018–2022) × seeds (0, 1, 2). Scores are used for ranking
only.

---

## 5. Result tables (`results/tables/`)

### `table_cases.csv`

The 34 pooled divergence cases, one row each, with the full recomputation
trace.

| Column group | Columns |
|---|---|
| Identity | `test_year`, `exporter`, `taxon`, `taxon_group`, `business_code`, `score` |
| Rule state | `si_T`, `si_T1`, `si_T2`, `si3_T` |
| Volume and mean | `v_T`, `m5_T`, `r5`, `v_T1`, `m5_T1`, `r_T1`, `v_T2`, `m5_T2`, `r_T2` |
| Support | `base_positive_years`, `base_years_all`, `base_years_used`, `base_records_median` |
| Records | `records_T1`, `rec_ratio_T1`, `records_T2`, `rec_jump_T2_T1`, `ratio_record_year`, `records_trigger_year`, `rec_ratio_trigger`, `low_trigger_records` |
| Import side | `single_importer`, `imp_vol_T1`, `impn_vol_T1`, `import_side`, `min_importers_T1`, `top_importer_share_T1` |
| Verdict | `verdict`, `verified`, `trigger_lags` |

`r5`, `r_T1`, `r_T2` are volume / five-year-mean ratios. `rec_ratio_trigger` is
the record count in the most recent trigger year divided by the median support
record count. `verdict` is one of `confirmed_multi_base`,
`confirmed_single_base`, `no_base_support`, `low_trigger_records`,
`construction_error`.

### `table_divergence.csv`

Divergence-set composition by selection rule.

| Column | Meaning |
|---|---|
| `selection` | `pooled_top50`, `per_year_top10`, `unfiltered_top50` |
| `n` | cases examined (50) |
| `n_defensible` | defensible cases |
| `n_three_year` | three-year-window cases |
| `n_near_4x` | near-threshold cases |
| `n_other_defensible` | other defensible codes |
| `plant_share` | plant share of defensible cases |
| `top1_exporter`, `top1_share` | most frequent exporter and its share |
| `n_gt`, `gt_share` | Guatemala count and share |

### `table_window_comparison.csv`

One row per arm on the fixed top-100 list.

| Column | Meaning |
|---|---|
| `arm` | `1y_3x`, `2y_4x`, `3y_3x`, `3y_4x` |
| `n_base_flagged` | one-year baseline count (34.2 per year) |
| `n_additional` | additional alerts per year |
| `n_additional_total` | total additional alerts over five years |
| `n_additional_multi_base` | of which multi-year supported |
| `n_additional_single_base` | of which single-year supported |
| `n_additional_low_records` | of which multi-year but low trigger records |
| `n_additional_verified` | of which verified |
| `verified_rate` | mean of yearly verified / additional |
| `n_after` | total flagged after the arm |

### `table_threshold_sensitivity.csv`

Same decomposition for each `(threshold, arm)` combination with a full five-year
average. `threshold` ∈ {2, 3, 4, 5}; `arm` ∈ {`1y`, `3y`}. Columns: `n_add`,
`n_multi`, `n_single`, `n_low` (multi-year support but too few trigger-year
records), `n_verified`, `n_after`.

### `table_threshold_by_year.csv`

The same arms broken out by test year: `threshold`, `arm`, `test_year`,
`n_add`, `n_multi`, `n_single`, `n_low`, `n_verified`, `n_after`.

### `table_model_family.csv`

18 rows: 2 model families (XGBoost, random forest) × 3 seeds × 3 selection
rules. The XGBoost thread count is pinned (`config.XGB_THREADS`) because the
histogram method builds different trees for different thread counts; the random
forest is thread invariant.

| Column | Meaning |
|---|---|
| `model`, `seed`, `selection` | combination |
| `n` | cases examined (50) |
| `n_defensible` | defensible cases |
| `n_three_year` | three-year-window cases |
| `n_near` | near-threshold cases |
| `overlap_lgb` | overlap with the LightGBM list |
| `plant_share` | plant share |
| `gt_share` | Guatemala share |

### `table_concentration.csv`

| Column | Meaning |
|---|---|
| `stratum` | `all_candidates`, `three_year_only`, `B3_positive`, `threshold_only` |
| `n` | units in the stratum |
| `gt_share` | Guatemala share |
| `plant_share` | plant share |

### `table_gt_sensitivity.csv`

Pooled divergence counts before and after removing Guatemala, and after
removing exporters whose reporting footprint at least doubled in 2017:
`selection`, `n`, `n_defensible`, `n_three_year`, `n_gt`, `top1`,
`top1_share`.

### `table_gt_reporting.csv`

Guatemala's annual reporting footprint, 2010–2023: `year`, `active_units`
(units with positive volume), `records`, `volume`, `taxa`. The manuscript's
Table 3 shows the 2014–2022 subset.

### `table_gt_reporting_level.csv`

Three-year pre/post comparison of the same series (2014–2016 vs 2017–2019):
`metric` (`records`, `active_units`, `taxa`), `pre_mean`, `post_mean`,
`ratio`. The manuscript's 2016→2017 year-over-year ratios (records 1.8×,
active units 7×) are checked separately from this smoothed comparison.

### `table_trigger_split.csv`

| Column | Meaning |
|---|---|
| `scope` | `cases` or `universe` |
| `group` | `T-1 only`, `T-1 and T-2`, `T-2 only`, `total`, `three-year positives` |
| `n` | count |

### `table_trigger_split_by_year.csv`

`test_year`, `n_pos3`, `n_t2_only` — the three-year positives and the subset
that fires only at `T−2`, per test year.

### `table_import_match.csv`

One row per case.

| Column | Meaning |
|---|---|
| `test_year`, `exporter`, `taxon`, `group` | case identity |
| `exporter_volume_T1` | export volume at `T−1` |
| `importer_volume_T1`, `importer_records` | matched importer totals |
| `ratio` | importer / exporter volume |
| `paired` | whether a match exists |
| `within_2x`, `within_5x` | ratio in [0.5, 2] or [0.2, 5] |
| `origin_eq_any_record`, `origin_eq_matched` | origin equality checks |
| `origin_missing_share` | share of matched records with empty origin |
| `origin_values` | non-empty origin values |

### `table_origin_match_strict.csv`

The strict-origin variant of the same matching (one row per case), for
transparency about the alternative definition.

### `table_report_lag_fields.csv`

`column_name`, `column_type`, `lag_field_candidate` — schema scan for any
submission/revision timestamp. In the 2026.1 export there is none.

### `table_trade_diversion_decision.csv`

`task`, `status`, `reason` — records that the trade-diversion analysis was not
run and why (no review dates or compliance outcomes in the database).

---

## 6. Figure data (`results/figure_data/`)

| File | Rows | Columns |
|---|---|---|
| `case_series.csv` | 204 | `exporter`, `taxon`, `test_year`, `offset` (−5…0), `year`, `volume`, `threshold`, `fires` |
| `case_flag_matrix.csv` | 34 | `exporter`, `taxon`, `test_year`, `-5`, `-4`, `-3`, `si_T2`, `si_T1`, `si_T`, `n_fire` |
| `case_timelines.csv` | 42 | `exporter`, `taxon`, `taxon_group`, `test_year`, `year`, `offset`, `volume` (six representative cases × seven years) |
| `divergence_composition.csv` | 150 | case-level label states for the three selection rules (wide table; one row per case × selection) |
| `window_by_year.csv` | 20 | `test_year`, `arm`, `n_base_flagged`, `n_additional`, `n_additional_multi_base`, `n_additional_single_base`, `n_additional_low_trigger_records`, `n_additional_verified`, `verified_rate`, `n_after` |
| `year_coverage.csv` | 19 | `year` (2005–2023), `active_units`, `records`, `volume` |

`case_series.csv` is the source of the per-case line charts: `threshold` is
4× the mean of the previous five years at each offset, and `fires` is the rule
state for that year.

---

## 7. Checks (`results/checks/`)

- `expected.json` — 66 frozen expectations grouped by block (`cases`, `flags`,
  `divergence`, `trigger_split`, `window`, `concentration`, `import`,
  `model_family`, `threshold`, `coverage`).
- `checks.csv` — one row per check: `block`, `name`, `expected`, `observed`,
  `status`, `tolerance`.
- `checks.json` — the same record as JSON, plus the run timestamp and the
  overall pass/fail.

`rst-audit checks --freeze` rewrites `expected.json`. It should be used only
when the manuscript numbers are deliberately changed.

---

## 8. `results/paper_numbers.json`

A single-file summary of the manuscript's headline numbers: pool size, case
counts, importer-side match counts, Guatemala sensitivities, and the
counterfactual arms. It is convenient for downstream reuse, but the check
contract lives in `results/checks/expected.json`, not here.

## 9. `results/model_results.csv`

One row per (label, feature set, test year, seed) with the ranking metrics of
the rule-free LightGBM: `n_test`, `n_pos`, hit counts and precision at 50/100/
200, recall@100, NDCG@100, average precision, the random-baseline interval and
the lift over that baseline. It is written by `rst-audit models` and holds 15
rows (5 test years × 3 seeds) for the primary label `B`.
