# Methods

This document describes the analysis exactly as implemented in
`src/rst_audit/`. It is the reference for the manuscript's Data and Methods
section; where the paper and this document disagree, the code and the checks in
`results/checks/` decide.

## 1. Scope

The audit evaluates the **detection layer** of the CITES Review of Significant
Trade (RST): the sharp-increase rule and its temporal window. It does not
evaluate non-detriment findings, review recommendations, compliance responses,
or conservation outcomes.

Two layers are distinguished:

- **Detection:** does the rule, applied to a given year, flag a unit that
  should enter review?
- **Post-trigger review:** what happens after a unit is flagged?

The 2026 amendment changes only the detection layer: the sharp-increase test is
applied to each of the three years since the previous screening rather than to
the focus year alone. The audit asks whether that change expands *record-
supported* detection coverage or merely increases the number of alerts.

## 2. Data and unit

### 2.1 Source

CITES Trade Database, version 2026.1. Panel years 2000–2024. The 2025 annual
file was incomplete at the study snapshot and is excluded. Labels can be built
from 2005 onward because the rule needs five prior years.

### 2.2 Primary volume

A unit is an exporter × taxon pair (ISO-2 country code plus a species or higher
taxon). The primary volume is

```
V(i,t) = sum of Quantity over records with
         Year = t, Exporter = i, Taxon = i,
         Reporter.type = 'E',
         (Origin IS NULL OR Origin = Exporter),
         (Unit IS NULL OR Unit = 'Number of specimens').
```

Records with `Quantity` NULL are counted as zero (the observed NULL rate in the
primary scope is 0.0). The panel also stores sensitivity volumes: all-unit
volume, wild-sourced volume, live wild-sourced volume, re-export volume,
importer-reported volume, and source/purpose/importer breakdowns.

Panel size: 301,914 unit-years; 32,062 exporter–taxon pairs occur in the
2018–2022 test period. Candidate counts per test year range from 18,591 to
22,191; the five test years contain 101,509 candidate unit-years.

### 2.3 Candidate universe

```
P_t = { i : M(i,t) > 0 and exists k in {1,2,3} with V(i,t−k) > 0 }
```

where `M(i,t)` is the mean of `V(i,t−5) … V(i,t−1)`. The universe therefore
contains units with a positive five-year mean and at least one positive year in
the previous three. This excludes units with no recent reported activity.

## 3. Sharp-increase rule and labels

### 3.1 Unit rule

```
S(i,t) = 1[ V(i,t) > 4 * M(i,t) and M(i,t) >= 100 ].
```

The multiplier 4 and the floor 100 are the values in the available 2026
documents. The floor applies to the five-year mean, not to the current year.
The audit does not claim that the 2026 amendment changed either value; it
claims only that no change is visible in the available documents.

### 3.2 Global rule

For taxon `s(i)` (the species or higher taxon used for the global aggregate),

```
V_s(t) = sum over units of taxon s(i) of V(i,t)
M_s(t) = mean of V_s(t−5) … V_s(t−1)
G(i,t) = 1[ V_s(t) > 4 * M_s(t) and M_s(t) >= 100
          and V(i,t) / V_s(t) >= 0.10 ].
```

### 3.3 Labels

One-year label (training target):

```
y1(i,t) = S(i,t) or G(i,t).
```

Three-year label (audit only, never a training target):

```
y3(i,t) = S(i,t) or S(i,t−1) or S(i,t−2) or G(i,t).
```

The asymmetry is deliberate: global surges are evaluated in the focus year,
while unit surges are evaluated in any of the three years. Training on `y1` and
auditing with `y3` avoids the circularity of training on the three-year label
and then evaluating three-year cases.

Frozen counts: 421,354 label rows; 4,998 one-year positives; 11,759 three-year
positives.

### 3.4 Window and threshold arms

For window length `w ∈ {1,2,3}` and multiplier `τ`:

```
S_{w,τ}(i,t) = 1[ exists k in {0,…,w−1}: V(i,t−k) > τ * M(i,t−k)
                  and M(i,t−k) >= 100 ].
```

The baseline is `w=1, τ=4`. The paper compares `w=2, τ=4`, `w=3, τ=4`,
`w=1, τ∈{2,3,5}`, `w=3, τ=5`, and `w=3, τ=3`, all applied to the same fixed
candidate list.

## 4. Rule-free feature set

### 4.1 Why exclude rule features

A model that can see the rule ratio, the five-year mean, or the five-year total
can reproduce the rule. Its top-ranked units would then be near-threshold
duplicates, and the resulting divergence set would be evidence about the
threshold, not about temporal coverage. The audit therefore uses a feature set
that excludes:

- sharp-increase flags and lagged flags;
- rule ratios and their near proxies;
- the five-year mean and five-year total in levels;
- the endangerment (EN) flag.

### 4.2 The 60 features

| Group | n | Examples |
|---|---|---|
| Trade-volume dynamics | 15 | log volume, lag-2 and lag-3 volume, five-year sum, active years, recency, coefficient of variation, slope, max/mean, zero years, record count |
| Network position | 18 | degree (exporter/species), global share, species HHI, peer volume and growth, partner counts, Jaccard neighbours, Adamic–Adar, genus-level country count, growth rank |
| Trade form | 19 | term entropy, live share, top term share, importer counts, top importer share, origin count, source shares (W/C/F/R/D/A/U), purpose shares, re-export share |
| Taxonomy (static) | 8 | appendix, class, order, family, genus codes, species/hybrid flags, token count |

The strict set has exactly 60 features. The five-year mean (`m5_log1p`) and
five-year total (`log5sum`) are excluded, as are the six rule-state fields
(`r1`, `r_prev`, `max_r3`, `si_lag1`, `si_lag2`, `hv200_lag`) and `en_flag`.
The full set (69 features) is kept for the transparency contrast only.

### 4.3 Temporal discipline

For label year `t`, all dynamic features use records from `t−1` and earlier.
Network and record-quality features are computed within the same cutoff.
Static taxonomy does not vary with time. String taxon attributes are
forward-filled only (never backward-filled), so a later name change cannot leak
into an earlier year.

## 5. Ranking model

### 5.1 Model

LightGBM binary classifier:

| Parameter | Value |
|---|---|
| Objective | binary log-loss |
| Trees | 300 |
| Learning rate | 0.05 |
| Leaves | 31 |
| Min data in leaf | 50 |
| Feature fraction | 0.8 |
| Bagging fraction / frequency | 0.8 / 1 |
| Positive weight | negative/positive ratio |

### 5.2 Rolling origin

For each test year `T ∈ {2018, …, 2022}`, the model is trained on all candidate
unit-years with label years `< T` and scored on year `T`. The model never sees
`T` or later when scoring `T`. No random train/test split is used.

The main audit uses seed 0. Seeds 1 and 2 are used to check whether the
divergence set is seed-dependent. Model-family checks replace LightGBM with
XGBoost or a random forest, using the same features and rolling origin, without
per-model tuning.

### 5.3 Role of the score

The score is used only to produce a candidate ranking. The audit depends on the
top of the ranking, not on AUC or calibration. The manuscript reports ranking
performance only as context.

## 6. Divergence set

For test year `T`, let `rank(i)` be the model rank of unit `i`. The divergence
set is

```
D_K(T) = { i in P_T : rank(i) <= K, S(i,T) = 0 }.
```

Three selection rules are reported:

| Rule | Definition |
|---|---|
| Pooled top 50 | all test years pooled; top 50 by score |
| Per-year top 10 | top 10 in each of five test years (50 total) |
| Unfiltered top 50 | top 50 by score without the candidate-universe filter |

Each case receives a business code:

- `three_year_window`: `S(i,T)=0` but `S(i,T−1)=1` or `S(i,T−2)=1`;
- `threshold_only`: close to the 4× threshold but below it;
- `high_volume_only`: passes the volume floor but not the ratio;
- `source_shift`, `report_disc`: alternative explanations retained for
  classification, not counted as defensible cases;
- `outside_high_volume`, `no_clear_reason`: residual categories.

A case is **defensible** when it is a three-year-window case, a near-threshold
case, or one of the other explicitly coded alternatives. In the shipped pooled
set, all 34 defensible cases are three-year-window cases and none is
near-threshold.

## 7. Case verification

### 7.1 Recomputation

Every case is recomputed from the raw panel: volume `V`, five-year mean `M`,
the 4× threshold, and the rule state for `T−5 … T`. Arithmetic inconsistencies
are labelled `construction_error` and fail the audit.

### 7.2 Support window

```
T_i = { y in {T−1, T−2} : S(i,y) = 1 }          trigger years
B_i = {T−5, …, T−1} \ T_i                        support window
B_i+ = { y in B_i : V(i,y) > 0 }                 positive support years
```

`B_i` is used only to evaluate historical record support on a common time
range. It is not the five-year mean window used to compute `S(i,T−1)` (that is
`T−6 … T−2`) or `S(i,T−2)` (`T−7 … T−3`).

### 7.3 Verdict

A case is **verified** (internal check passed) when

```
|B_i+| >= 2
and
records(i, max(T_i)) >= 0.5 * median{ records(i,y) : y in B_i+ }.
```

- `confirmed_multi_base`: at least two positive support years and the record
  ratio condition holds → verified.
- `confirmed_single_base`: exactly one positive support year → watchlist.
- `no_base_support`: zero positive support years → reported separately.
- `low_trigger_records`: multi-year support but record ratio below 0.5 → not
  verified.

"Verified" is a statement about internal record consistency. It is not a
confirmation of real trade expansion, unsustainability, or illegality.

## 8. Fixed-list comparison

The top 100 model candidates for each test year are frozen. Each arm
(window × threshold) is applied to that list, not to the full universe:

```
Top100(t) = { i in P_t : rank(i) <= 100 }
A_{w,τ}(t) = { i in Top100(t) : S_{w,τ}(i,t) = 1 }
Δ_{w,τ}(t) = |A_{w,τ}(t) \ A_{1,4}(t)|
```

For each arm the audit reports:

- additional alerts per year;
- of those, multi-year supported, single-year supported, low-record, verified;
- verified rate (mean of yearly rates);
- total flagged after the arm.

The baseline one-year rule flags 34.2 units per year on the fixed list. The
three-year window flags 60.4; the 3× threshold flags 40.4.

## 9. Sensitivity analyses

### 9.1 Threshold

Hold the model score fixed; vary `τ ∈ {2,3,4,5}` for the one-year and
three-year arms. Report the same additional/verified decomposition. The
window-versus-threshold ratio is 5.8 at 3×, 3.1 at 2×, and undefined at 4×/5×
because the one-year threshold adds no verified cases there.

### 9.2 Model family

XGBoost and random forest, three seeds each, three selection rules each: 18
combinations. Report the number of three-year cases and near-threshold cases in
the top 50, and overlap with the LightGBM list. The mechanism is stable if the
three-year share of defensible cases stays high; the specific list may change.

### 9.3 Concentration

Compare Guatemala's share in the candidate universe, in three-year-only cases,
and in all three-year positives. Then recompute the pooled divergence count
after removing Guatemala, and again after removing every exporter whose
reporting footprint at least doubled in 2017.

### 9.4 Importer-side matching

For each case, match importer-reported records on taxon, year `T−1`, and
exporter. Sum quantities within a key before comparing. Report:

- paired / unpaired;
- origin field missing share;
- non-empty origin values;
- import/export quantity ratio and whether it lies in [0.5, 2] or [0.2, 5];
- top importer share.

No unit conversion is applied, so the ratio is only a same-unit reference.

### 9.5 Lag-field availability

Scan the `trade` table schema for any field that could record submission or
revision dates. The shipped table `table_report_lag_fields.csv` records which
columns exist and whether each is a lag-field candidate. In the 2026.1 export,
no submission/revision timestamp is present; this is why the lag interpretation
is a literature-supported hypothesis rather than a measured mechanism.

### 9.6 Trade diversion

Not run. The database has no review dates or compliance outcomes, and a
descriptive shift cannot identify diversion caused by the window. The decision
and reason are recorded in `table_trade_diversion_decision.csv`.

## 10. Reproduction contract

`results/checks/expected.json` contains 66 quantities computed from the
shipped tables:

| Block | Examples |
|---|---|
| Cases | total, verified, multi/single support, low records, construction errors |
| Flags | `T−2`, `T−1`, `T` fire counts |
| Divergence | pooled/per-year/unfiltered counts |
| Trigger split | cases by trigger year; universe three-year positives |
| Window arms | baseline and each arm's additional/verified counts |
| Concentration | Guatemala and plant shares by stratum |
| Import match | paired count, origin missing, ratio bins |
| Model family | three-year share range, overlap range |
| Threshold | each arm's verified additions |
| Coverage | active units and records by year |

`rst-audit checks` recomputes each quantity and compares it with the frozen
value at tolerance `5e-4`. A mismatch exits non-zero. `rst-audit checks
--freeze` regenerates the expectations; this should only be done when the
manuscript numbers are deliberately revised.

## 11. Limitations

1. **Single snapshot.** Submission dates and revisions are not observable. The
   lag interpretation is consistent with the literature and the case structure
   but is not a measured submission curve.
2. **Model dependence.** The divergence set depends on the model and feature
   set. Model-family and seed checks show the mechanism is stable, but the
   exact list is not. Rankings are not causal claims.
3. **Internal verification only.** "Verified" means the records are internally
   consistent and have multi-year support. It does not confirm that the trade
   expansion is real or unsustainable.
4. **Importer side.** Origin fields are mostly missing and units are not
   harmonised. Importer records can be paired but cannot verify magnitude or
   the trade chain.
5. **Detection layer only.** Post-trigger review and conservation outcomes are
   outside the data. The policy conclusion applies to detection coverage, not
   to RST effectiveness as a whole.
6. **Plants.** The plant share of the divergence set is high. The data cannot
   separate biological and logistical factors from reporting conventions; the
   audit flags the need for plant-specific review capacity rather than
   attributing the share to one cause.
