"""Detection-layer audit: divergence sets, case checks, window comparison.

The unit of analysis is an exporter x taxon x focus year (T).  A case is
"checked" when its support window contains at least two positive-record years
and the trigger-year record count is at least half the median of the support
years.  The record-ratio numerator belongs to the most recent trigger year.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

from . import config as C
from .io_utils import Logger, configure_stdout, read_csv, write_csv, write_json
from .labels import roll_sum, YEARS

DEFENSIBLE = {"three_year_window", "near_threshold_4x", "min_mean_gate",
              "new_flow", "relaxed_3y"}
CODE_ORDER = ["three_year_window", "near_threshold_4x", "min_mean_gate",
              "new_flow", "relaxed_3y", "outside_source_shift",
              "outside_report_disc", "outside_high_volume", "no_clear_reason"]
CODE_LABEL = {
    "three_year_window": "Window blind spot",
    "near_threshold_4x": "Near threshold (3-4x)",
    "min_mean_gate": "Minimum-volume gate",
    "new_flow": "New trade flow",
    "relaxed_3y": "3-year 3x window",
    "outside_source_shift": "Outside: source shift",
    "outside_report_disc": "Outside: reporting discrepancy",
    "outside_high_volume": "Outside: high volume (HV200)",
    "no_clear_reason": "No clear SI reason",
}


def si_tables(panel: pd.DataFrame, mult: float = C.SI_MULT,
              minmean: float = C.SI_MIN_MEAN, global_share: float = C.GLOBAL_SHARE) -> pd.DataFrame:
    """Long frame of unit and global sharp-increase atoms for every unit-year."""
    units, p, mat = _unit_matrices(panel)
    V = mat("v_cnt")
    M5 = roll_sum(V, -5, 0) / 5.0
    sp_ids, sp_inv = np.unique(units["taxon"].values, return_inverse=True)
    Vsp = np.zeros((len(sp_ids), len(YEARS)))
    np.add.at(Vsp, sp_inv, V)
    Msp = roll_sum(Vsp, -5, 0) / 5.0
    si_unit = (V > mult * np.maximum(M5, 1e-9)) & (M5 >= minmean)
    si_global = (Vsp > mult * np.maximum(Msp, 1e-9)) & (Msp >= minmean)
    si_g_u = si_global[sp_inv] & (V / np.maximum(Vsp[sp_inv], 1e-9) >= global_share)
    rows = []
    for j, year in enumerate(YEARS):
        rows.append(pd.DataFrame({
            "exporter": units["exporter"].values,
            "taxon": units["taxon"].values,
            "year": year,
            "v_cnt": V[:, j],
            "m5": M5[:, j],
            "si_c": si_unit[:, j],
            "si_g_u": si_g_u[:, j],
        }))
    return pd.concat(rows, ignore_index=True)


def _unit_matrices(panel: pd.DataFrame):
    from .labels import _unit_matrices as base
    return base(panel)


def m3_matrix(panel: pd.DataFrame) -> pd.DataFrame:
    """Mean volume over the three years before each column year."""
    wide = panel.pivot_table(index=["exporter", "taxon"], columns="year",
                             values="v_cnt", fill_value=0).reindex(columns=YEARS, fill_value=0)
    arr = wide.to_numpy(float)
    out = np.zeros_like(arr)
    for j in range(len(YEARS)):
        lo = max(0, j - 3)
        if j > lo:
            out[:, j] = arr[:, lo:j].mean(axis=1)
    return pd.DataFrame(out, index=wide.index, columns=wide.columns)


def taxon_group(panel: pd.DataFrame) -> pd.Series:
    meta = panel[["taxon", "cls", "ord"]].drop_duplicates("taxon").fillna("UNK")
    values = np.where(meta["cls"].values != "UNK", "animal",
                      np.where(meta["ord"].values != "UNK", "plant", "other"))
    return pd.Series(values, index=meta["taxon"].values)


def code_business(row) -> str:
    if row["si3_c"] == 1 and row["si_c"] == 0:
        return "three_year_window"
    if 3 <= row["r5"] < 4 and row["m5_prev"] >= 100:
        return "near_threshold_4x"
    if row["r5"] >= 4 and 0 < row["m5_prev"] < 100:
        return "min_mean_gate"
    if row["m5_prev"] < 10 and row["v_cnt"] >= 100:
        return "new_flow"
    if row["r3"] >= 3 and row["m3_prev"] >= 100:
        return "relaxed_3y"
    if row["source_shift"] == 1:
        return "outside_source_shift"
    if row["report_disc_relax"] == 1:
        return "outside_report_disc"
    if row["hv200"] == 1:
        return "outside_high_volume"
    return "no_clear_reason"


def build_audit_pool(labels: pd.DataFrame, panel: pd.DataFrame,
                     score_label: str = C.LABEL, feature_set: str = C.FEATURE_SET,
                     seed: int = 0, years: list[int] | None = None) -> pd.DataFrame:
    years = years or C.TEST_YEARS
    m3 = m3_matrix(panel)
    rows = []
    for T in years:
        score_path = C.score_path(score_label, feature_set, T, seed)
        if not score_path.exists():
            raise FileNotFoundError(f"missing model scores: {score_path}; run run_models first")
        scores = pd.read_parquet(score_path)
        # Score files may carry their own copy of the labels; the audit pool reads
        # labels from the single authoritative label table to avoid *_x/*_y merges.
        scores = scores.drop(columns=[c for c in ("y", "y_B", "y_B3") if c in scores.columns])
        d = scores.merge(labels[labels["year"] == T], on=["exporter", "taxon"],
                         how="left", validate="one_to_one")
        if d["y_B"].isna().any():
            raise RuntimeError(f"label merge failed at {T}")
        d["test_year"] = T
        prev = labels[labels["year"] == T - 1][["exporter", "taxon", "m5"]].rename(
            columns={"m5": "m5_prev"})
        d = d.merge(prev, on=["exporter", "taxon"], how="left")
        d["r5"] = np.where(d["m5"] > 0, d["v_cnt"] / d["m5"], np.inf)
        idx = pd.MultiIndex.from_arrays([d["exporter"], d["taxon"]])
        d["m3_prev"] = m3.reindex(idx)[T].to_numpy()
        d["r3"] = np.where(d["m3_prev"] > 0, d["v_cnt"] / d["m3_prev"], np.inf)
        rows.append(d)
    pool = pd.concat(rows, ignore_index=True)
    pool["business_code"] = pool.apply(code_business, axis=1)
    pool["defensible"] = pool["business_code"].isin(DEFENSIBLE)
    pool["three_year_only"] = (pool["y_B3"] == 1) & (pool["y_B"] == 0)
    pool["threshold_only"] = (pool["si_3x"] == 1) & (pool["si_c"] == 0)
    groups = taxon_group(panel).rename_axis("taxon").rename("taxon_group").reset_index()
    pool = pool.merge(groups, on="taxon", how="left")
    pool["taxon_group"] = pool["taxon_group"].fillna("other")
    return pool


def divergence_sets(pool: pd.DataFrame) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    out = {}
    base = pool[pool["y_B"] == 0]
    out["pooled_top50"] = base[base["v_cnt"] > 0].nlargest(C.DIVERGENCE_TOP, "score").copy()
    out["per_year_top10"] = (
        base[base["v_cnt"] > 0]
        .sort_values(["test_year", "score"], ascending=[True, False])
        .groupby("test_year").head(C.PER_YEAR_TOP).copy()
    )
    out["unfiltered_top50"] = base.nlargest(C.DIVERGENCE_TOP, "score").copy()
    rows = []
    for name, d in out.items():
        rows.append(dict(
            selection=name, n=len(d), n_defensible=int(d["defensible"].sum()),
            n_three_year=int((d["business_code"] == "three_year_window").sum()),
            n_near_4x=int((d["business_code"] == "near_threshold_4x").sum()),
            n_other_defensible=int(d["defensible"].sum()
                                   - (d["business_code"] == "three_year_window").sum()
                                   - (d["business_code"] == "near_threshold_4x").sum()),
            plant_share=float((d["taxon_group"] == "plant").mean()),
            top1_exporter=d["exporter"].value_counts().index[0],
            top1_share=float(d["exporter"].value_counts().iloc[0] / len(d)),
            n_gt=int((d["exporter"] == "GT").sum()),
            gt_share=float((d["exporter"] == "GT").mean()),
        ))
    return out, pd.DataFrame(rows)


def case_verify(pool, panel: pd.DataFrame, labels: pd.DataFrame,
                cases: pd.DataFrame) -> pd.DataFrame:
    """Recompute every case from the panel and apply the two-part check."""
    si = si_tables(panel).set_index(["exporter", "taxon", "year"])
    rows = []
    for _, r in cases.iterrows():
        T, exporter, taxon = int(r["test_year"]), r["exporter"], r["taxon"]
        sub = panel[(panel.exporter == exporter) & (panel.taxon == taxon)].set_index("year")

        def sget(year, col):
            key = (exporter, taxon, year)
            return float(si.loc[key, col]) if key in si.index else 0.0

        def pget(year, col):
            return float(sub.loc[year, col]) if year in sub.index else 0.0

        lab = labels[(labels.exporter == exporter) & (labels.taxon == taxon)].set_index("year")
        si1, si2 = int(sget(T - 1, "si_c")), int(sget(T - 2, "si_c"))
        triggers = {T - 1 if si1 else None, T - 2 if si2 else None} - {None}
        all_base = list(range(T - 5, T))
        base_years = [y for y in all_base if y not in triggers]
        base_pos = [y for y in base_years if pget(y, "v_cnt") > 0]
        base_records = [pget(y, "records") for y in base_pos]
        median_records = float(np.median(base_records)) if base_records else np.nan
        rec_T1 = pget(T - 1, "records")
        rr1 = rec_T1 / median_records if median_records and median_records > 0 else np.nan
        rec_T2 = pget(T - 2, "records")
        rec_jump = rec_T1 / rec_T2 if rec_T2 > 0 else np.nan
        rec_year = T - 1 if si1 else (T - 2 if si2 else T - 1)
        rec_trigger = rec_T1 if rec_year == T - 1 else rec_T2
        rr_trigger = rec_trigger / median_records if median_records and median_records > 0 else np.nan
        v = float(lab.loc[T, "v_cnt"]) if T in lab.index else 0.0
        m5 = float(lab.loc[T, "m5"]) if T in lab.index else 0.0
        si_T = int(lab.loc[T, "si_c"]) if T in lab.index else 0
        si3_T = int(lab.loc[T, "si3_c"]) if T in lab.index else 0
        v1, m51 = sget(T - 1, "v_cnt"), sget(T - 1, "m5")
        v2, m52 = sget(T - 2, "v_cnt"), sget(T - 2, "m5")
        imp1 = pget(T - 1, "v_imp_total")
        impn1 = pget(T - 1, "v_impn_total")
        if si_T != 0 or si3_T != 1 or (si1 == 0 and si2 == 0):
            verdict = "construction_error"
        elif len(base_pos) >= 2:
            verdict = "confirmed_multi_base"
        elif len(base_pos) == 1:
            verdict = "confirmed_single_base"
        else:
            verdict = "no_base"
        low_records = (not np.isnan(rr_trigger)) and rr_trigger < C.MIN_RECORD_RATIO
        single_importer = pget(T - 1, "top_importer_share") >= 0.9
        verified = (verdict == "confirmed_multi_base") and not low_records
        rows.append(dict(
            test_year=T, exporter=exporter, taxon=taxon,
            taxon_group=r.get("taxon_group", "other"), business_code=r.get("business_code", ""),
            score=float(r["score"]), si_T=si_T, si_T1=si1, si_T2=si2, si3_T=si3_T,
            v_T=v, m5_T=m5, r5=v / m5 if m5 > 0 else np.nan,
            v_T1=v1, m5_T1=m51, r_T1=v1 / m51 if m51 > 0 else np.nan,
            v_T2=v2, m5_T2=m52, r_T2=v2 / m52 if m52 > 0 else np.nan,
            base_positive_years=len(base_pos),
            base_years_all="|".join(map(str, all_base)),
            base_years_used="|".join(map(str, base_years)),
            base_records_median=median_records, records_T1=rec_T1, rec_ratio_T1=rr1,
            records_T2=rec_T2, rec_jump_T2_T1=rec_jump,
            ratio_record_year=rec_year, records_trigger_year=rec_trigger,
            rec_ratio_trigger=rr_trigger, low_trigger_records=bool(low_records),
            single_importer=single_importer, imp_vol_T1=imp1, impn_vol_T1=impn1,
            import_side="missing" if imp1 == 0 else "present",
            min_importers_T1=pget(T - 1, "importers_n"),
            top_importer_share_T1=pget(T - 1, "top_importer_share"),
            verdict=verdict, verified=bool(verified),
            trigger_lags="|".join([x for x, flag in [("T-1", si1), ("T-2", si2)] if flag]),
        ))
    return pd.DataFrame(rows)


def window_comparison(pool: pd.DataFrame, panel: pd.DataFrame,
                      labels: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Compare window extension and threshold cuts on the same top-100 list."""
    rows, detail = [], []
    prev = labels[["exporter", "taxon", "year", "si_c"]].rename(columns={"si_c": "si_c_prev"})
    prev["year"] = prev["year"] + 1
    for T in C.TEST_YEARS:
        d = pool[pool["test_year"] == T].nlargest(C.TOP_K, "score").copy()
        d = d.merge(prev, on=["exporter", "taxon", "year"], how="left", validate="one_to_one")
        d["si_c_prev"] = d["si_c_prev"].fillna(False).astype(int)
        base = d["y_B"] == 1
        arms = {
            "2y_4x": ((d["y_B"] == 1) | (d["si_c_prev"] == 1)) & ~base,
            "3y_4x": (d["y_B3"] == 1) & ~base,
            "1y_3x": (d["si_3x"] == 1) & ~base,
            "3y_3x": (d["si3x_3y"] == 1) & ~base,
        }
        for arm, mask in arms.items():
            additions = d[mask].copy()
            if len(additions):
                checked = case_verify(None, panel, labels, additions)
                n_multi = int((checked.verdict == "confirmed_multi_base").sum())
                n_single = int((checked.verdict == "confirmed_single_base").sum())
                n_low = int(checked.low_trigger_records.sum())
                n_verified = int(checked.verified.sum())
            else:
                n_multi = n_single = n_low = n_verified = 0
            rows.append(dict(
                test_year=T, arm=arm, n_base_flagged=int(base.sum()),
                n_additional=int(mask.sum()), n_additional_multi_base=n_multi,
                n_additional_single_base=n_single,
                n_additional_low_trigger_records=n_low,
                n_additional_verified=n_verified,
                verified_rate=float(n_verified / mask.sum()) if mask.sum() else np.nan,
                n_after=int(base.sum() + mask.sum()),
            ))
            if len(additions):
                detail.append(checked.assign(arm=arm))
    yearly = pd.DataFrame(rows)
    agg = yearly.groupby("arm").agg(
        n_base_flagged=("n_base_flagged", "mean"),
        n_additional=("n_additional", "mean"),
        n_additional_total=("n_additional", "sum"),
        n_additional_multi_base=("n_additional_multi_base", "mean"),
        n_additional_single_base=("n_additional_single_base", "mean"),
        n_additional_low_records=("n_additional_low_trigger_records", "mean"),
        n_additional_verified=("n_additional_verified", "mean"),
        verified_rate=("verified_rate", "mean"),
        n_after=("n_after", "mean"),
    ).reset_index()
    detail = pd.concat(detail, ignore_index=True) if detail else pd.DataFrame()
    return yearly, agg, detail


def reporting_break(panel: pd.DataFrame) -> set[str]:
    """Exporters whose reporting footprint at least doubled in 2017."""
    out: set[str] = set()
    for col, agg in [("records", "sum"), ("v_cnt", "size")]:
        before = panel[panel.year == 2016].groupby("exporter")[col].agg(agg)
        after = panel[panel.year == 2017].groupby("exporter")[col].agg(agg)
        ratio = (after / before.replace(0, np.nan)).dropna()
        out |= set(ratio[ratio >= 2.0].index)
    before = panel[panel.year == 2016].groupby("exporter")["taxon"].nunique()
    after = panel[panel.year == 2017].groupby("exporter")["taxon"].nunique()
    ratio = (after / before.replace(0, np.nan)).dropna()
    out |= set(ratio[ratio >= 2.0].index)
    return out


def gt_sensitivity(pool: pd.DataFrame, panel: pd.DataFrame):
    rows = []
    base = pool[(pool["y_B"] == 0) & (pool["v_cnt"] > 0)]
    for name, d in [
        ("all", base),
        ("exclude_GT", base[base["exporter"] != "GT"]),
        ("exclude_reporting_break", base[~base["exporter"].isin(reporting_break(panel))]),
    ]:
        selected = d.nlargest(C.DIVERGENCE_TOP, "score")
        rows.append(dict(
            selection=name, n=len(selected), n_defensible=int(selected["defensible"].sum()),
            n_three_year=int((selected["business_code"] == "three_year_window").sum()),
            n_gt=int((selected["exporter"] == "GT").sum()),
            top1=selected["exporter"].value_counts().index[0],
            top1_share=float(selected["exporter"].value_counts().iloc[0] / len(selected)),
        ))
    selection = pd.DataFrame(rows)

    series = []
    for year in range(2010, 2024):
        g = panel[(panel["year"] == year) & (panel["exporter"] == "GT") & (panel["v_cnt"] > 0)]
        series.append(dict(year=year, active_units=len(g), records=float(g["records"].sum()),
                           volume=float(g["v_cnt"].sum()), taxa=int(g["taxon"].nunique())))
    series = pd.DataFrame(series)
    pre = series[(series.year >= 2014) & (series.year <= 2016)]
    post = series[(series.year >= 2017) & (series.year <= 2019)]
    level = pd.DataFrame([
        dict(metric="records", pre_mean=pre.records.mean(), post_mean=post.records.mean(),
             ratio=post.records.mean() / pre.records.mean()),
        dict(metric="active_units", pre_mean=pre.active_units.mean(),
             post_mean=post.active_units.mean(),
             ratio=post.active_units.mean() / pre.active_units.mean()),
        dict(metric="taxa", pre_mean=pre.taxa.mean(), post_mean=post.taxa.mean(),
             ratio=post.taxa.mean() / pre.taxa.mean()),
    ])

    concentration = []
    for name, mask in [
        ("all_candidates", pool["y_B"] == 0),
        ("three_year_only", pool["three_year_only"]),
        ("B3_positive", pool["y_B3"] == 1),
        ("threshold_only", pool["threshold_only"]),
    ]:
        sub = pool[mask]
        concentration.append(dict(stratum=name, n=len(sub),
                                  gt_share=float((sub.exporter == "GT").mean()),
                                  plant_share=float((sub.taxon_group == "plant").mean())))
    return selection, series, level, pd.DataFrame(concentration)


def import_side(cases: pd.DataFrame):
    """Can importer records be tied to the exporter of origin?"""
    import duckdb

    if not C.CITES_DB.exists():
        raise FileNotFoundError(C.CITES_DB)
    con = duckdb.connect(str(C.CITES_DB), read_only=True)
    taxa = sorted(cases["taxon"].unique())
    marks = ",".join(["?"] * len(taxa))
    query = f"""
        SELECT Year, Taxon, Importer, Exporter, Origin, Source,
               SUM(Quantity) AS q, COUNT(*) AS n
        FROM trade
        WHERE "Reporter.type"='I' AND Year BETWEEN 2016 AND 2022 AND Taxon IN ({marks})
        GROUP BY 1,2,3,4,5,6
    """
    raw = con.execute(query, taxa).fetchdf()
    con.close()
    rows = []
    for _, r in cases.iterrows():
        T, exporter, taxon = int(r.test_year), r.exporter, r.taxon
        for year, lag in [(T - 1, "T-1"), (T, "T")]:
            sub = raw[(raw.Year == year) & (raw.Taxon == taxon)]
            if len(sub) == 0:
                rows.append(dict(test_year=T, exporter=exporter, taxon=taxon, year=year, lag=lag,
                                 importer_records=0, origin_missing_share=np.nan,
                                 origin_eq_exporter_share=np.nan, importer_volume=0.0))
                continue
            rows.append(dict(
                test_year=T, exporter=exporter, taxon=taxon, year=year, lag=lag,
                importer_records=int(sub["n"].sum()),
                origin_missing_share=float(sub["Origin"].isna().mean()),
                origin_eq_exporter_share=float((sub["Origin"] == exporter).mean()),
                importer_volume=float(sub["q"].sum()),
            ))
    return pd.DataFrame(rows), raw


def import_multifield(raw: pd.DataFrame, cases: pd.DataFrame) -> pd.DataFrame:
    """Pair importer records by the reported Exporter field, then inspect Origin."""
    rows = []
    for _, r in cases.iterrows():
        T, exporter, taxon = int(r.test_year), r.exporter, r.taxon
        sub = raw[(raw.Taxon == taxon) & (raw.Year == T - 1)]
        matched = sub[sub.Exporter == exporter]
        importer_q = float(matched.q.sum())
        exporter_q = float(r.v_T1)
        ratio = importer_q / exporter_q if exporter_q > 0 else np.nan
        paired = bool(len(matched) > 0)
        rows.append(dict(
            test_year=T, exporter=exporter, taxon=taxon, group=r.taxon_group,
            exporter_volume_T1=exporter_q, importer_volume_T1=importer_q,
            importer_records=int(matched.n.sum()), ratio=ratio, paired=paired,
            within_2x=bool(ratio == ratio and 0.5 <= ratio <= 2.0),
            within_5x=bool(ratio == ratio and 0.2 <= ratio <= 5.0),
            origin_eq_any_record=bool((sub.Origin == exporter).any()),
            origin_eq_matched=bool((matched.Origin == exporter).any()),
            origin_missing_share=float(matched.Origin.isna().mean()) if paired else np.nan,
            origin_values=("|".join(sorted(set(matched.Origin.dropna().astype(str))))
                           if paired else ""),
        ))
    return pd.DataFrame(rows)


def case_timelines(cases: pd.DataFrame, panel: pd.DataFrame, n: int = 6):
    chosen = []
    for _, r in cases.sort_values("score", ascending=False).iterrows():
        if (r["exporter"], r["test_year"]) in [(c["exporter"], c["test_year"]) for c in chosen]:
            continue
        chosen.append(r)
        if len(chosen) >= n:
            break
    rows = []
    for r in chosen:
        T = int(r["test_year"])
        sub = panel[(panel.exporter == r["exporter"]) & (panel.taxon == r["taxon"])]
        for year in range(T - 5, min(T + 1, C.YEAR_MAX) + 1):
            v = sub[sub.year == year]["v_cnt"]
            rows.append(dict(exporter=r["exporter"], taxon=r["taxon"],
                             taxon_group=r["taxon_group"], test_year=T, year=year,
                             offset=year - T, volume=float(v.iloc[0]) if len(v) else 0.0))
    return pd.DataFrame(rows), chosen


def case_series_frame(cases: pd.DataFrame, panel: pd.DataFrame) -> pd.DataFrame:
    """Annual volume and the 4x threshold for T-5..T, per audited case.

    The threshold for year y is four times the mean of the five years before y,
    so the series needs T-11..T.  Missing years are zero trade.
    """
    rows = []
    for _, r in cases.iterrows():
        T = int(r["test_year"])
        sub = panel[(panel.exporter == r["exporter"]) & (panel.taxon == r["taxon"])]
        vv = (sub.set_index("year")["v_cnt"]
              .reindex(range(T - 11, T + 1)).fillna(0.0).to_numpy(float))
        for i, off in enumerate(range(-5, 1)):
            j = 6 + i
            m5 = vv[j - 5:j].mean()
            v = float(vv[j])
            rows.append(dict(
                exporter=r["exporter"], taxon=r["taxon"], test_year=T, offset=off,
                year=T + off, volume=v, threshold=C.SI_MULT * m5,
                fires=bool(v > C.SI_MULT * m5 and m5 >= C.SI_MIN_MEAN),
            ))
    return pd.DataFrame(rows)


def case_flag_matrix(series: pd.DataFrame) -> pd.DataFrame:
    wide = series.pivot_table(index=["exporter", "taxon", "test_year"],
                              columns="offset", values="fires", aggfunc="first")
    out = wide.rename(columns={-2: "si_T2", -1: "si_T1", 0: "si_T"}).reset_index()
    for col in ("si_T2", "si_T1", "si_T"):
        out[col] = out[col].fillna(False).astype(bool)
    out["n_fire"] = out[["si_T2", "si_T1", "si_T"]].sum(axis=1)
    return out


def year_coverage(panel: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for y in range(2005, C.YEAR_MAX):
        g = panel[(panel.year == y) & (panel.v_cnt > 0)]
        rows.append(dict(year=y, active_units=len(g),
                         records=float(g["records"].sum()),
                         volume=float(g["v_cnt"].sum())))
    return pd.DataFrame(rows)


def trigger_split(pool: pd.DataFrame, cases: pd.DataFrame,
                  labels: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Which lag triggers the rule, and how large is the T-2-only stratum."""
    lag = cases["trigger_lags"].fillna("")
    n_t1 = int((lag == "T-1").sum())
    n_both = int((lag == "T-1|T-2").sum())
    n_t2 = int((lag == "T-2").sum())
    if n_t1 + n_both + n_t2 != len(cases):
        raise RuntimeError("trigger_lags has an unexpected value")

    si = labels.set_index(["exporter", "taxon", "year"])["si_c"]
    idx = pd.MultiIndex.from_arrays(
        [pool["exporter"].values, pool["taxon"].values, (pool["test_year"] - 1).values])
    si_t1 = si.reindex(idx).fillna(False).values.astype(bool)
    y3 = pool["y_B3"].astype(bool)
    y2 = pool["y_B"].astype(bool) | si_t1
    if not (y3 >= y2).all():
        raise RuntimeError("labels are not nested")
    by_year = (pd.DataFrame({"test_year": pool["test_year"], "y3": y3, "t2": y3 & ~y2})
               .groupby("test_year").agg(n_pos3=("y3", "sum"), n_t2_only=("t2", "sum"))
               .reset_index())
    n_pos = int(y3.sum())
    n_t2only = int((y3 & ~y2).sum())
    rows = [
        dict(scope="cases", group="T-1 only", n=n_t1),
        dict(scope="cases", group="T-1 and T-2", n=n_both),
        dict(scope="cases", group="T-2 only", n=n_t2),
        dict(scope="cases", group="total", n=len(cases)),
        dict(scope="universe", group="three-year positives", n=n_pos),
        dict(scope="universe", group="T-2 only", n=n_t2only),
    ]
    return by_year, pd.DataFrame(rows)


def run_audit(force: bool = False) -> dict:
    configure_stdout()
    C.ensure_dirs()
    log = Logger(C.LOG_DIR / "audit.log")
    t0 = time.time()
    panel = pd.read_parquet(C.PANEL)
    labels = pd.read_parquet(C.LABELS)
    pool = build_audit_pool(labels, panel)
    write_csv(pool, C.AUDIT_DIR / "pool_strict.csv")

    sets, summary = divergence_sets(pool)
    for name, frame in sets.items():
        write_csv(frame, C.AUDIT_DIR / f"divergence_{name}.csv")
    write_csv(summary, C.RES_DIR / "tables" / "table_divergence.csv")

    cases = sets["pooled_top50"]
    cases = cases[cases["business_code"] == "three_year_window"].copy()
    checked = case_verify(pool, panel, labels, cases)
    write_csv(checked, C.RES_DIR / "tables" / "table_cases.csv")

    yearly, agg, detail = window_comparison(pool, panel, labels)
    write_csv(yearly, C.AUDIT_DIR / "window_by_year.csv")
    write_csv(agg, C.RES_DIR / "tables" / "table_window_comparison.csv")
    write_csv(detail, C.AUDIT_DIR / "window_checked_cases.csv")

    gt, series, level, concentration = gt_sensitivity(pool, panel)
    write_csv(gt, C.RES_DIR / "tables" / "table_gt_sensitivity.csv")
    write_csv(series, C.RES_DIR / "tables" / "table_gt_reporting.csv")
    write_csv(level, C.RES_DIR / "tables" / "table_gt_reporting_level.csv")
    write_csv(concentration, C.RES_DIR / "tables" / "table_concentration.csv")

    imp, raw = import_side(checked)
    write_csv(imp, C.AUDIT_DIR / "import_side.csv")
    matched = import_multifield(raw, checked)
    write_csv(matched, C.RES_DIR / "tables" / "table_import_match.csv")
    strict = matched[matched["paired"]][
        ["test_year", "exporter", "taxon", "group", "paired",
         "origin_eq_any_record", "origin_eq_matched", "origin_missing_share",
         "origin_values", "ratio", "within_2x"]]
    write_csv(strict, C.RES_DIR / "tables" / "table_origin_match_strict.csv")

    by_year, split = trigger_split(pool, checked, labels)
    write_csv(by_year, C.RES_DIR / "tables" / "table_trigger_split_by_year.csv")
    write_csv(split, C.RES_DIR / "tables" / "table_trigger_split.csv")

    timelines, chosen = case_timelines(checked, panel)
    write_csv(timelines, C.RES_DIR / "figure_data" / "case_timelines.csv")
    write_json({"chosen": [dict(exporter=c["exporter"], taxon=c["taxon"],
                                test_year=int(c["test_year"])) for c in chosen]},
               C.AUDIT_DIR / "case_choice.json")

    series = case_series_frame(checked, panel)
    write_csv(series, C.RES_DIR / "figure_data" / "case_series.csv")
    write_csv(case_flag_matrix(series), C.RES_DIR / "figure_data" / "case_flag_matrix.csv")
    write_csv(year_coverage(panel), C.RES_DIR / "figure_data" / "year_coverage.csv")
    composition = pd.concat(
        [frame.assign(selection=name) for name, frame in sets.items()],
        ignore_index=True,
    )
    write_csv(composition, C.RES_DIR / "figure_data" / "divergence_composition.csv")
    write_csv(yearly, C.RES_DIR / "figure_data" / "window_by_year.csv")

    key = dict(
        n_pool=int(len(pool)),
        cases=dict(n=len(checked), multi=int((checked.verdict == "confirmed_multi_base").sum()),
                   single=int((checked.verdict == "confirmed_single_base").sum()),
                   verified=int(checked.verified.sum()),
                   low_trigger_records=int(checked.low_trigger_records.sum()),
                   construction_errors=int((checked.verdict == "construction_error").sum())),
        window=agg.to_dict("records"),
        import_side=dict(
            cases=int((imp.lag == "T-1").sum()),
            with_importer_records=int(((imp.lag == "T-1") & (imp.importer_records > 0)).sum()),
            origin_eq_exporter=int(((imp.lag == "T-1") & (imp.origin_eq_exporter_share > 0)).sum()),
            mean_origin_missing=float(imp[imp.lag == "T-1"].origin_missing_share.mean()),
        ),
        gt=dict(selection=gt.to_dict("records"), level=level.to_dict("records")),
    )
    write_json(key, C.RES_DIR / "paper_numbers.json")
    log(f"cases={len(checked)} verified={int(checked.verified.sum())} "
        f"pool={len(pool):,} elapsed={time.time() - t0:.1f}s")
    log.close()
    return key
