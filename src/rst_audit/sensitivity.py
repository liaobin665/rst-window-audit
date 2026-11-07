"""Stress tests around the frozen detection-layer definitions.

Three questions are asked, each by changing one thing at a time:

1. Does the window result survive a different sharp-increase multiplier?
2. Does the candidate list survive a different ranking model family?
3. Is the importer-side check sensitive to how records are paired?

None of these tests feeds back into the main specification.  They are reported
as robustness checks, and the main tables always use the frozen 4x rule,
LightGBM, and the strict feature set.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

from . import config as C
from .audit import case_verify, si_tables
from .features import FEATURE_SETS, load_master
from .io_utils import Logger, configure_stdout, write_csv


def _top(frame: pd.DataFrame, by: str, n: int) -> pd.DataFrame:
    return frame.nlargest(n, by)


def threshold_sensitivity(panel: pd.DataFrame, labels: pd.DataFrame,
                          pool: pd.DataFrame, multipliers=(2.0, 3.0, 4.0, 5.0)):
    """Move only the one-year multiplier; model scores stay fixed.

    The three-year arm is rebuilt from the same multiplier (any of T, T-1,
    T-2), so each row answers: with fixed scores, how many *extra* cases would
    a different cut add to the top-100 list, and how many of them pass the
    two-part check?
    """
    base_pool = pool.drop(columns=["si_3x", "si3x_3y"], errors="ignore").copy()
    rows = []
    for k in multipliers:
        col = f"si_{int(k)}x"
        si = si_tables(panel, mult=k, minmean=C.SI_MIN_MEAN,
                       global_share=C.GLOBAL_SHARE)
        si = si[["exporter", "taxon", "year", "si_c"]].rename(columns={"si_c": col})
        p = base_pool.merge(si, on=["exporter", "taxon", "year"], how="left",
                            validate="one_to_one")
        p[col] = p[col].fillna(False).astype(int)
        wide = si.pivot_table(index=["exporter", "taxon"], columns="year",
                              values=col, aggfunc="first").fillna(False)
        parts = []
        for T in C.TEST_YEARS:
            present = [y for y in (T, T - 1, T - 2) if y in wide.columns]
            mask = wide[present].astype(bool).any(axis=1)
            add = mask.rename(f"si3_{int(k)}x").reset_index()
            add["year"] = T
            parts.append(add)
        p = p.merge(pd.concat(parts, ignore_index=True),
                    on=["exporter", "taxon", "year"], how="left", validate="one_to_one")
        p[f"si3_{int(k)}x"] = p[f"si3_{int(k)}x"].fillna(False).astype(int)
        for T in C.TEST_YEARS:
            d = _top(p[p["test_year"] == T], "score", C.TOP_K).copy()
            base = d["y_B"] == 1
            for arm, mask in {
                f"1y_{int(k)}x": (d[col] == 1) & ~base,
                f"3y_{int(k)}x": (d[f"si3_{int(k)}x"] == 1) & ~base,
            }.items():
                add = d[mask].copy()
                if len(add):
                    checked = case_verify(None, panel, labels, add)
                    n_multi = int((checked.verdict == "confirmed_multi_base").sum())
                    n_single = int((checked.verdict == "confirmed_single_base").sum())
                    n_verified = int(checked.verified.sum())
                    n_low = int(checked.low_trigger_records.sum())
                else:
                    n_multi = n_single = n_verified = n_low = 0
                rows.append(dict(
                    threshold=k, arm=arm, test_year=T, n_add=int(mask.sum()),
                    n_multi=n_multi, n_single=n_single, n_low=n_low,
                    n_verified=n_verified, n_after=int(base.sum() + mask.sum()),
                ))
    per_year = pd.DataFrame(rows)
    agg = per_year.groupby(["threshold", "arm"]).agg(
        n_add=("n_add", "mean"), n_multi=("n_multi", "mean"),
        n_single=("n_single", "mean"), n_low=("n_low", "mean"),
        n_verified=("n_verified", "mean"), n_after=("n_after", "mean"),
    ).reset_index()
    write_csv(per_year, C.AUDIT_DIR / "threshold_by_year.csv")
    write_csv(per_year, C.RES_DIR / "tables" / "table_threshold_by_year.csv")
    write_csv(agg, C.RES_DIR / "tables" / "table_threshold_sensitivity.csv")
    return agg


def _fit_family(kind: str, Xtr, ytr, Xte, seed: int):
    if kind == "xgb":
        from xgboost import XGBClassifier

        model = XGBClassifier(
            n_estimators=C.N_ROUNDS, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, tree_method="hist",
            eval_metric="logloss", n_jobs=C.XGB_THREADS, random_state=seed,
        )
    elif kind == "rf":
        from sklearn.ensemble import RandomForestClassifier

        model = RandomForestClassifier(
            n_estimators=C.N_ROUNDS, max_depth=10, min_samples_leaf=5,
            n_jobs=-1, random_state=seed,
            class_weight="balanced_subsample",
        )
    else:
        raise ValueError(kind)
    model.fit(Xtr, ytr)
    return model.predict_proba(Xte)[:, 1]


def model_family(pool: pd.DataFrame, seeds=None):
    """Re-rank the same universe with XGBoost and a random forest."""
    seeds = seeds or C.SEEDS
    master = load_master()
    cols = FEATURE_SETS["no_leak_strict"]
    lgb = pool[pool["y_B"] == 0]
    lgb_pool = lgb[lgb["v_cnt"] > 0].nlargest(C.DIVERGENCE_TOP, "score")
    lgb_keys = set(zip(lgb_pool["exporter"], lgb_pool["taxon"], lgb_pool["test_year"]))
    rows = []
    for kind in ("xgb", "rf"):
        for seed in seeds:
            parts = []
            for T in C.TEST_YEARS:
                tr = master[master["year"] < T]
                te = master[master["year"] == T]
                score = _fit_family(
                    kind,
                    tr[cols].to_numpy(np.float32),
                    tr["y_B"].values.astype(int),
                    te[cols].to_numpy(np.float32),
                    seed,
                )
                parts.append(pd.DataFrame({
                    "exporter": te["exporter"].values,
                    "taxon": te["taxon"].values,
                    "score_new": score,
                    "test_year": T,
                }))
            sc = pd.concat(parts, ignore_index=True)
            p = pool.merge(sc, on=["exporter", "taxon", "test_year"], how="left",
                           validate="one_to_one")
            if p["score_new"].isna().any():
                raise RuntimeError("model score merge loss")
            for selection in ("pooled_top50", "per_year_top10", "unfiltered_top50"):
                base = p[p["y_B"] == 0]
                if selection == "pooled_top50":
                    sel = base[base["v_cnt"] > 0].nlargest(C.DIVERGENCE_TOP, "score_new")
                elif selection == "per_year_top10":
                    sel = (base[base["v_cnt"] > 0]
                           .sort_values(["test_year", "score_new"], ascending=[True, False])
                           .groupby("test_year").head(C.PER_YEAR_TOP))
                else:
                    sel = base.nlargest(C.DIVERGENCE_TOP, "score_new")
                keys = set(zip(sel["exporter"], sel["taxon"], sel["test_year"]))
                rows.append(dict(
                    model=kind, seed=seed, selection=selection, n=len(sel),
                    n_defensible=int(sel["defensible"].sum()),
                    n_three_year=int((sel["business_code"] == "three_year_window").sum()),
                    n_near=int((sel["business_code"] == "near_threshold_4x").sum()),
                    overlap_lgb=len(keys & lgb_keys),
                    plant_share=float((sel["taxon_group"] == "plant").mean()),
                    gt_share=float((sel["exporter"] == "GT").mean()),
                ))
    out = pd.DataFrame(rows)
    write_csv(out, C.RES_DIR / "tables" / "table_model_family.csv")
    return out


def lag_fields() -> int:
    """Confirm the public export carries no submission or update timestamp."""
    import duckdb

    if not C.CITES_DB.exists():
        raise FileNotFoundError(C.CITES_DB)
    con = duckdb.connect(str(C.CITES_DB), read_only=True)
    cols = con.execute("DESCRIBE trade").fetchdf()
    con.close()
    flags = cols["column_name"].str.contains(
        "date|update|submit|batch|version|modified", case=False, regex=True)
    out = cols[["column_name", "column_type"]].copy()
    out["lag_field_candidate"] = flags.values
    write_csv(out, C.RES_DIR / "tables" / "table_report_lag_fields.csv")
    return int(flags.sum())


def diversion_decision() -> pd.DataFrame:
    """Record why trade diversion is outside the detection layer."""
    out = pd.DataFrame([dict(
        task="trade diversion",
        status="not_run",
        reason=("outside the detection layer: the database has no review dates "
                "or compliance outcomes, and a descriptive shift cannot identify "
                "diversion caused by the window"),
    )])
    write_csv(out, C.RES_DIR / "tables" / "table_trade_diversion_decision.csv")
    return out


def run_sensitivity(panel: pd.DataFrame, labels: pd.DataFrame, pool: pd.DataFrame,
                    skip_models: bool = False) -> dict:
    configure_stdout()
    C.ensure_dirs()
    log = Logger(C.LOG_DIR / "sensitivity.log")
    t0 = time.time()
    thresholds = threshold_sensitivity(panel, labels, pool)
    families = None if skip_models else model_family(pool)
    n_lag = lag_fields() if C.CITES_DB.exists() else None
    diversion_decision()
    log(f"thresholds={len(thresholds)} families={0 if families is None else len(families)} "
        f"lag_fields={n_lag} elapsed={time.time() - t0:.1f}s")
    log.close()
    return dict(thresholds=thresholds, families=families, n_lag_fields=n_lag)
