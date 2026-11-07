"""Build the exporter x taxon x year panel from the CITES trade table.

Primary volume follows the paper's frozen caliber: exporter-reported direct
exports, count unit (``Number of specimens``), quantity summed.  Import-side
volumes are kept separately for the data-boundary analysis.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

from . import config as C
from .io_utils import Logger, configure_stdout, sha1_file, write_json


def _connect():
    import duckdb

    if not C.CITES_DB.exists():
        raise FileNotFoundError(
            f"CITES database not found: {C.CITES_DB}\n"
            "Download the bulk export from https://trade.cites.org/ and see data/README.md."
        )
    con = duckdb.connect(str(C.CITES_DB), read_only=True)
    con.execute(f"SET threads={C.N_THREADS}")
    return con


def _cached(name: str, fn):
    path = C.DATA_DIR / f"_agg_{name}.parquet"
    if path.exists():
        return pd.read_parquet(path)
    df = fn()
    df.to_parquet(path, index=False)
    return df


def _primary_scope() -> str:
    return (
        f"Year BETWEEN {C.YEAR_MIN} AND {C.YEAR_MAX} AND \"Reporter.type\"='E' "
        "AND (Origin IS NULL OR Origin=Exporter) "
        "AND (Unit IS NULL OR Unit='Number of specimens')"
    )


def build_panel(force: bool = False) -> pd.DataFrame:
    configure_stdout()
    C.ensure_dirs()
    log = Logger(C.LOG_DIR / "build_panel.log")
    t0 = time.time()
    if C.PANEL.exists() and not force:
        log(f"panel exists, reuse: {C.PANEL}")
        return pd.read_parquet(C.PANEL)

    con = _connect()
    scope = _primary_scope()
    n_rows = con.execute("SELECT count(*) FROM trade").fetchone()[0]
    log(f"trade rows all years = {n_rows:,}")

    src_sums = ",\n".join(
        f"sum(CASE WHEN Source='{s}' THEN Quantity ELSE 0 END) AS src_{s}" for s in C.SOURCES
    )
    src_unk = "sum(CASE WHEN Source IS NULL OR Source NOT IN ({}) THEN Quantity ELSE 0 END) AS src_UNK".format(
        ",".join(f"'{s}'" for s in C.SOURCES)
    )
    live_in = ",".join(f"'{t}'" for t in C.LIVE_TERMS)

    def base():
        log("aggregate A: unit-year totals")
        return con.execute(f"""
            SELECT Year AS year, Exporter AS exporter, Taxon AS taxon,
                   sum(Quantity) AS v_cnt,
                   sum(CASE WHEN Source='W' THEN Quantity ELSE 0 END) AS v_wild,
                   sum(CASE WHEN Source='W' AND Term IN ({live_in}) THEN Quantity ELSE 0 END) AS v_wlive,
                   count(*) AS records,
                   count(DISTINCT Importer) AS importers_n,
                   count(DISTINCT Term) AS terms_n,
                   count(DISTINCT Origin) AS origin_n,
                   any_value(Appendix) AS appendix,
                   any_value(Class) AS cls,
                   any_value("Order") AS ord,
                   any_value(Family) AS family,
                   any_value(Genus) AS genus,
                   {src_sums},
                   {src_unk}
            FROM trade WHERE {scope}
            GROUP BY 1, 2, 3
        """).df()

    def import_side_strict():
        log("aggregate B: importer-reported volume (Origin = Exporter)")
        return con.execute(f"""
            SELECT Year AS year, Exporter AS exporter, Taxon AS taxon,
                   coalesce(Source,'UNK') AS src, sum(Quantity) AS v_imp
            FROM trade
            WHERE Year BETWEEN {C.YEAR_MIN} AND {C.YEAR_MAX}
              AND "Reporter.type"='I' AND Origin=Exporter
              AND (Unit IS NULL OR Unit='Number of specimens')
            GROUP BY 1, 2, 3, 4
        """).df()

    def terms():
        log("aggregate C: term-level volume")
        return con.execute(f"""
            SELECT Year AS year, Exporter AS exporter, Taxon AS taxon,
                   coalesce(Term,'UNK') AS term, sum(Quantity) AS v
            FROM trade WHERE {scope}
            GROUP BY 1, 2, 3, 4
        """).df()

    def all_units():
        log("aggregate D: all-unit volume (sensitivity)")
        return con.execute(f"""
            SELECT Year AS year, Exporter AS exporter, Taxon AS taxon,
                   sum(Quantity) AS v_allunit, count(*) AS records_allunit
            FROM trade
            WHERE Year BETWEEN {C.YEAR_MIN} AND {C.YEAR_MAX} AND "Reporter.type"='E'
              AND (Origin IS NULL OR Origin=Exporter)
            GROUP BY 1, 2, 3
        """).df()

    def purposes():
        log("aggregate E: purpose-level volume")
        return con.execute(f"""
            SELECT Year AS year, Exporter AS exporter, Taxon AS taxon,
                   coalesce(Purpose,'UNK') AS purpose, sum(Quantity) AS v
            FROM trade WHERE {scope}
            GROUP BY 1, 2, 3, 4
        """).df()

    def reexports():
        log("aggregate F: re-export volume")
        return con.execute(f"""
            SELECT Year AS year, Exporter AS exporter, Taxon AS taxon,
                   sum(Quantity) AS v_reexport
            FROM trade
            WHERE Year BETWEEN {C.YEAR_MIN} AND {C.YEAR_MAX} AND "Reporter.type"='E'
              AND Origin IS NOT NULL AND Origin <> Exporter
              AND (Unit IS NULL OR Unit='Number of specimens')
            GROUP BY 1, 2, 3
        """).df()

    def importers():
        log("aggregate G: importer-level volume")
        return con.execute(f"""
            SELECT Year AS year, Exporter AS exporter, Taxon AS taxon,
                   coalesce(Importer,'UNK') AS importer, sum(Quantity) AS v
            FROM trade WHERE {scope}
            GROUP BY 1, 2, 3, 4
        """).df()

    def import_side_relaxed():
        log("aggregate H: importer-reported volume (Origin NULL or Exporter)")
        return con.execute(f"""
            SELECT Year AS year, Exporter AS exporter, Taxon AS taxon,
                   coalesce(Source,'UNK') AS src, sum(Quantity) AS v_imp
            FROM trade
            WHERE Year BETWEEN {C.YEAR_MIN} AND {C.YEAR_MAX}
              AND "Reporter.type"='I' AND (Origin IS NULL OR Origin=Exporter)
              AND (Unit IS NULL OR Unit='Number of specimens')
            GROUP BY 1, 2, 3, 4
        """).df()

    base_df = _cached("base", base)
    imp = _cached("imp", import_side_strict)
    term = _cached("term", terms)
    allunit = _cached("allunit", all_units)
    purp = _cached("purp", purposes)
    reexp = _cached("reexp", reexports)
    imp2 = _cached("importer", importers)
    impn = _cached("impn", import_side_relaxed)
    con.close()
    log(f"unit-years={len(base_df):,} volume={base_df['v_cnt'].sum():,.0f}")

    panel = base_df.merge(
        allunit[["year", "exporter", "taxon", "v_allunit", "records_allunit"]],
        on=["year", "exporter", "taxon"], how="left", validate="one_to_one",
    )

    # Term features.
    total = term.groupby(["year", "exporter", "taxon"])["v"].transform("sum")
    term["share"] = term["v"] / total.replace(0, np.nan)
    top = term.loc[term.groupby(["year", "exporter", "taxon"])["v"].idxmax()].rename(
        columns={"term": "top_term"}
    )[["year", "exporter", "taxon", "top_term", "share"]].rename(
        columns={"share": "top_term_share"}
    )
    ent = (
        term.assign(plogp=lambda d: d["share"] * np.log(d["share"].clip(lower=1e-12)))
        .groupby(["year", "exporter", "taxon"])["plogp"].sum().rename("term_entropy").reset_index()
    )
    ent["term_entropy"] = -ent["term_entropy"]
    live = (
        term[term["term"].isin(C.LIVE_TERMS)]
        .groupby(["year", "exporter", "taxon"])["v"].sum().rename("v_live").reset_index()
    )
    panel = panel.merge(top, on=["year", "exporter", "taxon"], how="left")
    panel = panel.merge(ent, on=["year", "exporter", "taxon"], how="left")
    panel = panel.merge(live, on=["year", "exporter", "taxon"], how="left")
    panel["v_live"] = panel["v_live"].fillna(0.0)
    panel["live_share"] = panel["v_live"] / panel["v_cnt"].replace(0, np.nan)

    # Purpose shares.
    ptot = purp.groupby(["year", "exporter", "taxon"])["v"].transform("sum")
    purp["share"] = purp["v"] / ptot.replace(0, np.nan)
    for purpose in ["T", "P", "H", "M", "S", "Q", "Z", "B"]:
        s = (
            purp[purp["purpose"] == purpose]
            .groupby(["year", "exporter", "taxon"])["share"].sum()
            .rename(f"p_{purpose}").reset_index()
        )
        panel = panel.merge(s, on=["year", "exporter", "taxon"], how="left")
        panel[f"p_{purpose}"] = panel[f"p_{purpose}"].fillna(0.0)

    def pivot_sources(frame: pd.DataFrame, prefix: str) -> pd.DataFrame:
        wide = frame.pivot_table(
            index=["year", "exporter", "taxon"], columns="src", values="v_imp", aggfunc="sum"
        ).add_prefix(prefix)
        for s in C.SOURCE_KEYS:
            if f"{prefix}{s}" not in wide.columns:
                wide[f"{prefix}{s}"] = np.nan
        return wide[[f"{prefix}{s}" for s in C.SOURCE_KEYS]].reset_index()

    panel = panel.merge(pivot_sources(imp, "imp_"), on=["year", "exporter", "taxon"], how="left")
    panel["v_imp_total"] = panel[[f"imp_{s}" for s in C.SOURCE_KEYS]].sum(axis=1, min_count=1)
    panel["v_imp_total"] = panel["v_imp_total"].fillna(0.0)

    panel = panel.merge(pivot_sources(impn, "impn_"), on=["year", "exporter", "taxon"], how="left")
    panel["v_impn_total"] = panel[[f"impn_{s}" for s in C.SOURCE_KEYS]].sum(axis=1, min_count=1)
    panel["v_impn_total"] = panel["v_impn_total"].fillna(0.0)

    panel = panel.merge(reexp, on=["year", "exporter", "taxon"], how="left")
    panel["v_reexport"] = panel["v_reexport"].fillna(0.0)

    itot = imp2.groupby(["year", "exporter", "taxon"])["v"].transform("sum")
    imp2["share"] = imp2["v"] / itot.replace(0, np.nan)
    top_importer = (
        imp2.groupby(["year", "exporter", "taxon"])["share"].max()
        .rename("top_importer_share").reset_index()
    )
    panel = panel.merge(top_importer, on=["year", "exporter", "taxon"], how="left")

    panel = panel.sort_values(["exporter", "taxon", "year"]).reset_index(drop=True)
    for col in ["genus", "cls", "ord", "family", "appendix", "top_term"]:
        panel[col] = panel[col].fillna("UNK")

    assert panel.duplicated(["year", "exporter", "taxon"]).sum() == 0
    assert panel["v_cnt"].notna().all() and (panel["v_cnt"] >= 0).all()
    assert panel["year"].between(C.YEAR_MIN, C.YEAR_MAX).all()

    panel.to_parquet(C.PANEL, index=False)
    coverage = (
        panel.groupby("year")[["v_cnt", "v_allunit"]].sum()
        .assign(countunit_share=lambda d: d["v_cnt"] / d["v_allunit"].clip(lower=1e-9))
        ["countunit_share"]
    )
    summary = {
        "panel_rows": int(len(panel)),
        "years": [int(panel["year"].min()), int(panel["year"].max())],
        "n_exporters": int(panel["exporter"].nunique()),
        "n_taxa": int(panel["taxon"].nunique()),
        "v_cnt_total": float(panel["v_cnt"].sum()),
        "v_cnt_quantiles": {
            str(q): float(panel["v_cnt"].quantile(q)) for q in [0.5, 0.9, 0.99, 0.999]
        },
        "rows_per_year": {int(k): int(v) for k, v in panel.groupby("year").size().items()},
        "countunit_share_by_year": {int(k): float(v) for k, v in coverage.items()},
        "panel_sha1": sha1_file(C.PANEL),
    }
    write_json(summary, C.LOG_DIR / "panel_summary.json")
    log(f"panel rows={len(panel):,} -> {C.PANEL}")
    log(f"elapsed={time.time() - t0:.1f}s")
    log.close()
    return panel
