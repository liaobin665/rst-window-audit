"""Machine checks for every headline number in the manuscript.

Each check recomputes a quantity from the shipped aggregate tables and compares
it with a frozen expectation in ``results/checks/expected.json``.  The expected
file is produced once from the analysis environment (``--freeze``) and then
treated as a contract: a passing run means that the tables in this repository
still support the sentences in the paper.

Exit status is non-zero if any check fails.
"""
from __future__ import annotations

import argparse
import json
import sys

import numpy as np
import pandas as pd

from . import config as C
from .io_utils import Logger, configure_stdout, read_csv, read_json, write_csv, write_json

EXPECTED = C.RES_DIR / "checks" / "expected.json"
TOL = 5e-4


def _t(name: str) -> pd.DataFrame:
    return read_csv(C.RES_DIR / "tables" / name)


def _f(name: str) -> pd.DataFrame:
    return read_csv(C.RES_DIR / "figure_data" / name)


def compute() -> dict:
    """Recompute the headline quantities from shipped tables only."""
    cases = _t("table_cases.csv")
    flags = _f("case_flag_matrix.csv")
    div = _t("table_divergence.csv").set_index("selection")
    window = _t("table_window_comparison.csv").set_index("arm")
    split = _t("table_trigger_split.csv").set_index(["scope", "group"])["n"]
    imp = _t("table_import_match.csv")
    conc = _t("table_concentration.csv").set_index("stratum")
    gt = _t("table_gt_sensitivity.csv").set_index("selection")
    thr = _t("table_threshold_sensitivity.csv")
    fam = _t("table_model_family.csv")
    coverage = _f("year_coverage.csv")
    reporting = _t("table_gt_reporting.csv").set_index("year")

    paired = imp[imp["paired"].astype(bool)]
    positives = int(split.loc[("universe", "three-year positives")])
    t2_only = int(split.loc[("universe", "T-2 only")])
    w3 = window.loc["3y_4x"]
    w1 = window.loc["1y_3x"]
    w2 = window.loc["2y_4x"]
    w33 = window.loc["3y_3x"]

    thr4 = thr[thr["threshold"] == 4.0].set_index("arm")
    return {
        "cases": {
            "n": int(len(cases)),
            "multi": int((cases["verdict"] == "confirmed_multi_base").sum()),
            "single": int((cases["verdict"] == "confirmed_single_base").sum()),
            "verified": int(cases["verified"].astype(bool).sum()),
            "low_records": int(cases["low_trigger_records"].astype(bool).sum()),
            "construction_errors": int((cases["verdict"] == "construction_error").sum()),
        },
        "flags": {
            "T-2": int(flags["si_T2"].astype(bool).sum()),
            "T-1": int(flags["si_T1"].astype(bool).sum()),
            "T": int(flags["si_T"].astype(bool).sum()),
        },
        "divergence": {
            "pooled_n": int(div.loc["pooled_top50", "n"]),
            "pooled_defensible": int(div.loc["pooled_top50", "n_defensible"]),
            "per_year_defensible": int(div.loc["per_year_top10", "n_defensible"]),
            "unfiltered_defensible": int(div.loc["unfiltered_top50", "n_defensible"]),
        },
        "trigger_split": {
            "cases_total": int(split.loc[("cases", "total")]),
            "cases_t1_only": int(split.loc[("cases", "T-1 only")]),
            "cases_both": int(split.loc[("cases", "T-1 and T-2")]),
            "universe_positives": positives,
            "universe_t2_only": t2_only,
            "t2_only_share": t2_only / positives,
        },
        "window": {
            "base": float(w3["n_base_flagged"]),
            "three_year_additional": float(w3["n_additional"]),
            "three_year_verified": float(w3["n_additional_verified"]),
            "three_year_after": float(w3["n_after"]),
            "two_year_verified": float(w2["n_additional_verified"]),
            "one_year_verified": float(w1["n_additional_verified"]),
            "one_year_after": float(w1["n_after"]),
            "three_year_3x_verified": float(w33["n_additional_verified"]),
            "three_year_3x_after": float(w33["n_after"]),
            "ratio_verified": float(w3["n_additional_verified"]) / float(w1["n_additional_verified"]),
        },
        "threshold_4x": {
            "one_year_add": float(thr4.loc["1y_4x", "n_add"]),
            "three_year_verified": float(thr4.loc["3y_4x", "n_verified"]),
            "three_year_after": float(thr4.loc["3y_4x", "n_after"]),
            "one_year_after": float(thr4.loc["1y_4x", "n_after"]),
        },
        "import": {
            "n": int(len(imp)),
            "paired": int(imp["paired"].astype(bool).sum()),
            "origin_full_missing": int((paired["origin_missing_share"] == 1.0).sum()),
            "origin_eq_any": int(paired["origin_eq_any_record"].astype(bool).sum()),
            "origin_eq_matched": int(paired["origin_eq_matched"].astype(bool).sum()),
            "within_2x": int(paired["within_2x"].astype(bool).sum()),
            "within_5x": int(paired["within_5x"].astype(bool).sum()),
            "median_ratio": float(paired["ratio"].median()),
        },
        "GT": {
            "share_all": float(conc.loc["all_candidates", "gt_share"]),
            "share_three_year": float(conc.loc["three_year_only", "gt_share"]),
            "share_B3": float(conc.loc["B3_positive", "gt_share"]),
            "defensible_all": int(gt.loc["all", "n_defensible"]),
            "defensible_exclude_GT": int(gt.loc["exclude_GT", "n_defensible"]),
            "defensible_exclude_break": int(gt.loc["exclude_reporting_break", "n_defensible"]),
            "records_2016": float(reporting.loc[2016, "records"]),
            "records_2017": float(reporting.loc[2017, "records"]),
            "active_2016": float(reporting.loc[2016, "active_units"]),
            "active_2017": float(reporting.loc[2017, "active_units"]),
            "records_jump": float(reporting.loc[2017, "records"]
                                  / reporting.loc[2016, "records"]),
            "active_jump": float(reporting.loc[2017, "active_units"]
                                 / reporting.loc[2016, "active_units"]),
        },
        "model_family": {
            "rows": int(len(fam)),
            "pooled_defensible_min": float(
                fam[fam["selection"] == "pooled_top50"]["n_defensible"].min()),
            "pooled_defensible_max": float(
                fam[fam["selection"] == "pooled_top50"]["n_defensible"].max()),
            "three_year_min": int(fam["n_three_year"].min()),
            "three_year_max": int(fam["n_three_year"].max()),
            "three_year_share_min": float((fam["n_three_year"] / fam["n_defensible"]).min()),
            "three_year_share_max": float((fam["n_three_year"] / fam["n_defensible"]).max()),
            "overlap_min": int(fam["overlap_lgb"].min()),
            "overlap_max": int(fam["overlap_lgb"].max()),
        },
        "coverage": {
            "first_year": int(coverage["year"].min()),
            "last_year": int(coverage["year"].max()),
            "max_active_units": int(coverage["active_units"].max()),
            "max_active_year": int(coverage.loc[coverage["active_units"].idxmax(), "year"]),
        },
    }


def _flatten(prefix: str, obj) -> list[tuple[str, float]]:
    out = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            out += _flatten(f"{prefix}.{k}" if prefix else k, v)
    else:
        out.append((prefix, obj))
    return out


def compare(actual: dict, expected: dict, tol: float = TOL) -> list[dict]:
    rows = []
    a = dict(_flatten("", actual))
    e = dict(_flatten("", expected))
    missing = sorted(set(e) - set(a))
    extra = sorted(set(a) - set(e))
    for key in sorted(e):
        if key not in a:
            rows.append(dict(check=key, actual="", expected=e[key], status="MISSING"))
            continue
        av, ev = a[key], e[key]
        if isinstance(ev, (int, float, np.integer, np.floating)):
            ok = abs(float(av) - float(ev)) <= tol
        else:
            ok = str(av) == str(ev)
        rows.append(dict(check=key, actual=av, expected=ev,
                         status="PASS" if ok else "FAIL"))
    for key in extra:
        rows.append(dict(check=key, actual=a[key], expected="", status="UNEXPECTED"))
    return rows


def run_checks(strict: bool = True, freeze: bool = False) -> pd.DataFrame:
    configure_stdout()
    C.ensure_dirs()
    actual = compute()
    if freeze:
        write_json(actual, EXPECTED)
        print(f"froze {len(_flatten('', actual))} expectations -> {EXPECTED}")
        return pd.DataFrame()
    if not EXPECTED.exists():
        raise FileNotFoundError(f"{EXPECTED} missing; run with --freeze once")
    expected = read_json(EXPECTED)
    rows = compare(actual, expected)
    out = pd.DataFrame(rows)
    dest = C.RES_DIR / "checks" / "checks.csv"
    write_csv(out, dest)
    write_json({"checks": rows,
                "failed": int((out["status"] != "PASS").sum()),
                "total": int(len(out))}, C.RES_DIR / "checks" / "checks.json")
    log = Logger(C.LOG_DIR / "checks.log")
    log(f"checks={len(out)} failed={(out['status'] != 'PASS').sum()}")
    log.close()
    print(out[out["status"] != "PASS"].to_string(index=False) if (out["status"] != "PASS").any()
          else f"all {len(out)} checks passed")
    if strict and (out["status"] != "PASS").any():
        raise SystemExit(1)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--freeze", action="store_true", help="write expected.json from current tables")
    ap.add_argument("--no-strict", action="store_true", help="report but exit zero")
    args = ap.parse_args(argv)
    run_checks(strict=not args.no_strict, freeze=args.freeze)
    return 0


if __name__ == "__main__":
    sys.exit(main())
