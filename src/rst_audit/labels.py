"""Sharp-increase atoms and the one-year / three-year labels.

Definitions are frozen before modelling:

* unit SI at year t: V(t) > 4 x mean(V(t-5..t-1)) and mean >= 100;
* global SI: the species total satisfies the same test, and the unit holds at
  least 10% of world exports;
* ``y_B``  = unit SI or global-unit SI in the focus year T;
* ``y_B3`` = unit SI in T, T-1 or T-2, or global-unit SI in T.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

from . import config as C
from .io_utils import Logger, configure_stdout, df_fingerprint, sha1_file, write_csv

EPS = 1e-9
YEARS = list(range(C.YEAR_MIN, C.YEAR_MAX + 1))


def roll_sum(matrix: np.ndarray, lo: int, hi: int) -> np.ndarray:
    """Sum of columns t+lo .. t+hi-1 for each column t (hi <= 0)."""
    n, n_years = matrix.shape
    pad = np.zeros((n, n_years + 1), dtype=np.float64)
    pad[:, 1:] = np.cumsum(matrix.astype(np.float64), axis=1)
    out = np.zeros_like(matrix, dtype=np.float64)
    for j in range(n_years):
        a = int(np.clip(j + lo, 0, n_years))
        b = int(np.clip(j + hi, 0, n_years))
        out[:, j] = pad[:, b] - pad[:, a]
    return out


def _unit_matrices(panel: pd.DataFrame):
    units = panel[["exporter", "taxon"]].drop_duplicates().reset_index(drop=True)
    units["uid"] = np.arange(len(units))
    p = units[["uid", "exporter", "taxon"]].merge(panel, on=["exporter", "taxon"], how="left")
    p["yi"] = p["year"] - YEARS[0]
    n_units, n_years = len(units), len(YEARS)

    def mat(col: str) -> np.ndarray:
        out = np.zeros((n_units, n_years), dtype=np.float64)
        ok = p[col].notna().values
        out[p.loc[ok, "uid"].values, p.loc[ok, "yi"].values] = p.loc[ok, col].values
        return out

    return units, p, mat


def build_labels(force: bool = False) -> pd.DataFrame:
    configure_stdout()
    C.ensure_dirs()
    log = Logger(C.LOG_DIR / "build_labels.log")
    t0 = time.time()
    if C.LABELS.exists() and not force:
        log(f"labels exist, reuse: {C.LABELS}")
        return pd.read_parquet(C.LABELS)
    if not C.PANEL.exists():
        raise FileNotFoundError(f"missing panel: {C.PANEL}; run build_panel first")

    panel = pd.read_parquet(C.PANEL)
    units, p, mat = _unit_matrices(panel)
    n_units, n_years = len(units), len(YEARS)

    V = mat("v_cnt")
    M5 = roll_sum(V, -5, 0) / 5.0
    ACT5 = roll_sum((V > 0).astype(float), -5, 0)
    RECENT3 = roll_sum((V > 0).astype(float), -3, 0)

    sp_ids, sp_inv = np.unique(units["taxon"].values, return_inverse=True)
    Vsp = np.zeros((len(sp_ids), n_years))
    np.add.at(Vsp, sp_inv, V)
    Msp = roll_sum(Vsp, -5, 0) / 5.0

    def si_at(mult: float) -> tuple[np.ndarray, np.ndarray]:
        unit = (V > mult * np.maximum(M5, EPS)) & (M5 >= C.SI_MIN_MEAN)
        glob = (Vsp > mult * np.maximum(Msp, EPS)) & (Msp >= C.SI_MIN_MEAN)
        glob_unit = glob[sp_inv] & (V / np.maximum(Vsp[sp_inv], EPS) >= C.GLOBAL_SHARE)
        return unit, glob_unit

    si_unit, si_global_unit = si_at(C.SI_MULT)
    si3 = si_unit.copy()
    si3[:, 2:] |= si_unit[:, 1:-1] | si_unit[:, :-2]

    si3x_unit, _ = si_at(3.0)
    si3x_3y = si3x_unit.copy()
    si3x_3y[:, 2:] |= si3x_unit[:, 1:-1] | si3x_unit[:, :-2]

    # Source shift: one source surges while another declines, with a
    # non-trivial five-year mean.  Used only to classify divergence cases.
    src = {s: mat(f"src_{s}") for s in C.SOURCES}
    src_mean = {s: roll_sum(a, -5, 0) / 5.0 for s, a in src.items()}
    surge = np.zeros((n_units, n_years), dtype=bool)
    decline = np.zeros((n_units, n_years), dtype=bool)
    for s in C.SOURCES:
        surge |= src[s] > 2 * src_mean[s] + EPS
        decline |= src[s] < src_mean[s] - EPS
    source_shift = surge & decline & (M5 >= 50.0)

    # Relaxed reporting discrepancy: importer records with Origin NULL or equal
    # to the exporter.  The strict variant is too sparse to classify cases.
    impn_tot = mat("v_impn_total")
    impn_ok = impn_tot > 0
    repn_src = np.zeros((n_units, n_years), dtype=bool)
    for s in C.SOURCES:
        e, i = src[s], mat(f"impn_{s}")
        rel = np.abs(e - i) / np.maximum(np.maximum(e, i), EPS)
        repn_src |= (rel > 0.10) & (e + i > 20)
    repn_tot = np.abs(V - impn_tot) / np.maximum(np.maximum(V, impn_tot), EPS)
    report_disc_relax = impn_ok & (repn_tot < 0.25) & repn_src

    hv200 = M5 > 200.0
    candidate = (M5 > 0) & (RECENT3 > 0)

    # EN is not used by this paper.  If an IUCN status table is present we keep
    # the column for completeness; otherwise it is zero.
    en = np.zeros((n_units, n_years), dtype=bool)
    en_path = C.DATA_DIR / "iucn_status.csv"
    if en_path.exists():
        status = pd.read_csv(en_path)
        mapping = status.set_index("taxon")["status"]
        en_unit = mapping.reindex(units["taxon"]).isin(["CR", "EN"]).values
        en = np.repeat(en_unit[:, None], n_years, axis=1)

    rows, counts = [], []
    for T in C.LABEL_YEARS:
        j = T - YEARS[0]
        mask = candidate[:, j]
        if not mask.any():
            raise RuntimeError(f"empty candidate universe at {T}")
        sub = units.loc[mask, ["exporter", "taxon"]].copy()
        sub["year"] = T
        sub["v_cnt"] = V[mask, j]
        sub["m5"] = M5[mask, j]
        sub["active5"] = ACT5[mask, j]
        sub["en"] = en[mask, j]
        sub["si_c"] = si_unit[mask, j]
        sub["si3_c"] = si3[mask, j]
        sub["si_g_u"] = si_global_unit[mask, j]
        sub["si_3x"] = si3x_unit[mask, j]
        sub["si3x_3y"] = si3x_3y[mask, j]
        sub["hv200"] = hv200[mask, j]
        sub["source_shift"] = source_shift[mask, j]
        sub["report_disc_relax"] = report_disc_relax[mask, j]
        sub["y_B"] = (sub["si_c"] | sub["si_g_u"]).astype(int)
        sub["y_B3"] = (sub["si3_c"] | sub["si_g_u"]).astype(int)
        rows.append(sub)
        counts.append(dict(
            year=T, candidates=int(mask.sum()), positives=int(sub["y_B"].sum()),
            rate=float(sub["y_B"].mean()), positives_3y=int(sub["y_B3"].sum()),
        ))

    labels = pd.concat(rows, ignore_index=True)
    assert labels.duplicated(["year", "exporter", "taxon"]).sum() == 0
    assert labels["y_B"].any() and labels["y_B3"].any()
    labels.to_parquet(C.LABELS, index=False)
    write_csv(pd.DataFrame(counts), C.RES_DIR / "label_counts.csv")

    log(f"rows={len(labels):,} positives={int(labels['y_B'].sum()):,} "
        f"three-year positives={int(labels['y_B3'].sum()):,}")
    log(f"fingerprint={df_fingerprint(labels)} panel_sha1={sha1_file(C.PANEL)[:12]}")
    log(f"elapsed={time.time() - t0:.1f}s")
    log.close()
    return labels
